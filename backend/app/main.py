from collections import defaultdict, deque
import io
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import httpx
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ultralytics_api_url: str = "https://predict.ultralytics.com"
    ultralytics_api_key: str = ""
    ultralytics_model_url: str = ""
    ultralytics_timeout_sec: float = 30.0
    max_image_bytes: int = 8 * 1024 * 1024
    normalize_max_dim: int = 1280


settings = Settings()

app = FastAPI(title="Ultralytics YOLO Face & Object Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_request_log: dict[str, deque[float]] = defaultdict(deque)


def _check_rate_limit(key: str = "global", limit: int = 15, window_sec: int = 10) -> None:
    now = time.monotonic()
    bucket = _request_log[key]
    while bucket and bucket[0] <= now - window_sec:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit reached. Slow down live camera capture slightly.",
            headers={"Retry-After": "3"},
        )
    bucket.append(now)


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class DetectResponse(BaseModel):
    request_id: str
    image_dimensions: dict[str, int]
    inference_ms: float
    total_ms: float
    detections: list[Detection]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "api_url": settings.ultralytics_api_url,
        "api_key_configured": bool(settings.ultralytics_api_key),
    }


@app.post("/api/v1/detect", response_model=DetectResponse)
async def detect_objects(
    file: UploadFile = File(...),
    conf: float = Form(0.25, ge=0.05, le=0.95),
    iou: float = Form(0.45, ge=0.10, le=0.95),
    imgsz: int = Form(640, ge=160, le=1280),
    override_api_key: str = Form(""),
    override_model_url: str = Form(""),
):
    t_start = time.perf_counter()
    _check_rate_limit()

    api_key = override_api_key.strip() or settings.ultralytics_api_key.strip()
    model_or_endpoint = override_model_url.strip() or settings.ultralytics_model_url.strip()

    # If a dedicated *.run.app URL is provided, POST directly to it without a 'model' form field
    target_url = settings.ultralytics_api_url.strip()
    form_data: dict[str, str] = {"imgsz": str(imgsz), "conf": str(conf), "iou": str(iou)}

    if model_or_endpoint:
        if "run.app" in model_or_endpoint or model_or_endpoint.startswith("https://predict-"):
            target_url = model_or_endpoint
        else:
            form_data["model"] = model_or_endpoint

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ultralytics API key missing. Set ULTRALYTICS_API_KEY in backend/.env.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > settings.max_image_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image exceeds 8 MB limit.")

    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes)))
        if img.mode != "RGB":
            img = img.convert("RGB")
        if max(img.width, img.height) > settings.normalize_max_dim:
            img.thumbnail((settings.normalize_max_dim, settings.normalize_max_dim), Image.Resampling.LANCZOS)

        width, height = img.width, img.height
        out_buf = io.BytesIO()
        img.save(out_buf, format="JPEG", quality=88, optimize=True)
        clean_jpeg = out_buf.getvalue()
        img.close()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or unreadable image file.") from exc

    t_infer = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=settings.ultralytics_timeout_sec) as client:
            resp = await client.post(
                target_url,
                headers={"x-api-key": api_key},
                data=form_data,
                files={"file": ("frame.jpg", clean_jpeg, "image/jpeg")},
            )
            # Retry once with Bearer auth if endpoint uses Ultralytics Platform auth scheme
            if resp.status_code == 401:
                resp = await client.post(
                    target_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=form_data,
                    files={"file": ("frame.jpg", clean_jpeg, "image/jpeg")},
                )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Ultralytics endpoint timed out (container may be waking up—try once more).",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Could not reach endpoint: {exc}") from exc

    inference_ms = (time.perf_counter() - t_infer) * 1000.0

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ultralytics API error ({resp.status_code}): {resp.text[:250]}",
        )

    payload = resp.json()
    print("Ultralytics raw response:", payload)  # Helpful debug log in VS Code terminal

    raw_items = []
    if isinstance(payload.get("images"), list) and payload["images"]:
        first = payload["images"][0]
        raw_items = first.get("results", [])
        speed = first.get("speed")
        if isinstance(speed, dict):
            reported = sum(float(speed.get(k, 0.0)) for k in ("preprocess", "inference", "postprocess"))
            if reported > 0:
                inference_ms = reported
    elif isinstance(payload.get("results"), list):
        raw_items = payload["results"]
    elif isinstance(payload.get("predictions"), list):
        raw_items = payload["predictions"]

    detections: list[Detection] = []
    for item in raw_items:
        box = item.get("box") or item.get("bbox") or {}
        if isinstance(box, list) and len(box) >= 4:
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        else:
            x1 = float(box.get("x1", 0.0))
            y1 = float(box.get("y1", 0.0))
            x2 = float(box.get("x2", 0.0))
            y2 = float(box.get("y2", 0.0))

        detections.append(
            Detection(
                class_id=int(item.get("class", item.get("class_id", 0))),
                class_name=str(item.get("name", item.get("class_name", "face"))),
                confidence=round(float(item.get("confidence", item.get("score", 0.0))), 4),
                bbox=BoundingBox(
                    x1=round(x1, 2),
                    y1=round(y1, 2),
                    x2=round(x2, 2),
                    y2=round(y2, 2),
                ),
            )
        )

    total_ms = (time.perf_counter() - t_start) * 1000.0
    return DetectResponse(
        request_id=f"req_{uuid.uuid4().hex[:10]}",
        image_dimensions={"width": width, "height": height},
        inference_ms=round(inference_ms, 2),
        total_ms=round(total_ms, 2),
        detections=detections,
    )