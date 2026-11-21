import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  Code2,
  KeyRound,
  Pause,
  Play,
  RefreshCw,
  ScanFace,
  SwitchCamera,
  Upload,
} from "lucide-react";
import { drawDetections, type DetectionItem } from "./utils/canvasDraw";

interface DetectResult {
  request_id: string;
  image_dimensions: { width: number; height: number };
  inference_ms: number;
  total_ms: number;
  detections: DetectionItem[];
}

// Uses same-origin relative path so Vite's HTTPS proxy forwards /api to FastAPI on port 8000
const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export default function App() {
  const [mode, setMode] = useState<"camera" | "upload">("camera");
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isAutoDetecting, setIsAutoDetecting] = useState(false);

  const [conf, setConf] = useState(0.25);
  const [showKeyDrawer, setShowKeyDrawer] = useState(false);
  const [customApiKey, setCustomApiKey] = useState("");
  const [customModelUrl, setCustomModelUrl] = useState("");

  const [uploadedImg, setUploadedImg] = useState<HTMLImageElement | null>(null);
  const [uploadedBlob, setUploadedBlob] = useState<Blob | null>(null);

  const [result, setResult] = useState<DetectResult | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showJson, setShowJson] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inFlightRef = useRef(false);

  const stopCamera = useCallback(() => {
    setIsAutoDetecting(false);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCameraOn(false);
  }, []);

  const startCamera = useCallback(
    async (targetFacing: "user" | "environment" = facingMode) => {
      setError(null);
      stopCamera();

      if (!navigator.mediaDevices?.getUserMedia) {
        setError("Camera API unavailable. Ensure you opened the https:// URL.");
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: targetFacing, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setIsCameraOn(true);
      } catch (err) {
        setError((err as Error).message || "Camera permission denied or device not found.");
      }
    },
    [facingMode, stopCamera]
  );

  const flipCamera = () => {
    const next = facingMode === "user" ? "environment" : "user";
    setFacingMode(next);
    if (isCameraOn) startCamera(next);
  };

  const sendFrameForDetection = useCallback(
    async (blob: Blob) => {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      setIsBusy(true);
      setError(null);

      try {
        const form = new FormData();
        form.append("file", blob, "frame.jpg");
        form.append("conf", String(conf));
        if (customApiKey.trim()) form.append("override_api_key", customApiKey.trim());
        if (customModelUrl.trim()) form.append("override_model_url", customModelUrl.trim());

        const res = await fetch(`${API_BASE}/api/v1/detect`, {
          method: "POST",
          body: form,
        });

        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || `Server error (${res.status})`);
        }

        const data: DetectResult = await res.json();
        setResult(data);
      } catch (err) {
        setError((err as Error).message || "Detection request failed.");
        setIsAutoDetecting(false);
      } finally {
        inFlightRef.current = false;
        setIsBusy(false);
      }
    },
    [conf, customApiKey, customModelUrl]
  );

  const captureAndDetectOnce = useCallback(async () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !video.videoWidth) return;

    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement("canvas");
    }
    const c = captureCanvasRef.current;
    c.width = video.videoWidth;
    c.height = video.videoHeight;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, c.width, c.height);

    c.toBlob(
      (blob) => {
        if (blob) sendFrameForDetection(blob);
      },
      "image/jpeg",
      0.85
    );
  }, [sendFrameForDetection]);

  useEffect(() => {
    if (mode !== "camera" || !isCameraOn || !isAutoDetecting) return;
    captureAndDetectOnce();
    const id = window.setInterval(captureAndDetectOnce, 900);
    return () => window.clearInterval(id);
  }, [mode, isCameraOn, isAutoDetecting, captureAndDetectOnce]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (mode === "upload" && uploadedImg) {
      const dims = { width: uploadedImg.naturalWidth, height: uploadedImg.naturalHeight };
      drawDetections(canvas, result?.detections ?? [], dims, result?.image_dimensions ?? dims, uploadedImg);
    } else if (mode === "camera" && isCameraOn && videoRef.current) {
      const dims = {
        width: videoRef.current.videoWidth,
        height: videoRef.current.videoHeight,
      };
      drawDetections(canvas, result?.detections ?? [], dims, result?.image_dimensions ?? dims, null);
    }
  }, [mode, uploadedImg, isCameraOn, result]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setUploadedImg(img);
      setUploadedBlob(file);
      setResult(null);
      sendFrameForDetection(file);
    };
    img.src = url;
  };

  useEffect(() => stopCamera, [stopCamera]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-6xl px-4 py-5 sm:px-6">
        {/* Header */}
        <header className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-slate-900/70 p-4 backdrop-blur">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-sky-500/15 p-2.5 text-sky-400">
              <ScanFace className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-base font-bold text-white sm:text-lg">
                YOLO Face & Object Detection
              </h1>
              <p className="text-xs text-slate-400">Ultralytics Cloud Inference · Web & Mobile</p>
            </div>
          </div>

          <button
            type="button"
            onClick={() => setShowKeyDrawer((p) => !p)}
            className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-medium text-slate-200 hover:bg-white/10"
          >
            <KeyRound className="h-3.5 w-3.5 text-sky-400" />
            {showKeyDrawer ? "Hide API Config" : "API Key Config"}
          </button>
        </header>

        {/* Optional API Key Override Drawer */}
        {showKeyDrawer && (
          <div className="mb-5 grid grid-cols-1 gap-3 rounded-2xl border border-sky-500/30 bg-slate-900/90 p-4 text-xs sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-slate-400">
                Ultralytics API Key (optional override — defaults to backend .env)
              </label>
              <input
                type="password"
                value={customApiKey}
                onChange={(e) => setCustomApiKey(e.target.value)}
                placeholder="Leave blank to use server .env key"
                className="w-full rounded-xl border border-white/10 bg-slate-950 px-3 py-2 font-mono text-slate-100 focus:border-sky-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-slate-400">
                Model URL (optional override)
              </label>
              <input
                type="text"
                value={customModelUrl}
                onChange={(e) => setCustomModelUrl(e.target.value)}
                placeholder="https://hub.ultralytics.com/models/..."
                className="w-full rounded-xl border border-white/10 bg-slate-950 px-3 py-2 font-mono text-slate-100 focus:border-sky-500 focus:outline-none"
              />
            </div>
          </div>
        )}

        {error && (
          <div className="mb-4 flex items-center gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-200">
            <AlertTriangle className="h-4 w-4 shrink-0 text-red-400" />
            <span>{error}</span>
          </div>
        )}

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
          {/* Main Viewport */}
          <div className="flex flex-col gap-4 lg:col-span-8">
            {/* Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-white/10 bg-slate-900/70 p-3">
              <div className="inline-flex rounded-xl border border-white/10 bg-slate-950 p-1">
                <button
                  type="button"
                  onClick={() => {
                    setResult(null);
                    setMode("camera");
                  }}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                    mode === "camera" ? "bg-sky-500 text-slate-950" : "text-slate-300"
                  }`}
                >
                  <Camera className="h-3.5 w-3.5" />
                  Live Camera
                </button>
                <button
                  type="button"
                  onClick={() => {
                    stopCamera();
                    setResult(null);
                    setMode("upload");
                  }}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                    mode === "upload" ? "bg-sky-500 text-slate-950" : "text-slate-300"
                  }`}
                >
                  <Upload className="h-3.5 w-3.5" />
                  Photo / Upload
                </button>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {mode === "camera" ? (
                  !isCameraOn ? (
                    <button
                      type="button"
                      onClick={() => startCamera()}
                      className="inline-flex items-center gap-1.5 rounded-xl bg-emerald-500 px-3.5 py-2 text-xs font-semibold text-slate-950 hover:bg-emerald-400"
                    >
                      <Camera className="h-3.5 w-3.5" />
                      Start Camera
                    </button>
                  ) : (
                    <>
                      <button
                        type="button"
                        onClick={() => setIsAutoDetecting((p) => !p)}
                        className={`inline-flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-semibold ${
                          isAutoDetecting
                            ? "bg-amber-400 text-slate-950"
                            : "bg-sky-500 text-slate-950"
                        }`}
                      >
                        {isAutoDetecting ? (
                          <>
                            <Pause className="h-3.5 w-3.5" /> Pause Live
                          </>
                        ) : (
                          <>
                            <Play className="h-3.5 w-3.5" /> Auto Detect
                          </>
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={captureAndDetectOnce}
                        disabled={isBusy}
                        className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-medium hover:bg-white/10 disabled:opacity-40"
                      >
                        Snap Once
                      </button>
                      <button
                        type="button"
                        onClick={flipCamera}
                        title="Switch Front / Back Camera"
                        className="rounded-xl border border-white/10 bg-white/5 p-2 text-slate-200 hover:bg-white/10"
                      >
                        <SwitchCamera className="h-4 w-4" />
                      </button>
                      <button
                        type="button"
                        onClick={stopCamera}
                        className="rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300"
                      >
                        Stop
                      </button>
                    </>
                  )
                ) : (
                  <>
                    <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-xl bg-sky-500 px-3.5 py-2 text-xs font-semibold text-slate-950 hover:bg-sky-400">
                      <Upload className="h-3.5 w-3.5" />
                      Choose or Take Photo
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="hidden"
                      />
                    </label>
                    {uploadedBlob && (
                      <button
                        type="button"
                        disabled={isBusy}
                        onClick={() => sendFrameForDetection(uploadedBlob)}
                        className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs hover:bg-white/10"
                      >
                        <RefreshCw className={`h-3.5 w-3.5 ${isBusy ? "animate-spin" : ""}`} />
                        Re-run
                      </button>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* Canvas / Video Stage */}
            <div className="relative flex min-h-[340px] items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-slate-900/60 p-2 sm:min-h-[420px]">
              {((mode === "camera" && !isCameraOn) || (mode === "upload" && !uploadedImg)) && (
                <div className="p-6 text-center text-slate-400">
                  <ScanFace className="mx-auto mb-3 h-10 w-10 text-sky-400" />
                  <p className="text-sm font-medium text-slate-200">
                    {mode === "camera"
                      ? "Tap Start Camera to begin live detection"
                      : "Select a photo or take a picture on mobile"}
                  </p>
                </div>
              )}

              <div className={mode === "camera" && isCameraOn ? "relative inline-block max-w-full" : "hidden"}>
                <video
                  ref={videoRef}
                  playsInline
                  muted
                  className="block max-h-[65vh] w-auto max-w-full rounded-xl bg-black"
                />
                <canvas
                  ref={mode === "camera" ? canvasRef : undefined}
                  className="pointer-events-none absolute inset-0 h-full w-full rounded-xl"
                />
              </div>

              {mode === "upload" && uploadedImg && (
                <canvas
                  ref={canvasRef}
                  className="block max-h-[65vh] w-auto max-w-full rounded-xl bg-black"
                />
              )}
            </div>

            {/* Confidence Slider */}
            <div className="flex items-center justify-between gap-4 rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3 text-xs">
              <span className="text-slate-300">
                Confidence Threshold: <strong className="font-mono text-sky-400">{conf.toFixed(2)}</strong>
              </span>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={conf}
                onChange={(e) => setConf(Number(e.target.value))}
                className="w-44 accent-sky-500 sm:w-64"
              />
            </div>
          </div>

          {/* Telemetry & Detections Sidebar */}
          <aside className="flex flex-col gap-3 rounded-2xl border border-white/10 bg-slate-900/70 p-4 lg:col-span-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-white">Detections & Telemetry</h2>
              <button
                type="button"
                onClick={() => setShowJson((p) => !p)}
                className="inline-flex items-center gap-1 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-slate-300"
              >
                <Code2 className="h-3.5 w-3.5" />
                {showJson ? "Cards" : "JSON"}
              </button>
            </div>

            {!result ? (
              <p className="py-10 text-center text-xs text-slate-500">
                No detections yet. Capture a frame or upload an image.
              </p>
            ) : showJson ? (
              <pre className="max-h-[400px] overflow-auto rounded-xl bg-slate-950 p-3 font-mono text-[11px] text-sky-300">
                {JSON.stringify(result, null, 2)}
              </pre>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="rounded-xl bg-slate-950/80 p-3">
                    <div className="text-slate-400">YOLO Inference</div>
                    <div className="mt-0.5 font-mono text-sm font-bold text-sky-400">
                      {result.inference_ms} ms
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-950/80 p-3">
                    <div className="text-slate-400">Total Roundtrip</div>
                    <div className="mt-0.5 font-mono text-sm font-bold text-emerald-400">
                      {result.total_ms} ms
                    </div>
                  </div>
                </div>

                <div className="flex flex-col gap-2">
                  {result.detections.length === 0 ? (
                    <div className="rounded-xl bg-slate-950/60 p-3 text-center text-xs text-slate-400">
                      No faces or objects matched above {(conf * 100).toFixed(0)}% confidence.
                    </div>
                  ) : (
                    result.detections.map((d, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between rounded-xl border border-white/10 bg-slate-950/80 px-3 py-2.5 text-xs"
                      >
                        <span className="font-semibold uppercase text-slate-100">
                          {d.class_name}
                        </span>
                        <span className="font-mono font-semibold text-sky-400">
                          {(d.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}