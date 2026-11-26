import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";

interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface Detection {
  id: string;
  label: string;
  confidence: number;
  bbox: BBox;
  cropUrl?: string;
}

interface Telemetry {
  inferenceMs: number;
  roundtripMs: number;
  sourceWidth: number;
  sourceHeight: number;
  timestamp: string;
  endpointUsed: string;
}

interface HistoryEntry {
  id: string;
  timestamp: string;
  mode: "photo" | "camera";
  imageUrl: string;
  detections: Detection[];
  telemetry: Telemetry;
}

type ThemeMode = "light" | "dark";
type ViewMode = "photo" | "camera";
type OverlayStyle = "corners" | "box" | "spotlight";
type InspectorTab = "detections" | "analytics" | "history" | "json";

const ENV = (import.meta as unknown as { env?: Record<string, string | undefined> }).env ?? {};
const DEFAULT_API_URL = (
  ENV.VITE_API_BASE_URL ||
  ENV.VITE_API_URL ||
  "https://face-object-detection-api.onrender.com"
).replace(/\/+$/, "");

const CLASS_COLORS_LIGHT = ["#0284C7", "#0D9488", "#0369A1", "#059669", "#0891B2"];
const CLASS_COLORS_DARK = ["#67B7DC", "#5EC2B2", "#7CC4FA", "#6EE7B7", "#5EEAD4"];

function getClassColor(label: string, theme: ThemeMode): string {
  const palette = theme === "dark" ? CLASS_COLORS_DARK : CLASS_COLORS_LIGHT;
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = (hash * 31 + label.charCodeAt(i)) | 0;
  }
  return palette[Math.abs(hash) % palette.length];
}

function cropFacesFromMedia(
  mediaEl: HTMLImageElement | HTMLVideoElement,
  detections: Detection[]
): Detection[] {
  const srcW = "naturalWidth" in mediaEl ? mediaEl.naturalWidth : mediaEl.videoWidth;
  const srcH = "naturalHeight" in mediaEl ? mediaEl.naturalHeight : mediaEl.videoHeight;
  if (!srcW || !srcH) return detections;

  return detections.map((det) => {
    try {
      const pad = 10;
      const x = Math.max(0, det.bbox.x1 - pad);
      const y = Math.max(0, det.bbox.y1 - pad);
      const w = Math.min(srcW - x, det.bbox.x2 - det.bbox.x1 + pad * 2);
      const h = Math.min(srcH - y, det.bbox.y2 - det.bbox.y1 + pad * 2);
      if (w < 4 || h < 4) return det;

      const c = document.createElement("canvas");
      c.width = 96;
      c.height = 96;
      const ctx = c.getContext("2d");
      if (!ctx) return det;
      ctx.drawImage(mediaEl, x, y, w, h, 0, 0, 96, 96);
      return { ...det, cropUrl: c.toDataURL("image/jpeg", 0.85) };
    } catch {
      return det;
    }
  });
}

function normalizeDetections(
  payload: unknown,
  fallbackW: number,
  fallbackH: number
): { detections: Detection[]; inferenceMs: number } {
  const data = (payload ?? {}) as Record<string, unknown>;
  const rawList =
    (Array.isArray(data.detections) && data.detections) ||
    (Array.isArray(data.predictions) && data.predictions) ||
    (Array.isArray(data.results) && data.results) ||
    (Array.isArray(data.boxes) && data.boxes) ||
    (Array.isArray((data.images as Array<Record<string, unknown>> | undefined)?.[0]?.results) &&
      ((data.images as Array<Record<string, unknown>>)[0].results as unknown[])) ||
    [];

  const telemetryObj = (data.telemetry_ms ?? data.timing ?? {}) as Record<string, unknown>;
  const inferenceMs = Number(
    telemetryObj.detection ??
      telemetryObj.inference ??
      data.inference_ms ??
      data.inferenceMs ??
      data.latency_ms ??
      0
  );

  const parsed: Detection[] = rawList
    .map((item, idx): Detection | null => {
      const obj = (item ?? {}) as Record<string, unknown>;
      const label = String(obj.class_name ?? obj.name ?? obj.label ?? obj.class ?? "FACE");
      const rawConf = Number(obj.confidence ?? obj.score ?? obj.conf ?? 0);
      const confidence = rawConf > 1 ? rawConf / 100 : rawConf;

      let x1 = 0;
      let y1 = 0;
      let x2 = 0;
      let y2 = 0;

      const box = (obj.bbox ?? obj.box ?? obj.xyxy) as unknown;
      if (Array.isArray(box) && box.length >= 4) {
        [x1, y1, x2, y2] = box.map(Number);
      } else if (box && typeof box === "object") {
        const b = box as Record<string, unknown>;
        if ("x1" in b && "x2" in b) {
          x1 = Number(b.x1);
          y1 = Number(b.y1);
          x2 = Number(b.x2);
          y2 = Number(b.y2);
        } else if ("x" in b && "w" in b) {
          x1 = Number(b.x);
          y1 = Number(b.y);
          x2 = x1 + Number(b.w);
          y2 = y1 + Number(b.h);
        }
      } else if ("x1" in obj && "x2" in obj) {
        x1 = Number(obj.x1);
        y1 = Number(obj.y1);
        x2 = Number(obj.x2);
        y2 = Number(obj.y2);
      }

      if (x2 <= 1 && y2 <= 1 && x2 > 0 && y2 > 0) {
        x1 *= fallbackW;
        x2 *= fallbackW;
        y1 *= fallbackH;
        y2 *= fallbackH;
      }

      if (x2 <= x1 || y2 <= y1) return null;

      return {
        id: `${label}-${idx}-${Math.round(x1)}`,
        label,
        confidence: Math.max(0, Math.min(1, confidence)),
        bbox: {
          x1: Math.round(x1),
          y1: Math.round(y1),
          x2: Math.round(x2),
          y2: Math.round(y2),
        },
      };
    })
    .filter((d): d is Detection => d !== null);

  return { detections: parsed, inferenceMs };
}

interface Soft3DCardProps {
  children: React.ReactNode;
  className?: string;
  enabled: boolean;
}

function Soft3DCard({ children, className = "", enabled }: Soft3DCardProps) {
  const [transform, setTransform] = useState(
    "perspective(1100px) rotateX(0deg) rotateY(0deg) translateY(0px)"
  );

  const handleMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!enabled) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    const rotY = Math.max(-0.8, Math.min(0.8, px * 1.6));
    const rotX = Math.max(-0.8, Math.min(0.8, -py * 1.6));
    setTransform(
      `perspective(1100px) rotateX(${rotX.toFixed(2)}deg) rotateY(${rotY.toFixed(2)}deg) translateY(-1px)`
    );
  };

  const handleLeave = () => {
    setTransform("perspective(1100px) rotateX(0deg) rotateY(0deg) translateY(0px)");
  };

  return (
    <div
      className={`soft-3d-card ${className}`}
      style={{ transform: enabled ? transform : undefined }}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
    >
      {children}
    </div>
  );
}

const IconScan = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2" />
    <circle cx="12" cy="10" r="3" />
    <path d="M8 17c1-2 7-2 8 0" />
  </svg>
);
const IconSun = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
  </svg>
);
const IconMoon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79z" />
  </svg>
);
const IconSettings = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c.26.6.85 1 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);
const IconUpload = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
  </svg>
);
const IconCamera = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
    <circle cx="12" cy="13" r="4" />
  </svg>
);
const IconDownload = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
  </svg>
);
const IconArrowUR = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M7 17L17 7M7 7h10v10" />
  </svg>
);
const IconRefresh = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 4v6h-6M1 20v-6h6" />
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
  </svg>
);
const IconCopy = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
);
const IconUsers = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>
);
const IconBolt = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
  </svg>
);
const IconClock = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </svg>
);
const IconShield = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <path d="M9 12l2 2 4-4" />
  </svg>
);

export default function App() {
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const saved = localStorage.getItem("fd_theme");
    return saved === "dark" ? "dark" : "light";
  });
  const [motion3D, setMotion3D] = useState<boolean>(() => {
    return localStorage.getItem("fd_motion3d") !== "off";
  });
  const [mode, setMode] = useState<ViewMode>("photo");
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("detections");

  const [apiBaseUrl, setApiBaseUrl] = useState<string>(() => {
    const saved = (localStorage.getItem("yolo_api_base_url") || "").trim().replace(/\/+$/, "");
    return saved || DEFAULT_API_URL;
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [backendStatus, setBackendStatus] = useState<"ready" | "checking" | "offline">("checking");
  const [healthMessage, setHealthMessage] = useState("");

  // Inference controls
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [imgsz, setImgsz] = useState<number>(640);
  const [overlayStyle, setOverlayStyle] = useState<OverlayStyle>("corners");
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [showCoords, setShowCoords] = useState<boolean>(false);

  // Media & detection state
  const [imageUrl, setImageUrl] = useState<string>("");
  const [lastFile, setLastFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [statusNote, setStatusNote] = useState<string>("");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [allDetections, setAllDetections] = useState<Detection[]>([]);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [latencySeries, setLatencySeries] = useState<number[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [copiedJson, setCopiedJson] = useState(false);

  // Camera state
  const [cameraActive, setCameraActive] = useState(false);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [autoDetect, setAutoDetect] = useState(false);
  const [cameraIntervalMs, setCameraIntervalMs] = useState<number>(1200);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inFlightRef = useRef<boolean>(false);

  // Clear any leftover demo flag and wake Render backend on startup
  useEffect(() => {
    localStorage.removeItem("fd_demo_mode");
    const clean = apiBaseUrl.replace(/\/+$/, "");
    if (!clean) {
      setBackendStatus("offline");
      return;
    }
    setBackendStatus("checking");
    fetch(`${clean}/health`)
      .then((res) => setBackendStatus(res.ok ? "ready" : "offline"))
      .catch(() => setBackendStatus("offline"));
  }, [apiBaseUrl]);

  useEffect(() => {
    localStorage.setItem("fd_theme", theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem("fd_motion3d", motion3D ? "subtle" : "off");
  }, [motion3D]);

  const visibleDetections = useMemo(
    () => allDetections.filter((d) => d.confidence >= confThreshold),
    [allDetections, confThreshold]
  );

  const avgConfidence = useMemo(() => {
    if (!visibleDetections.length) return 0;
    const sum = visibleDetections.reduce((acc, d) => acc + d.confidence, 0);
    return (sum / visibleDetections.length) * 100;
  }, [visibleDetections]);

  const latencyStats = useMemo(() => {
    if (!latencySeries.length) return { avg: 0, min: 0, max: 0, fps: 0 };
    const avg = latencySeries.reduce((a, b) => a + b, 0) / latencySeries.length;
    const min = Math.min(...latencySeries);
    const max = Math.max(...latencySeries);
    return { avg, min, max, fps: avg > 0 ? 1000 / avg : 0 };
  }, [latencySeries]);

  const renderOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const mediaEl = mode === "camera" ? videoRef.current : imgRef.current;
    if (!mediaEl) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const rect = mediaEl.getBoundingClientRect();
    const natW =
      mode === "camera"
        ? (mediaEl as HTMLVideoElement).videoWidth
        : (mediaEl as HTMLImageElement).naturalWidth;
    const natH =
      mode === "camera"
        ? (mediaEl as HTMLVideoElement).videoHeight
        : (mediaEl as HTMLImageElement).naturalHeight;

    if (!rect.width || !rect.height || !natW || !natH) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    if (!showBoxes || visibleDetections.length === 0) return;

    const scale = Math.min(rect.width / natW, rect.height / natH);
    const drawW = natW * scale;
    const drawH = natH * scale;
    const offsetX = (rect.width - drawW) / 2;
    const offsetY = (rect.height - drawH) / 2;

    if (overlayStyle === "spotlight") {
      ctx.save();
      ctx.fillStyle = "rgba(9, 17, 24, 0.48)";
      ctx.fillRect(offsetX, offsetY, drawW, drawH);
      ctx.globalCompositeOperation = "destination-out";
      visibleDetections.forEach((det) => {
        const x = offsetX + det.bbox.x1 * scale;
        const y = offsetY + det.bbox.y1 * scale;
        const w = (det.bbox.x2 - det.bbox.x1) * scale;
        const h = (det.bbox.y2 - det.bbox.y1) * scale;
        ctx.fillRect(x, y, w, h);
      });
      ctx.restore();
    }

    visibleDetections.forEach((det) => {
      const color = getClassColor(det.label, theme);
      const isActive = hoveredId === det.id;
      const x = offsetX + det.bbox.x1 * scale;
      const y = offsetY + det.bbox.y1 * scale;
      const w = (det.bbox.x2 - det.bbox.x1) * scale;
      const h = (det.bbox.y2 - det.bbox.y1) * scale;

      ctx.strokeStyle = color;
      ctx.lineWidth = isActive ? 3 : 2.1;

      if (overlayStyle === "corners") {
        const cLen = Math.max(10, Math.min(24, Math.min(w, h) * 0.25));
        ctx.beginPath();
        ctx.moveTo(x, y + cLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cLen, y);
        ctx.moveTo(x + w - cLen, y);
        ctx.lineTo(x + w, y);
        ctx.lineTo(x + w, y + cLen);
        ctx.moveTo(x + w, y + h - cLen);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x + w - cLen, y + h);
        ctx.moveTo(x + cLen, y + h);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x, y + h - cLen);
        ctx.stroke();

        const cx = x + w / 2;
        const cy = y + h / 2;
        ctx.save();
        ctx.globalAlpha = 0.55;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.moveTo(cx - 5, cy);
        ctx.lineTo(cx + 5, cy);
        ctx.moveTo(cx, cy - 5);
        ctx.lineTo(cx, cy + 5);
        ctx.stroke();
        ctx.restore();
      } else {
        ctx.strokeRect(x, y, w, h);
      }

      if (isActive) {
        ctx.save();
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.14;
        ctx.fillRect(x, y, w, h);
        ctx.restore();
      }

      if (showLabels) {
        const pct = `${(det.confidence * 100).toFixed(1)}%`;
        const text = showCoords
          ? `${det.label} ${pct} [${det.bbox.x1},${det.bbox.y1}]`
          : `${det.label} ${pct}`;
        ctx.font = "600 12px system-ui, -apple-system, sans-serif";
        const tw = ctx.measureText(text).width + 14;
        const th = 22;
        const ly = y - th - 4 >= offsetY ? y - th - 4 : y + 4;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(x, ly, tw, th, 6);
        ctx.fill();

        ctx.fillStyle = theme === "dark" ? "#0B1217" : "#FFFFFF";
        ctx.fillText(text, x + 7, ly + 15);
      }
    });
  }, [mode, visibleDetections, showBoxes, showLabels, showCoords, overlayStyle, hoveredId, theme]);

  useEffect(() => {
    renderOverlay();
    window.addEventListener("resize", renderOverlay);
    return () => window.removeEventListener("resize", renderOverlay);
  }, [renderOverlay, imageUrl, cameraActive]);

  const recordRun = useCallback(
    (
      dets: Detection[],
      tel: Telemetry,
      snapshotUrl: string,
      runMode: "photo" | "camera"
    ) => {
      setAllDetections(dets);
      setTelemetry(tel);
      setLatencySeries((prev) => [...prev.slice(-19), tel.inferenceMs]);
      if (snapshotUrl) {
        setHistory((prev) => [
          {
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            timestamp: tel.timestamp,
            mode: runMode,
            imageUrl: snapshotUrl,
            detections: dets,
            telemetry: tel,
          },
          ...prev.slice(0, 11),
        ]);
      }
    },
    []
  );

  const runInferenceOnBlob = useCallback(
    async (
      blob: Blob,
      filename: string,
      srcW: number,
      srcH: number,
      snapUrl: string,
      runMode: "photo" | "camera"
    ) => {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      setIsAnalyzing(true);
      setErrorMsg("");
      setStatusNote("Sending frame to YOLO26 backend...");

      const t0 = performance.now();
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), 35000);

      try {
        const cleanBase = (apiBaseUrl || DEFAULT_API_URL).replace(/\/+$/, "");
        if (!cleanBase) {
          setBackendStatus("offline");
          setErrorMsg("No backend URL configured. Open Settings and enter your Render URL.");
          return;
        }

        const q = `?conf=${encodeURIComponent(String(Math.min(confThreshold, 0.15)))}&imgsz=${encodeURIComponent(String(imgsz))}`;
        const endpoints = [`${cleanBase}/predict${q}`, `${cleanBase}/api/v1/vision/analyze`];
        let responseJson: unknown = null;
        let usedEndpoint = "";
        let lastErr = "";

        for (const ep of endpoints) {
          try {
            const fd = new FormData();
            fd.append("file", blob, filename);
            fd.append("conf", String(Math.min(confThreshold, 0.15)));
            fd.append("imgsz", String(imgsz));
            if (ep.includes("/vision/analyze")) {
              fd.append("model_id", "yolo26-exp");
              fd.append("verifyBiometrics", "false");
            }
            const res = await fetch(ep, {
              method: "POST",
              body: fd,
              signal: controller.signal,
            });
            if (res.ok) {
              responseJson = await res.json();
              usedEndpoint = ep.replace(cleanBase, "").split("?")[0];
              break;
            } else {
              const errBody = await res.text().catch(() => "");
              lastErr = `HTTP ${res.status}${errBody ? `: ${errBody.slice(0, 120)}` : ""}`;
            }
          } catch (err) {
            if (err instanceof DOMException && err.name === "AbortError") {
              lastErr = "Request timed out after 35s (Render may be waking up—click Snap Once again)";
              break;
            }
            lastErr = err instanceof Error ? err.message : "Network error";
          }
        }

        if (!responseJson) {
          setBackendStatus("offline");
          setErrorMsg(`Inference failed (${lastErr}). Click Settings -> Ping /health to wake Render.`);
          return;
        }

        setBackendStatus("ready");
        const roundtripMs = Number((performance.now() - t0).toFixed(2));
        const { detections, inferenceMs } = normalizeDetections(
          responseJson,
          srcW || 960,
          srcH || 600
        );
        const mediaEl = runMode === "camera" ? videoRef.current : imgRef.current;
        const withCrops = mediaEl ? cropFacesFromMedia(mediaEl, detections) : detections;

        setStatusNote(
          detections.length === 0
            ? "Frame analyzed: 0 faces detected above threshold."
            : `Detected ${detections.length} face${detections.length === 1 ? "" : "s"}.`
        );

        recordRun(
          withCrops,
          {
            inferenceMs: inferenceMs > 0 ? Number(inferenceMs.toFixed(2)) : roundtripMs,
            roundtripMs,
            sourceWidth: srcW || 960,
            sourceHeight: srcH || 600,
            timestamp: new Date().toLocaleTimeString(),
            endpointUsed: usedEndpoint,
          },
          snapUrl,
          runMode
        );
      } finally {
        window.clearTimeout(timeoutId);
        setIsAnalyzing(false);
        setStatusNote("");
        inFlightRef.current = false;
      }
    },
    [apiBaseUrl, confThreshold, imgsz, recordRun]
  );

  const handleFileSelected = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) {
        setErrorMsg("Please select a valid image file (JPG, PNG, or WebP).");
        return;
      }
      setMode("photo");
      setLastFile(file);
      setAllDetections([]);
      setTelemetry(null);
      setHoveredId(null);
      setErrorMsg("");

      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = String(reader.result || "");
        setImageUrl(dataUrl);
        const img = new Image();
        img.onload = () => {
          setTimeout(() => {
            void runInferenceOnBlob(
              file,
              file.name,
              img.naturalWidth,
              img.naturalHeight,
              dataUrl,
              "photo"
            );
          }, 60);
        };
        img.src = dataUrl;
      };
      reader.readAsDataURL(file);
    },
    [runInferenceOnBlob]
  );

  const handleRerun = useCallback(() => {
    if (lastFile && imgRef.current) {
      void runInferenceOnBlob(
        lastFile,
        lastFile.name,
        imgRef.current.naturalWidth || 960,
        imgRef.current.naturalHeight || 600,
        imageUrl,
        "photo"
      );
    }
  }, [imageUrl, lastFile, runInferenceOnBlob]);

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith("image/")) {
          const f = items[i].getAsFile();
          if (f) handleFileSelected(f);
          break;
        }
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [handleFileSelected]);

  const stopCamera = useCallback(() => {
    setAutoDetect(false);
    setCameraActive(false);
    setStatusNote("");
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const startCamera = useCallback(
    async (requestedFacing = facingMode) => {
      stopCamera();
      setAllDetections([]);
      setTelemetry(null);
      setHoveredId(null);
      setErrorMsg("");
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: requestedFacing, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        streamRef.current = stream;
        setCameraActive(true);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch {
        setErrorMsg("Camera access was denied or no webcam was found on this device.");
      }
    },
    [facingMode, stopCamera]
  );

  const captureCameraFrame = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !cameraActive) {
      setErrorMsg("Start the camera first before capturing a frame.");
      return;
    }
    if (inFlightRef.current) return;

    // Wait briefly if metadata/first frame hasn't settled yet
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      setStatusNote("Waiting for camera frame...");
      await new Promise<void>((resolve) => {
        const onReady = () => {
          video.removeEventListener("loadeddata", onReady);
          resolve();
        };
        video.addEventListener("loadeddata", onReady);
        window.setTimeout(() => {
          video.removeEventListener("loadeddata", onReady);
          resolve();
        }, 1200);
      });
    }

    const w = video.videoWidth || video.clientWidth;
    const h = video.videoHeight || video.clientHeight;
    if (!w || !h) {
      setStatusNote("");
      setErrorMsg("Camera frame not ready yet—wait 1 second and click Snap Once again.");
      return;
    }

    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, w, h);
    const snapUrl = c.toDataURL("image/jpeg", 0.88);

    const blob: Blob | null = await new Promise((resolve) =>
      c.toBlob((b) => resolve(b), "image/jpeg", 0.88)
    );

    if (blob) {
      void runInferenceOnBlob(blob, "webcam_frame.jpg", w, h, snapUrl, "camera");
    } else {
      const res = await fetch(snapUrl);
      const fallbackBlob = await res.blob();
      void runInferenceOnBlob(fallbackBlob, "webcam_frame.jpg", w, h, snapUrl, "camera");
    }
  }, [cameraActive, runInferenceOnBlob]);

  useEffect(() => {
    if (!autoDetect || !cameraActive || mode !== "camera") return;
    const id = window.setInterval(() => {
      void captureCameraFrame();
    }, cameraIntervalMs);
    return () => window.clearInterval(id);
  }, [autoDetect, cameraActive, mode, cameraIntervalMs, captureCameraFrame]);

  useEffect(() => {
    if (mode !== "camera") stopCamera();
  }, [mode, stopCamera]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const handleExportPng = () => {
    const mediaEl = mode === "camera" ? videoRef.current : imgRef.current;
    if (!mediaEl) return;
    const natW =
      mode === "camera"
        ? (mediaEl as HTMLVideoElement).videoWidth
        : (mediaEl as HTMLImageElement).naturalWidth;
    const natH =
      mode === "camera"
        ? (mediaEl as HTMLVideoElement).videoHeight
        : (mediaEl as HTMLImageElement).naturalHeight;
    if (!natW || !natH) return;

    const out = document.createElement("canvas");
    out.width = natW;
    out.height = natH;
    const ctx = out.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(mediaEl, 0, 0, natW, natH);

    if (showBoxes && visibleDetections.length) {
      if (overlayStyle === "spotlight") {
        ctx.save();
        ctx.fillStyle = "rgba(9, 17, 24, 0.45)";
        ctx.fillRect(0, 0, natW, natH);
        ctx.globalCompositeOperation = "destination-out";
        visibleDetections.forEach((d) =>
          ctx.fillRect(d.bbox.x1, d.bbox.y1, d.bbox.x2 - d.bbox.x1, d.bbox.y2 - d.bbox.y1)
        );
        ctx.restore();
      }
      visibleDetections.forEach((d) => {
        const color = getClassColor(d.label, theme);
        const w = d.bbox.x2 - d.bbox.x1;
        const h = d.bbox.y2 - d.bbox.y1;
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(3, natW / 320);
        ctx.strokeRect(d.bbox.x1, d.bbox.y1, w, h);
        if (showLabels) {
          const text = `${d.label} ${(d.confidence * 100).toFixed(1)}%`;
          const fs = Math.max(15, Math.round(natW / 55));
          ctx.font = `600 ${fs}px system-ui, sans-serif`;
          const tw = ctx.measureText(text).width + 18;
          const th = fs + 12;
          const ly = Math.max(0, d.bbox.y1 - th - 4);
          ctx.fillStyle = color;
          ctx.fillRect(d.bbox.x1, ly, tw, th);
          ctx.fillStyle = "#FFFFFF";
          ctx.fillText(text, d.bbox.x1 + 9, ly + fs + 3);
        }
      });
    }

    const a = document.createElement("a");
    a.href = out.toDataURL("image/png");
    a.download = `facedetect-${Date.now()}.png`;
    a.click();
  };

  const handleExportCsv = () => {
    const rows = [
      ["label", "confidence", "x1", "y1", "x2", "y2"],
      ...visibleDetections.map((d) => [
        d.label,
        d.confidence.toFixed(4),
        String(d.bbox.x1),
        String(d.bbox.y1),
        String(d.bbox.x2),
        String(d.bbox.y2),
      ]),
    ];
    const csv = rows.map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `detections-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePingHealth = async () => {
    const clean = apiBaseUrl.replace(/\/+$/, "");
    localStorage.setItem("yolo_api_base_url", clean);
    if (!clean) {
      setHealthMessage("Enter your Render backend URL first.");
      return;
    }
    setBackendStatus("checking");
    setHealthMessage("Pinging /health...");
    const t0 = performance.now();
    try {
      const res = await fetch(`${clean}/health`);
      const ms = Math.round(performance.now() - t0);
      if (res.ok) {
        setBackendStatus("ready");
        setHealthMessage(`Connected successfully (${ms} ms). Live backend active.`);
      } else {
        setBackendStatus("offline");
        setHealthMessage(`Server responded with HTTP ${res.status}.`);
      }
    } catch {
      setBackendStatus("offline");
      setHealthMessage("Could not reach /health. Render may be waking up or CORS is blocked.");
    }
  };

  const jsonPayload = useMemo(
    () =>
      JSON.stringify(
        {
          status: "ok",
          threshold: confThreshold,
          imgsz,
          telemetry,
          count: visibleDetections.length,
          detections: visibleDetections.map((d) => ({
            id: d.id,
            label: d.label,
            confidence: d.confidence,
            bbox: d.bbox,
          })),
        },
        null,
        2
      ),
    [confThreshold, imgsz, telemetry, visibleDetections]
  );

  const sparklinePath = useMemo(() => {
    if (latencySeries.length < 2) return "";
    const max = Math.max(...latencySeries, 100);
    const min = Math.min(...latencySeries, 0);
    const span = Math.max(10, max - min);
    return latencySeries
      .map((v, i) => {
        const x = (i / (latencySeries.length - 1)) * 280;
        const y = 62 - ((v - min) / span) * 52;
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [latencySeries]);

  return (
    <div className={`fd-root fd-${theme}`}>
      <style>{`
        html, body, #root {
          margin: 0;
          padding: 0;
          width: 100%;
          max-width: none;
          min-height: 100vh;
          display: block;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
          -webkit-font-smoothing: antialiased;
        }
        * { box-sizing: border-box; }

        .fd-root {
          min-height: 100vh;
          width: 100%;
          transition: background 420ms ease, color 320ms ease;
          padding-bottom: 36px;
        }
        .fd-light {
          --bg: linear-gradient(155deg, #F3FAFC 0%, #F2FAF7 52%, #EDF6F9 100%);
          --glass: rgba(255, 255, 255, 0.78);
          --glass-strong: rgba(255, 255, 255, 0.90);
          --border: rgba(14, 116, 144, 0.14);
          --border-top: rgba(255, 255, 255, 0.95);
          --text: #0F2530;
          --muted: #54717D;
          --accent: #0284C7;
          --accent-2: #0D9488;
          --pill-bg: rgba(226, 244, 248, 0.78);
          --tile-1: linear-gradient(135deg, #0284C7 0%, #0EA5E9 100%);
          --tile-2: linear-gradient(135deg, #0369A1 0%, #0D9488 100%);
          --tile-3: linear-gradient(135deg, #0891B2 0%, #10B981 100%);
          --tile-4: linear-gradient(135deg, #0F766E 0%, #14B8A6 100%);
          --shadow: 0 10px 28px rgba(14, 74, 96, 0.07);
          --stage-bg: rgba(235, 246, 249, 0.65);
          --row-bg: rgba(255, 255, 255, 0.72);
          background: var(--bg);
          color: var(--text);
        }
        .fd-dark {
          --bg: linear-gradient(155deg, #10171D 0%, #121D21 55%, #0E161A 100%);
          --glass: rgba(22, 33, 40, 0.76);
          --glass-strong: rgba(27, 40, 48, 0.92);
          --border: rgba(148, 197, 204, 0.14);
          --border-top: rgba(186, 230, 235, 0.16);
          --text: #E2EEF2;
          --muted: #8FA9B4;
          --accent: #56B4D3;
          --accent-2: #52B7A0;
          --pill-bg: rgba(31, 47, 56, 0.78);
          --tile-1: linear-gradient(135deg, #1A4B66 0%, #24647A 100%);
          --tile-2: linear-gradient(135deg, #1A4E5D 0%, #236B67 100%);
          --tile-3: linear-gradient(135deg, #1B5461 0%, #266E60 100%);
          --tile-4: linear-gradient(135deg, #1C5052 0%, #2B6D62 100%);
          --shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
          --stage-bg: rgba(13, 21, 26, 0.78);
          --row-bg: rgba(29, 43, 52, 0.65);
          background: var(--bg);
          color: var(--text);
        }

        .soft-3d-card {
          transition: transform 520ms cubic-bezier(0.22, 1, 0.36, 1), box-shadow 420ms ease, border-color 300ms ease;
          will-change: transform;
        }

        .fd-nav {
          position: sticky;
          top: 0;
          z-index: 30;
          backdrop-filter: blur(18px);
          -webkit-backdrop-filter: blur(18px);
          background: var(--glass);
          border-bottom: 1px solid var(--border);
        }
        .fd-nav-inner {
          max-width: 1320px;
          margin: 0 auto;
          padding: 12px 20px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
        }
        .fd-brand {
          display: flex;
          align-items: center;
          gap: 10px;
          font-weight: 700;
          font-size: 16px;
          letter-spacing: -0.01em;
        }
        .fd-brand-icon {
          width: 36px;
          height: 36px;
          border-radius: 11px;
          display: grid;
          place-items: center;
          color: #FFFFFF;
          background: linear-gradient(135deg, var(--accent), var(--accent-2));
          box-shadow: 0 4px 12px rgba(2, 132, 199, 0.22);
        }
        .fd-brand-sub {
          display: block;
          font-size: 11px;
          font-weight: 500;
          color: var(--muted);
        }
        .fd-status-pill {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 7px 14px;
          border-radius: 999px;
          font-size: 12.5px;
          font-weight: 600;
          background: var(--pill-bg);
          border: 1px solid var(--border);
          color: var(--text);
        }
        .fd-dot {
          width: 8px;
          height: 8px;
          border-radius: 999px;
          background: #10B981;
        }
        .fd-dot.checking { background: #0EA5E9; }
        .fd-dot.offline { background: #F59E0B; }

        .fd-nav-actions {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .fd-icon-btn {
          height: 36px;
          padding: 0 12px;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: var(--glass-strong);
          color: var(--text);
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-size: 12.5px;
          font-weight: 600;
          cursor: pointer;
          transition: background 200ms ease, border-color 200ms ease;
        }
        .fd-icon-btn:hover { border-color: var(--accent); }

        .fd-main {
          max-width: 1320px;
          margin: 22px auto 0;
          padding: 0 20px;
          display: grid;
          grid-template-columns: minmax(0, 1.48fr) minmax(320px, 1fr);
          gap: 20px;
          align-items: start;
        }
        @media (max-width: 980px) {
          .fd-main { grid-template-columns: 1fr; }
        }

        .fd-panel {
          background: var(--glass);
          backdrop-filter: blur(18px);
          -webkit-backdrop-filter: blur(18px);
          border: 1px solid var(--border);
          border-top: 1px solid var(--border-top);
          border-radius: 20px;
          box-shadow: var(--shadow);
          padding: 18px;
        }

        .fd-mode-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 14px;
          flex-wrap: wrap;
        }
        .fd-seg {
          display: inline-flex;
          padding: 4px;
          border-radius: 13px;
          background: var(--pill-bg);
          border: 1px solid var(--border);
          gap: 4px;
        }
        .fd-seg-btn {
          border: none;
          background: transparent;
          color: var(--muted);
          padding: 7px 14px;
          border-radius: 9px;
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
          display: inline-flex;
          align-items: center;
          gap: 6px;
          transition: all 240ms ease;
        }
        .fd-seg-btn.active {
          background: linear-gradient(135deg, var(--accent), var(--accent-2));
          color: #FFFFFF;
          box-shadow: 0 3px 10px rgba(2, 132, 199, 0.22);
        }

        .fd-hud {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 8px;
          flex-wrap: wrap;
          margin-bottom: 12px;
          padding: 8px 10px;
          border-radius: 12px;
          background: var(--row-bg);
          border: 1px solid var(--border);
        }
        .fd-chip-group {
          display: flex;
          align-items: center;
          gap: 6px;
          flex-wrap: wrap;
        }
        .fd-chip {
          padding: 5px 10px;
          border-radius: 8px;
          border: 1px solid var(--border);
          background: transparent;
          color: var(--muted);
          font-size: 12px;
          font-weight: 600;
          cursor: pointer;
          display: inline-flex;
          align-items: center;
          gap: 5px;
          transition: all 180ms ease;
        }
        .fd-chip.on {
          background: var(--pill-bg);
          color: var(--text);
          border-color: var(--accent);
        }

        .fd-stage {
          position: relative;
          width: 100%;
          aspect-ratio: 16 / 10;
          border-radius: 16px;
          overflow: hidden;
          background: var(--stage-bg);
          border: 1.5px dashed var(--border);
          display: grid;
          place-items: center;
        }
        .fd-stage.has-media { border-style: solid; }
        .fd-media {
          width: 100%;
          height: 100%;
          object-fit: contain;
          display: block;
        }
        .fd-canvas {
          position: absolute;
          inset: 0;
          pointer-events: none;
        }
        .fd-empty {
          text-align: center;
          padding: 28px;
          max-width: 390px;
        }
        .fd-empty-icon {
          width: 56px;
          height: 56px;
          margin: 0 auto 14px;
          border-radius: 16px;
          display: grid;
          place-items: center;
          color: #FFFFFF;
          background: linear-gradient(135deg, var(--accent), var(--accent-2));
        }
        .fd-empty h3 {
          margin: 0 0 6px;
          font-size: 16px;
          font-weight: 700;
        }
        .fd-empty p {
          margin: 0 0 16px;
          font-size: 13px;
          line-height: 1.5;
          color: var(--muted);
        }

        .fd-actions {
          margin-top: 14px;
          display: flex;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
        }
        .fd-btn-primary {
          padding: 10px 16px;
          border-radius: 12px;
          border: none;
          background: linear-gradient(135deg, var(--accent), var(--accent-2));
          color: #FFFFFF;
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
          display: inline-flex;
          align-items: center;
          gap: 7px;
          box-shadow: 0 4px 12px rgba(2, 132, 199, 0.22);
          transition: opacity 180ms ease;
        }
        .fd-btn-primary:hover { opacity: 0.93; }
        .fd-btn-primary:disabled { opacity: 0.55; cursor: not-allowed; }
        .fd-btn-secondary {
          padding: 10px 14px;
          border-radius: 12px;
          border: 1px solid var(--border);
          background: var(--glass-strong);
          color: var(--text);
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
          display: inline-flex;
          align-items: center;
          gap: 7px;
        }
        .fd-btn-secondary:hover { border-color: var(--accent); }
        .fd-btn-secondary:disabled { opacity: 0.55; cursor: not-allowed; }

        .fd-controls {
          margin-top: 14px;
          padding-top: 14px;
          border-top: 1px solid var(--border);
          display: grid;
          grid-template-columns: 1.4fr 1fr;
          gap: 16px;
          align-items: center;
        }
        @media (max-width: 640px) {
          .fd-controls { grid-template-columns: 1fr; }
        }
        .fd-label-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 12.5px;
          font-weight: 600;
          margin-bottom: 6px;
        }
        .fd-slider {
          width: 100%;
          accent-color: var(--accent);
          cursor: pointer;
        }

        .fd-right {
          display: flex;
          flex-direction: column;
          gap: 18px;
        }
        .fd-metrics-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }
        .fd-tile {
          border-radius: 18px;
          padding: 15px 16px;
          color: #FFFFFF;
          box-shadow: var(--shadow);
          position: relative;
          overflow: hidden;
        }
        .fd-tile.t1 { background: var(--tile-1); }
        .fd-tile.t2 { background: var(--tile-2); }
        .fd-tile.t3 { background: var(--tile-3); }
        .fd-tile.t4 { background: var(--tile-4); }
        .fd-tile-top {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .fd-tile-icon {
          width: 32px;
          height: 32px;
          border-radius: 10px;
          background: rgba(255, 255, 255, 0.18);
          display: grid;
          place-items: center;
        }
        .fd-tile-label {
          font-size: 12px;
          opacity: 0.88;
          font-weight: 500;
        }
        .fd-tile-val {
          font-size: 22px;
          font-weight: 700;
          letter-spacing: -0.02em;
          margin: 2px 0;
          font-variant-numeric: tabular-nums;
        }
        .fd-tile-sub {
          font-size: 11px;
          opacity: 0.8;
        }

        .fd-tabs {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 4px;
          padding: 4px;
          border-radius: 12px;
          background: var(--pill-bg);
          border: 1px solid var(--border);
          margin-bottom: 14px;
        }
        .fd-tab {
          border: none;
          background: transparent;
          color: var(--muted);
          padding: 7px 6px;
          border-radius: 9px;
          font-size: 12px;
          font-weight: 600;
          cursor: pointer;
        }
        .fd-tab.active {
          background: var(--glass-strong);
          color: var(--text);
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        }

        .fd-det-list {
          display: flex;
          flex-direction: column;
          gap: 10px;
          max-height: 340px;
          overflow-y: auto;
        }
        .fd-det-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          padding: 10px 12px;
          border-radius: 14px;
          background: var(--row-bg);
          border: 1px solid var(--border);
          cursor: pointer;
        }
        .fd-det-row.hovered { border-color: var(--accent); }
        .fd-det-left {
          display: flex;
          align-items: center;
          gap: 11px;
        }
        .fd-avatar {
          width: 40px;
          height: 40px;
          border-radius: 999px;
          object-fit: cover;
          border: 2px solid var(--accent);
          background: var(--pill-bg);
          display: grid;
          place-items: center;
          font-weight: 700;
          font-size: 13px;
        }
        .fd-det-name {
          font-size: 13.5px;
          font-weight: 700;
        }
        .fd-det-coords {
          font-size: 11px;
          color: var(--muted);
          font-variant-numeric: tabular-nums;
        }
        .fd-det-score {
          font-size: 13px;
          font-weight: 700;
          color: var(--accent);
          font-variant-numeric: tabular-nums;
        }

        .fd-pre {
          margin: 0;
          padding: 12px;
          border-radius: 12px;
          background: var(--stage-bg);
          border: 1px solid var(--border);
          font-size: 11.5px;
          line-height: 1.5;
          max-height: 280px;
          overflow: auto;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }

        .fd-note {
          margin-top: 10px;
          padding: 8px 12px;
          border-radius: 10px;
          font-size: 12.5px;
          background: var(--pill-bg);
          border: 1px solid var(--border);
          color: var(--text);
        }

        .fd-error {
          margin-top: 10px;
          padding: 10px 12px;
          border-radius: 10px;
          font-size: 12.5px;
          background: rgba(245, 158, 11, 0.14);
          border: 1px solid rgba(245, 158, 11, 0.35);
          color: var(--text);
        }

        .fd-modal-bg {
          position: fixed;
          inset: 0;
          z-index: 60;
          background: rgba(7, 15, 22, 0.48);
          backdrop-filter: blur(8px);
          display: grid;
          place-items: center;
          padding: 20px;
        }
        .fd-modal {
          width: 100%;
          max-width: 460px;
          background: var(--glass-strong);
          border: 1px solid var(--border);
          border-radius: 20px;
          padding: 22px;
          box-shadow: 0 22px 48px rgba(0, 0, 0, 0.25);
        }
        .fd-input {
          width: 100%;
          padding: 10px 12px;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: var(--stage-bg);
          color: var(--text);
          font-size: 13px;
          margin-top: 6px;
        }
      `}</style>

      {/* Top Glass Navbar */}
      <header className="fd-nav">
        <div className="fd-nav-inner">
          <div className="fd-brand">
            <div className="fd-brand-icon">
              <IconScan />
            </div>
            <div>
              FaceDetect Studio
              <span className="fd-brand-sub">YOLO26 Cloud Inference</span>
            </div>
          </div>

          <div className="fd-status-pill">
            <span
              className={`fd-dot ${
                backendStatus === "checking" ? "checking" : backendStatus === "offline" ? "offline" : ""
              }`}
            />
            {isAnalyzing
              ? "Running Inference..."
              : backendStatus === "checking"
              ? "Waking Backend..."
              : backendStatus === "offline"
              ? "Backend Offline"
              : "API Connected"}
          </div>

          <div className="fd-nav-actions">
            <button
              className="fd-icon-btn"
              onClick={() => setMotion3D((v) => !v)}
              title="Toggle subtle 3D card tilt"
            >
              3D: {motion3D ? "Subtle" : "Off"}
            </button>
            <button
              className="fd-icon-btn"
              onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
              title="Switch Light / Dark theme"
            >
              {theme === "light" ? <IconMoon /> : <IconSun />}
              {theme === "light" ? "Dark" : "Light"}
            </button>
            <button
              className="fd-icon-btn"
              onClick={() => setSettingsOpen(true)}
              title="Backend API Settings"
            >
              <IconSettings />
              Settings
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="fd-main">
        {/* Left: Unobstructed Detection Stage */}
        <Soft3DCard enabled={motion3D} className="fd-panel">
          <div className="fd-mode-row">
            <div className="fd-seg">
              <button
                className={`fd-seg-btn ${mode === "photo" ? "active" : ""}`}
                onClick={() => {
                  setMode("photo");
                  setAllDetections([]);
                  setTelemetry(null);
                  setHoveredId(null);
                }}
              >
                <IconUpload /> Photo / Upload
              </button>
              <button
                className={`fd-seg-btn ${mode === "camera" ? "active" : ""}`}
                onClick={() => {
                  setMode("camera");
                  setAllDetections([]);
                  setTelemetry(null);
                  setHoveredId(null);
                }}
              >
                <IconCamera /> Live Camera
              </button>
            </div>
          </div>

          {/* Overlay HUD */}
          <div className="fd-hud">
            <div className="fd-chip-group">
              {(["corners", "box", "spotlight"] as OverlayStyle[]).map((st) => (
                <button
                  key={st}
                  className={`fd-chip ${overlayStyle === st ? "on" : ""}`}
                  onClick={() => setOverlayStyle(st)}
                >
                  {st === "corners" ? "Corners" : st === "box" ? "Full Box" : "Spotlight"}
                </button>
              ))}
            </div>
            <div className="fd-chip-group">
              <button className={`fd-chip ${showBoxes ? "on" : ""}`} onClick={() => setShowBoxes((v) => !v)}>
                Boxes
              </button>
              <button className={`fd-chip ${showLabels ? "on" : ""}`} onClick={() => setShowLabels((v) => !v)}>
                Labels
              </button>
              <button className={`fd-chip ${showCoords ? "on" : ""}`} onClick={() => setShowCoords((v) => !v)}>
                Coords
              </button>
              <button
                className="fd-chip"
                onClick={handleExportPng}
                disabled={!imageUrl && !cameraActive}
              >
                <IconDownload /> Export PNG
              </button>
            </div>
          </div>

          {/* Stage Viewport */}
          <div
            className={`fd-stage ${imageUrl || cameraActive ? "has-media" : ""}`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) handleFileSelected(f);
            }}
          >
            {mode === "photo" ? (
              imageUrl ? (
                <img
                  ref={imgRef}
                  src={imageUrl}
                  alt="Detection input"
                  className="fd-media"
                  onLoad={renderOverlay}
                />
              ) : (
                <div className="fd-empty">
                  <div className="fd-empty-icon">
                    <IconScan />
                  </div>
                  <h3>Upload or Drop a Photo</h3>
                  <p>
                    Select a portrait or group photo, drag and drop it here, or paste from your
                    clipboard (Ctrl+V) to run live YOLO26 face detection.
                  </p>
                  <button
                    className="fd-btn-primary"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <IconUpload /> Choose Photo
                  </button>
                </div>
              )
            ) : (
              <>
                <video
                  ref={videoRef}
                  className="fd-media"
                  playsInline
                  muted
                  onLoadedMetadata={renderOverlay}
                  onLoadedData={renderOverlay}
                  style={{ display: cameraActive ? "block" : "none" }}
                />
                {!cameraActive && (
                  <div className="fd-empty">
                    <div className="fd-empty-icon">
                      <IconCamera />
                    </div>
                    <h3>Live Camera Detection</h3>
                    <p>Start your webcam to run continuous or single-frame YOLO26 detection.</p>
                    <button className="fd-btn-primary" onClick={() => void startCamera()}>
                      <IconCamera /> Start Camera
                    </button>
                  </div>
                )}
              </>
            )}

            <canvas ref={canvasRef} className="fd-canvas" />
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFileSelected(f);
              e.target.value = "";
            }}
          />

          {/* Action Buttons */}
          <div className="fd-actions">
            {mode === "photo" ? (
              <>
                <button className="fd-btn-primary" onClick={() => fileInputRef.current?.click()}>
                  <IconUpload /> {imageUrl ? "Change Photo" : "Upload Photo"}
                </button>
                <button
                  className="fd-btn-secondary"
                  onClick={handleRerun}
                  disabled={!lastFile || isAnalyzing}
                >
                  <IconRefresh /> Re-run Detection
                </button>
                {imageUrl && (
                  <button
                    className="fd-btn-secondary"
                    onClick={() => {
                      setImageUrl("");
                      setAllDetections([]);
                      setLastFile(null);
                      setTelemetry(null);
                    }}
                  >
                    Clear
                  </button>
                )}
              </>
            ) : (
              <>
                {!cameraActive ? (
                  <button className="fd-btn-primary" onClick={() => void startCamera()}>
                    <IconCamera /> Start Camera
                  </button>
                ) : (
                  <>
                    <button
                      className="fd-btn-primary"
                      onClick={() => setAutoDetect((v) => !v)}
                    >
                      {autoDetect ? "Pause Auto-Detect" : "Auto Detect"}
                    </button>
                    <button
                      className="fd-btn-secondary"
                      onClick={() => void captureCameraFrame()}
                      disabled={isAnalyzing}
                    >
                      {isAnalyzing ? "Analyzing..." : "Snap Once"}
                    </button>
                    <button
                      className="fd-btn-secondary"
                      onClick={() => {
                        const next = facingMode === "user" ? "environment" : "user";
                        setFacingMode(next);
                        void startCamera(next);
                      }}
                    >
                      Flip Camera
                    </button>
                    <button className="fd-btn-secondary" onClick={stopCamera}>
                      Stop
                    </button>
                    <div className="fd-chip-group">
                      {[800, 1200, 2000].map((ms) => (
                        <button
                          key={ms}
                          className={`fd-chip ${cameraIntervalMs === ms ? "on" : ""}`}
                          onClick={() => setCameraIntervalMs(ms)}
                        >
                          {ms}ms
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </>
            )}
          </div>

          {statusNote && <div className="fd-note">{statusNote}</div>}
          {errorMsg && <div className="fd-error">{errorMsg}</div>}

          {/* Confidence & Resolution Controls */}
          <div className="fd-controls">
            <div>
              <div className="fd-label-row">
                <span>Confidence Threshold: {confThreshold.toFixed(2)}</span>
                <div className="fd-chip-group">
                  {[
                    { v: 0.15, t: "0.15 Recall" },
                    { v: 0.25, t: "0.25 Balanced" },
                    { v: 0.5, t: "0.50 Strict" },
                  ].map((p) => (
                    <button
                      key={p.v}
                      className={`fd-chip ${Math.abs(confThreshold - p.v) < 0.005 ? "on" : ""}`}
                      onClick={() => setConfThreshold(p.v)}
                    >
                      {p.t}
                    </button>
                  ))}
                </div>
              </div>
              <input
                type="range"
                min={0.05}
                max={0.95}
                step={0.01}
                value={confThreshold}
                onChange={(e) => setConfThreshold(Number(e.target.value))}
                className="fd-slider"
              />
            </div>

            <div>
              <div className="fd-label-row">
                <span>Input Size (imgsz)</span>
                <span style={{ color: "var(--muted)" }}>{imgsz}px</span>
              </div>
              <div className="fd-chip-group">
                {[320, 640, 1280].map((sz) => (
                  <button
                    key={sz}
                    className={`fd-chip ${imgsz === sz ? "on" : ""}`}
                    onClick={() => setImgsz(sz)}
                  >
                    {sz}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </Soft3DCard>

        {/* Right: 2x2 Metric Cards + 4-Tab Inspector */}
        <div className="fd-right">
          <div className="fd-metrics-grid">
            <Soft3DCard enabled={motion3D} className="fd-tile t1">
              <div className="fd-tile-top">
                <div className="fd-tile-icon"><IconUsers /></div>
                <IconArrowUR />
              </div>
              <div className="fd-tile-label">Faces Found</div>
              <div className="fd-tile-val">{visibleDetections.length}</div>
              <div className="fd-tile-sub">Above {confThreshold.toFixed(2)} conf</div>
            </Soft3DCard>

            <Soft3DCard enabled={motion3D} className="fd-tile t2">
              <div className="fd-tile-top">
                <div className="fd-tile-icon"><IconBolt /></div>
                <IconArrowUR />
              </div>
              <div className="fd-tile-label">YOLO Inference</div>
              <div className="fd-tile-val">
                {telemetry ? `${telemetry.inferenceMs.toFixed(1)} ms` : "—"}
              </div>
              <div className="fd-tile-sub">{telemetry?.endpointUsed || "Waiting for frame"}</div>
            </Soft3DCard>

            <Soft3DCard enabled={motion3D} className="fd-tile t3">
              <div className="fd-tile-top">
                <div className="fd-tile-icon"><IconClock /></div>
                <IconArrowUR />
              </div>
              <div className="fd-tile-label">Total Roundtrip</div>
              <div className="fd-tile-val">
                {telemetry ? `${telemetry.roundtripMs.toFixed(0)} ms` : "—"}
              </div>
              <div className="fd-tile-sub">End-to-end latency</div>
            </Soft3DCard>

            <Soft3DCard enabled={motion3D} className="fd-tile t4">
              <div className="fd-tile-top">
                <div className="fd-tile-icon"><IconShield /></div>
                <IconArrowUR />
              </div>
              <div className="fd-tile-label">Avg Confidence</div>
              <div className="fd-tile-val">
                {visibleDetections.length ? `${avgConfidence.toFixed(1)}%` : "—"}
              </div>
              <div className="fd-tile-sub">Current active frame</div>
            </Soft3DCard>
          </div>

          {/* 4-Tab Inspector */}
          <Soft3DCard enabled={motion3D} className="fd-panel">
            <div className="fd-tabs">
              {(
                [
                  { id: "detections", label: `Faces (${visibleDetections.length})` },
                  { id: "analytics", label: "Analytics" },
                  { id: "history", label: `History (${history.length})` },
                  { id: "json", label: "JSON" },
                ] as const
              ).map((t) => (
                <button
                  key={t.id}
                  className={`fd-tab ${inspectorTab === t.id ? "active" : ""}`}
                  onClick={() => setInspectorTab(t.id)}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {inspectorTab === "detections" && (
              <div className="fd-det-list">
                {visibleDetections.length === 0 ? (
                  <p style={{ fontSize: 13, color: "var(--muted)", textAlign: "center", padding: "24px 0" }}>
                    No detections above {confThreshold.toFixed(2)}. Upload a photo or capture from Live Camera.
                  </p>
                ) : (
                  visibleDetections.map((det) => (
                    <div
                      key={det.id}
                      className={`fd-det-row ${hoveredId === det.id ? "hovered" : ""}`}
                      onMouseEnter={() => setHoveredId(det.id)}
                      onMouseLeave={() => setHoveredId(null)}
                    >
                      <div className="fd-det-left">
                        {det.cropUrl ? (
                          <img src={det.cropUrl} alt={det.label} className="fd-avatar" />
                        ) : (
                          <div className="fd-avatar">{det.label.slice(0, 2).toUpperCase()}</div>
                        )}
                        <div>
                          <div className="fd-det-name">{det.label}</div>
                          <div className="fd-det-coords">
                            [{det.bbox.x1}, {det.bbox.y1}, {det.bbox.x2}, {det.bbox.y2}]
                          </div>
                        </div>
                      </div>
                      <div className="fd-det-score">{(det.confidence * 100).toFixed(1)}%</div>
                    </div>
                  ))
                )}
              </div>
            )}

            {inspectorTab === "analytics" && (
              <div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginBottom: 14 }}>
                  <div className="fd-det-row" style={{ flexDirection: "column", alignItems: "flex-start", gap: 2 }}>
                    <span className="fd-det-coords">Avg Latency</span>
                    <strong>{latencyStats.avg ? `${latencyStats.avg.toFixed(1)} ms` : "—"}</strong>
                  </div>
                  <div className="fd-det-row" style={{ flexDirection: "column", alignItems: "flex-start", gap: 2 }}>
                    <span className="fd-det-coords">Min / Max</span>
                    <strong>
                      {latencyStats.avg
                        ? `${latencyStats.min.toFixed(0)} / ${latencyStats.max.toFixed(0)}`
                        : "—"}
                    </strong>
                  </div>
                  <div className="fd-det-row" style={{ flexDirection: "column", alignItems: "flex-start", gap: 2 }}>
                    <span className="fd-det-coords">Est. FPS</span>
                    <strong>{latencyStats.fps ? latencyStats.fps.toFixed(1) : "—"}</strong>
                  </div>
                </div>
                <div className="fd-det-row" style={{ flexDirection: "column", alignItems: "stretch" }}>
                  <span className="fd-det-coords" style={{ marginBottom: 6 }}>
                    Inference Latency Trend (last {latencySeries.length} runs)
                  </span>
                  {latencySeries.length >= 2 ? (
                    <svg viewBox="0 0 280 70" style={{ width: "100%", height: 74 }}>
                      <path
                        d={sparklinePath}
                        fill="none"
                        stroke="var(--accent)"
                        strokeWidth="2.4"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  ) : (
                    <p style={{ margin: "12px 0", fontSize: 12, color: "var(--muted)" }}>
                      Run at least 2 detections to plot the latency sparkline.
                    </p>
                  )}
                </div>
              </div>
            )}

            {inspectorTab === "history" && (
              <div className="fd-det-list">
                {history.length === 0 ? (
                  <p style={{ fontSize: 13, color: "var(--muted)", textAlign: "center", padding: "24px 0" }}>
                    Completed detection snapshots will appear here.
                  </p>
                ) : (
                  history.map((item) => (
                    <div
                      key={item.id}
                      className="fd-det-row"
                      onClick={() => {
                        setMode("photo");
                        setImageUrl(item.imageUrl);
                        setAllDetections(item.detections);
                        setTelemetry(item.telemetry);
                      }}
                    >
                      <div className="fd-det-left">
                        <img src={item.imageUrl} alt="Snapshot" className="fd-avatar" />
                        <div>
                          <div className="fd-det-name">
                            {item.detections.length} face{item.detections.length === 1 ? "" : "s"} ({item.mode})
                          </div>
                          <div className="fd-det-coords">
                            {item.timestamp} · {item.telemetry.inferenceMs.toFixed(1)} ms
                          </div>
                        </div>
                      </div>
                      <IconArrowUR />
                    </div>
                  ))
                )}
              </div>
            )}

            {inspectorTab === "json" && (
              <div>
                <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
                  <button
                    className="fd-chip on"
                    onClick={() => {
                      void navigator.clipboard.writeText(jsonPayload);
                      setCopiedJson(true);
                      setTimeout(() => setCopiedJson(false), 1600);
                    }}
                  >
                    <IconCopy /> {copiedJson ? "Copied!" : "Copy JSON"}
                  </button>
                  <button
                    className="fd-chip"
                    onClick={handleExportCsv}
                    disabled={!visibleDetections.length}
                  >
                    <IconDownload /> Export CSV
                  </button>
                </div>
                <pre className="fd-pre">{jsonPayload}</pre>
              </div>
            )}
          </Soft3DCard>
        </div>
      </main>

      {/* Backend Connection Settings Modal */}
      {settingsOpen && (
        <div className="fd-modal-bg" onClick={() => setSettingsOpen(false)}>
          <div className="fd-modal" onClick={(e) => e.stopPropagation()}>
            <h3 style={{ margin: "0 0 6px", fontSize: 17 }}>Backend Connection Settings</h3>
            <p style={{ margin: "0 0 14px", fontSize: 12.5, color: "var(--muted)" }}>
              Connect to your Render FastAPI proxy (`/predict` or `/api/v1/vision/analyze`).
            </p>

            <label style={{ fontSize: 12.5, fontWeight: 600 }}>
              FastAPI Backend Base URL
              <input
                type="url"
                className="fd-input"
                placeholder="https://face-object-detection-api.onrender.com"
                value={apiBaseUrl}
                onChange={(e) => setApiBaseUrl(e.target.value)}
              />
            </label>

            {healthMessage && (
              <div className="fd-error" style={{ marginTop: 12 }}>
                {healthMessage}
              </div>
            )}

            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 18 }}>
              <button className="fd-btn-secondary" onClick={() => void handlePingHealth()}>
                Ping /health
              </button>
              <button
                className="fd-btn-primary"
                onClick={() => {
                  localStorage.setItem("yolo_api_base_url", apiBaseUrl.replace(/\/+$/, ""));
                  setSettingsOpen(false);
                }}
              >
                Save & Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}