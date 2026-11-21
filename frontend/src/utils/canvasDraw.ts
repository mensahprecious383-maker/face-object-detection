export interface DetectionItem {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

const COLORS = ["#38bdf8", "#10b981", "#f59e0b", "#f43f5e", "#a78bfa", "#2dd4bf"];

export function drawDetections(
  canvas: HTMLCanvasElement,
  detections: DetectionItem[],
  sourceDims: { width: number; height: number },
  inferenceDims: { width: number; height: number },
  imageEl?: HTMLImageElement | null
) {
  if (!sourceDims.width || !sourceDims.height) return;

  if (canvas.width !== sourceDims.width || canvas.height !== sourceDims.height) {
    canvas.width = sourceDims.width;
    canvas.height = sourceDims.height;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (imageEl) {
    ctx.drawImage(imageEl, 0, 0, canvas.width, canvas.height);
  }

  const scaleX = inferenceDims.width > 0 ? canvas.width / inferenceDims.width : 1;
  const scaleY = inferenceDims.height > 0 ? canvas.height / inferenceDims.height : 1;
  const unit = Math.max(1, Math.min(canvas.width, canvas.height) / 640);

  const lineWidth = Math.round(3 * unit);
  const fontSize = Math.round(14 * unit);
  const padX = Math.round(8 * unit);
  const padY = Math.round(4 * unit);

  ctx.font = `700 ${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  ctx.textBaseline = "middle";

  for (const det of detections) {
    const color = COLORS[Math.abs(det.class_id) % COLORS.length];
    const x1 = det.bbox.x1 * scaleX;
    const y1 = det.bbox.y1 * scaleY;
    const w = (det.bbox.x2 - det.bbox.x1) * scaleX;
    const h = (det.bbox.y2 - det.bbox.y1) * scaleY;

    ctx.fillStyle = `${color}22`;
    ctx.fillRect(x1, y1, w, h);

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x1, y1, w, h);

    const label = `${det.class_name.toUpperCase()} ${(det.confidence * 100).toFixed(0)}%`;
    const pillW = ctx.measureText(label).width + padX * 2;
    const pillH = fontSize + padY * 2;
    const pillY = y1 - pillH - 4 * unit > 0 ? y1 - pillH - 4 * unit : y1 + 4 * unit;

    ctx.fillStyle = color;
    ctx.fillRect(x1, pillY, pillW, pillH);

    ctx.fillStyle = "#020617";
    ctx.fillText(label, x1 + padX, pillY + pillH / 2);
  }
}