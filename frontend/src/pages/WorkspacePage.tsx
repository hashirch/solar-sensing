import { ChangeEvent, useRef, useState } from "react";
import {
  Cpu,
  Check,
  CloudUpload,
  Play,
  X,
  ArrowRight,
  RotateCcw,
  Shield,
  Eye,
  Sliders,
  Sparkles,
  Download,
  Maximize2,
  Layers,
  Zap,
} from "lucide-react";
import { runSegmentationAnalysis, AnalysisOutput } from "../utils/inference";
import ImageComparisonModal from "../components/ImageComparisonModal";

type ModelId = "cnn" | "rcnn";

interface UploadItem {
  id: string;
  name: string;
  file?: File;
  url: string;
  sizeBytes: number;
  status: "queued" | "analyzing" | "ready" | "error";
  output?: AnalysisOutput;
  errorMessage?: string;
}

const models = [
  {
    id: "cnn" as ModelId,
    name: "CNN / U-Net",
    detail: "Dense surface area masks",
    accent: "green" as const,
    architecture: "ResNet-34 Encoder + U-Net Decoder",
  },
  {
    id: "rcnn" as ModelId,
    name: "Mask R-CNN",
    detail: "Instance detections & counts",
    accent: "gold" as const,
    architecture: "ResNet-50-FPN + ROI Align",
  },
];

const sampleImages = [
  {
    name: "Rooftop Array 01",
    path: "/samples/satellite_field_1.jpg",
    description: "High-density residential rooftops",
  },
  {
    name: "Commercial Plant 02",
    path: "/samples/satellite_field_2.jpg",
    description: "Industrial warehouse installation",
  },
  {
    name: "Ground Mount 03",
    path: "/samples/satellite_field_3.jpg",
    description: "Rural utility solar farm",
  },
];

export default function WorkspacePage() {
  const [selectedModels, setSelectedModels] = useState<ModelId[]>([
    "cnn",
    "rcnn",
  ]);
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [activeModalItem, setActiveModalItem] = useState<UploadItem | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  function addFiles(files: FileList | File[]) {
    const accepted = Array.from(files).filter((f) =>
      f.type.startsWith("image/")
    );
    const newItems: UploadItem[] = accepted.map((file) => ({
      id: `${file.name}-${file.lastModified}-${Math.random()}`,
      name: file.name,
      file,
      url: URL.createObjectURL(file),
      sizeBytes: file.size,
      status: "queued",
    }));
    setUploads((cur) => [...cur, ...newItems]);
  }

  function loadSampleImage(sample: (typeof sampleImages)[0]) {
    // Check if already in queue
    if (uploads.some((u) => u.name === sample.name)) return;

    const newItem: UploadItem = {
      id: `sample-${sample.name}-${Date.now()}`,
      name: sample.name,
      url: sample.path,
      sizeBytes: 320 * 1024,
      status: "queued",
    };
    setUploads((cur) => [...cur, newItem]);
  }

  function handleInput(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files) addFiles(e.target.files);
    e.target.value = "";
  }

  function toggleModel(id: ModelId) {
    setSelectedModels((cur) =>
      cur.includes(id) ? cur.filter((m) => m !== id) : [...cur, id]
    );
  }

  async function runAnalysis() {
    if (!uploads.length || !selectedModels.length || isRunning) return;
    setIsRunning(true);

    // Set all queued items to analyzing
    setUploads((cur) =>
      cur.map((u) => (u.status === "queued" ? { ...u, status: "analyzing" } : u))
    );

    // Process each image through the segmentation pipeline
    for (const item of uploads) {
      if (item.status === "ready") continue;

      try {
        const output = await runSegmentationAnalysis(item.url, item.name);
        setUploads((cur) =>
          cur.map((u) =>
            u.id === item.id ? { ...u, status: "ready", output } : u
          )
        );
      } catch (err: any) {
        setUploads((cur) =>
          cur.map((u) =>
            u.id === item.id
              ? {
                  ...u,
                  status: "error",
                  errorMessage: err?.message || "Inference failed",
                }
              : u
          )
        );
      }
    }

    setIsRunning(false);
  }

  function removeUpload(id: string) {
    setUploads((cur) => {
      const item = cur.find((u) => u.id === id);
      if (item && item.file) URL.revokeObjectURL(item.url);
      return cur.filter((u) => u.id !== id);
    });
  }

  function clearAll() {
    uploads.forEach((u) => {
      if (u.file) URL.revokeObjectURL(u.url);
    });
    setUploads([]);
  }

  function formatBytes(bytes: number) {
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  }

  const readyUploads = uploads.filter((u) => u.status === "ready" && u.output);
  const totalAreaEstimated = Math.round(
    readyUploads.reduce(
      (sum, u) => sum + (u.output?.metrics.totalAreaM2 || 0),
      0
    ) * 10
  ) / 10;

  return (
    <div className="workspace">
      {/* ─── Page header ─── */}
      <div className="page-header">
        <div>
          <span className="kicker">Inference Workspace</span>
          <h1>Satellite Image Segmentation</h1>
          <p>
            Upload satellite photos or test with benchmark samples to run dual-engine
            solar panel detection and surface area estimation.
          </p>
        </div>
      </div>

      <div className="workspace-layout single">
        {/* ─── Left: Main controls & workflow ─── */}
        <div className="workspace-main">
          {/* Step 1: Model Selection */}
          <section className="ws-section">
            <div className="ws-section-header">
              <div>
                <span className="step-badge">Step 1</span>
                <h2>Active Neural Engines</h2>
              </div>
              <span className="count-badge">
                {selectedModels.length} of {models.length} active
              </span>
            </div>
            <div className="model-select-grid">
              {models.map((model) => {
                const selected = selectedModels.includes(model.id);
                return (
                  <button
                    className={`model-card ${selected ? "selected" : ""}`}
                    key={model.id}
                    onClick={() => toggleModel(model.id)}
                    aria-pressed={selected}
                  >
                    <div className={`model-card-icon ${model.accent}`}>
                      <Cpu size={22} />
                    </div>
                    <div className="model-card-text">
                      <strong>{model.name}</strong>
                      <span>{model.detail}</span>
                      <small className="model-arch">{model.architecture}</small>
                    </div>
                    <span className={`check-circle ${selected ? "checked" : ""}`}>
                      {selected && <Check size={12} />}
                    </span>
                  </button>
                );
              })}
            </div>
          </section>

          {/* Step 2: Input / Upload */}
          <section className="ws-section">
            <div className="ws-section-header">
              <div>
                <span className="step-badge">Step 2</span>
                <h2>Imagery Source</h2>
              </div>
              <span className="file-hint">JPG, PNG, WEBP · Max 50 MB</span>
            </div>

            {/* Drag & Drop Area */}
            <div
              className={`drop-zone ${isDragging ? "active" : ""}`}
              role="button"
              tabIndex={0}
              onClick={() => fileInput.current?.click()}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ")
                  fileInput.current?.click();
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                addFiles(e.dataTransfer.files);
              }}
            >
              <input
                ref={fileInput}
                type="file"
                accept="image/*"
                multiple
                onChange={handleInput}
                hidden
              />
              <div className="drop-zone-icon">
                <CloudUpload size={28} />
              </div>
              <strong>Drop satellite imagery here</strong>
              <span>
                or <u>browse files from your computer</u>
              </span>
            </div>

            {/* Sample Imagery Selector */}
            <div className="sample-strip">
              <span className="sample-strip-label">
                <Sparkles size={14} /> Or try benchmark scenes:
              </span>
              <div className="sample-buttons">
                {sampleImages.map((s) => (
                  <button
                    key={s.name}
                    className="sample-pill-btn"
                    onClick={() => loadSampleImage(s)}
                    title={s.description}
                  >
                    <img src={s.path} alt="" className="sample-mini-thumb" />
                    <span>{s.name}</span>
                  </button>
                ))}
              </div>
            </div>
          </section>

          {/* Step 3: Analysis Queue */}
          {uploads.length > 0 && (
            <section className="ws-section">
              <div className="ws-section-header">
                <div>
                  <span className="step-badge">Step 3</span>
                  <h2>Image Queue ({uploads.length})</h2>
                </div>
                <button className="btn-link" onClick={clearAll}>
                  Clear all
                </button>
              </div>

              <div className="file-queue">
                {uploads.map((upload) => (
                  <div className="queue-row" key={upload.id}>
                    <img src={upload.url} alt="" className="queue-thumb" />
                    <div className="queue-info">
                      <strong>{upload.name}</strong>
                      <span>
                        {upload.file ? formatBytes(upload.sizeBytes) : "Benchmark Scene"}
                        {upload.output && ` · ${upload.output.imageDimensions.width}×${upload.output.imageDimensions.height}px`}
                      </span>
                    </div>

                    {upload.output && (
                      <span className="queue-metric-pill">
                        {upload.output.metrics.panelCount} panels · {upload.output.metrics.totalAreaM2} m²
                      </span>
                    )}

                    <span className={`queue-badge ${upload.status}`}>
                      {upload.status === "ready" && <Check size={12} />}
                      {upload.status === "analyzing" && (
                        <span className="spinner" />
                      )}
                      {upload.status === "error" && "Error"}
                      {upload.status}
                    </span>

                    <button
                      className="btn-icon-sm"
                      onClick={() => removeUpload(upload.id)}
                      aria-label={`Remove ${upload.name}`}
                    >
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>

              <button
                className="btn-primary btn-full"
                disabled={!selectedModels.length || isRunning}
                onClick={runAnalysis}
              >
                {isRunning ? (
                  <>
                    <span className="spinner light" /> Processing Segmentation Pipelines…
                  </>
                ) : (
                  <>
                    <Play size={16} fill="currentColor" /> Run Dual-Engine Analysis
                    <ArrowRight size={16} />
                  </>
                )}
              </button>
            </section>
          )}

          {/* Results Gallery */}
          {readyUploads.length > 0 && (
            <section className="ws-section results-section">
              <div className="ws-section-header">
                <div>
                  <span className="step-badge success">Analysis Complete</span>
                  <h2>Comparison Gallery</h2>
                </div>
                <div className="results-actions">
                  <button className="btn-link" onClick={runAnalysis}>
                    <RotateCcw size={14} /> Re-run
                  </button>
                </div>
              </div>

              {/* Cumulative stats summary bar */}
              <div className="cumulative-stats-bar">
                <div className="cum-stat">
                  <span className="stat-label">Processed Images</span>
                  <strong className="stat-val">{readyUploads.length}</strong>
                </div>
                <div className="cum-stat">
                  <span className="stat-label">Cumulative Area</span>
                  <strong className="stat-val text-green">{totalAreaEstimated} m²</strong>
                </div>
              </div>

              {/* Grid of Results */}
              <div className="results-grid">
                {readyUploads.map((upload) => {
                  const out = upload.output!;
                  return (
                    <article className="result-card" key={upload.id}>
                      <div
                        className="result-img-wrap"
                        onClick={() => setActiveModalItem(upload)}
                        role="button"
                        tabIndex={0}
                        title="Click to open full inspection comparison"
                      >
                        <img src={out.rcnnResultUrl} alt={upload.name} />
                        <div className="result-overlay">
                          <span className="tag-overlay">MASK R-CNN</span>
                          <span className="expand-pill">
                            <Maximize2 size={12} /> Inspect
                          </span>
                        </div>
                      </div>

                      <div className="result-meta">
                        <div className="result-meta-top">
                          <strong>{upload.name}</strong>
                          <span className="meta-dim">
                            {out.imageDimensions.width}×{out.imageDimensions.height}
                          </span>
                          <button
                            className="btn-icon-sm result-remove"
                            onClick={() => removeUpload(upload.id)}
                            aria-label={`Remove ${upload.name}`}
                            title="Remove from results"
                          >
                            <X size={15} />
                          </button>
                        </div>

                        {/* Metrics Pills */}
                        <div className="result-chips">
                          <span className="result-chip">
                            <Zap size={11} /> {out.metrics.panelCount} panels
                          </span>
                          <span className="result-chip green">
                            {out.metrics.totalAreaM2} m²
                          </span>
                          <span className="result-chip">
                            {out.metrics.avgConfidence}% conf
                          </span>
                          {out.modelSource === "pytorch_trained_weights" && (
                            <span className="result-chip text-green">
                              <Check size={10} /> PyTorch Weights
                            </span>
                          )}
                        </div>

                        <div className="result-card-actions">
                          <button
                            className="btn-primary btn-sm"
                            onClick={() => setActiveModalItem(upload)}
                          >
                            <Sliders size={13} /> Compare Views
                          </button>
                          <button
                            className="btn-secondary btn-sm"
                            onClick={() => {
                              const a = document.createElement("a");
                              a.href = out.rcnnResultUrl;
                              a.download = `${upload.name}_annotated.jpg`;
                              a.click();
                            }}
                            title="Download annotated image"
                          >
                            <Download size={13} />
                          </button>
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>

              {readyUploads.some((u) => u.output?.modelSource === "pytorch_trained_weights") ? (
                <p className="api-note success">
                  <Check size={15} />
                  <span>
                    <strong>Live PyTorch inference:</strong> segmentation is computed directly with your trained
                    model checkpoints (<code>best_cnn_unet.pth</code> &amp; <code>mask_rcnn_final.pth</code>).
                  </span>
                </p>
              ) : (
                <p className="api-note">
                  <Shield size={15} />
                  <span>Client preview mode active. The Python PyTorch server runs on port 8000.</span>
                </p>
              )}
            </section>
          )}
        </div>
      </div>

      {/* ─── Modal Inspector ─── */}
      {activeModalItem && activeModalItem.output && (
        <ImageComparisonModal
          output={activeModalItem.output}
          fileName={activeModalItem.name}
          onClose={() => setActiveModalItem(null)}
        />
      )}
    </div>
  );
}
