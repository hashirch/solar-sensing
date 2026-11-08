import {
  Activity,
  ArrowUpRight,
  BarChart3,
  Check,
  ChevronDown,
  CloudUpload,
  Cpu,
  FileImage,
  Layers3,
  Menu,
  PanelTop,
  Play,
  ShieldCheck,
  Sparkles,
  X,
  Zap,
} from "lucide-react";
import { ChangeEvent, useMemo, useRef, useState } from "react";

type ModelId = "cnn" | "rcnn";
type Model = { id: ModelId; name: string; shortName: string; detail: string; accent: string };
type Upload = { id: string; file: File; url: string; status: "queued" | "analyzing" | "ready" };

const models: Model[] = [
  { id: "cnn", name: "CNN / U-Net", shortName: "CNN/U-Net", detail: "Dense surface masks", accent: "green" },
  { id: "rcnn", name: "Mask R-CNN", shortName: "Mask R-CNN", detail: "Instance detections", accent: "gold" },
];

function App() {
  const [selectedModels, setSelectedModels] = useState<ModelId[]>(["cnn", "rcnn"]);
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const selectedLabel = useMemo(
    () => `${selectedModels.length} engine${selectedModels.length === 1 ? "" : "s"}`,
    [selectedModels],
  );

  function addFiles(files: FileList | File[]) {
    const accepted = Array.from(files).filter((file) => file.type.startsWith("image/"));
    setUploads((current) => [
      ...current,
      ...accepted.map((file) => ({
        id: `${file.name}-${file.lastModified}-${Math.random()}`,
        file,
        url: URL.createObjectURL(file),
        status: "queued" as const,
      })),
    ]);
  }

  function handleInput(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) addFiles(event.target.files);
    event.target.value = "";
  }

  function toggleModel(id: ModelId) {
    setSelectedModels((current) =>
      current.includes(id) ? current.filter((modelId) => modelId !== id) : [...current, id],
    );
  }

  function runAnalysis() {
    if (!uploads.length || !selectedModels.length || isRunning) return;
    setIsRunning(true);
    setUploads((current) => current.map((upload) => ({ ...upload, status: "analyzing" })));
    window.setTimeout(() => {
      setUploads((current) => current.map((upload) => ({ ...upload, status: "ready" })));
      setIsRunning(false);
    }, 1400);
  }

  function removeUpload(id: string) {
    setUploads((current) => {
      const upload = current.find((item) => item.id === id);
      if (upload) URL.revokeObjectURL(upload.url);
      return current.filter((item) => item.id !== id);
    });
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><PanelTop size={20} /></div>
          <div><strong>SolarSight</strong><span>Panel intelligence</span></div>
        </div>
        <nav className="side-nav" aria-label="Primary navigation">
          <a className="nav-item active" href="#workspace"><Activity size={17} /> Workspace</a>
          <a className="nav-item" href="#models"><Layers3 size={17} /> Model library</a>
          <a className="nav-item" href="#about"><BarChart3 size={17} /> Research notes</a>
        </nav>
        <div className="sidebar-bottom">
          <div className="side-status"><span className="status-dot" /> Local workspace</div>
          <p>Images stay on your machine while you compare model outputs.</p>
        </div>
      </aside>

      <main className="main-content" id="workspace">
        <header className="topbar">
          <button className="mobile-menu" aria-label="Open navigation"><Menu size={19} /></button>
          <div className="breadcrumb"><span>Workspace</span><ChevronDown size={14} /><b>New comparison</b></div>
          <div className="top-actions"><span className="device-badge"><span className="status-dot" /> CPU ready</span><button className="avatar" aria-label="Account">H</button></div>
        </header>

        <section className="hero">
          <div className="eyebrow"><Sparkles size={14} /> Solar panel segmentation · 01</div>
          <div className="hero-row">
            <div>
              <h1>See the surface.<br /><em>Measure the potential.</em></h1>
              <p>Compare computer-vision models on satellite imagery and surface the panels that matter for your next decision.</p>
            </div>
            <div className="hero-orbit" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit-core"><Zap size={24} /></div></div>
          </div>
        </section>

        <section className="metrics" aria-label="Workspace overview">
          <Metric icon={<Layers3 />} value="02" label="Segmentation engines" note="Ready to compare" />
          <Metric icon={<ShieldCheck />} value="100%" label="Privacy-first workflow" note="No remote uploads" />
          <Metric icon={<BarChart3 />} value="21–25" label="Multi-temporal dataset" note="Research coverage" />
        </section>

        <section className="workspace-grid">
          <div className="workspace-main">
            <div className="section-heading"><div><span className="section-kicker">01 / Configure</span><h2>Choose your engines</h2></div><span className="selection-count">{selectedLabel}</span></div>
            <div className="model-grid" id="models">
              {models.map((model) => {
                const selected = selectedModels.includes(model.id);
                return <button className={`model-option ${selected ? "selected" : ""}`} key={model.id} onClick={() => toggleModel(model.id)} aria-pressed={selected}>
                  <div className={`model-icon ${model.accent}`}><Cpu size={20} /></div>
                  <div className="model-copy"><strong>{model.name}</strong><span>{model.detail}</span></div>
                  <span className={`check ${selected ? "checked" : ""}`}>{selected && <Check size={13} />}</span>
                </button>;
              })}
            </div>

            <div className="section-heading upload-heading"><div><span className="section-kicker">02 / Input</span><h2>Bring in imagery</h2></div><span className="file-hint">JPG, PNG, WEBP · max 50 MB</span></div>
            <div className={`dropzone ${isDragging ? "dragging" : ""}`} onClick={() => fileInput.current?.click()} onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={(event) => { event.preventDefault(); setIsDragging(false); addFiles(event.dataTransfer.files); }}>
              <input ref={fileInput} type="file" accept="image/*" multiple onChange={handleInput} />
              <div className="upload-icon"><CloudUpload size={23} /></div>
              <strong>Drop satellite imagery here</strong>
              <span>or <u>browse from your device</u></span>
            </div>

            {uploads.length > 0 && <div className="queue">
              <div className="queue-header"><strong>Comparison queue</strong><span>{uploads.length} image{uploads.length === 1 ? "" : "s"}</span></div>
              {uploads.map((upload) => <div className="queue-item" key={upload.id}>
                <img src={upload.url} alt="" /><div className="queue-file"><strong>{upload.file.name}</strong><span>{formatBytes(upload.file.size)}</span></div>
                <span className={`queue-status ${upload.status}`}>{upload.status === "ready" && <Check size={13} />}{upload.status === "analyzing" && <span className="spinner" />}{upload.status}</span>
                <button className="remove-file" onClick={() => removeUpload(upload.id)} aria-label={`Remove ${upload.file.name}`}><X size={16} /></button>
              </div>)}
              <button className="run-button" disabled={!selectedModels.length || isRunning} onClick={runAnalysis}>{isRunning ? <><span className="spinner light" /> Running comparison...</> : <><Play size={16} fill="currentColor" /> Run comparison <ArrowUpRight size={16} /></>}</button>
            </div>}
          </div>

          <aside className="insight-card">
            <div className="insight-top"><span className="section-kicker">Live workspace</span><span className="live-dot">LIVE</span></div>
            <div className="insight-graphic"><div className="graphic-grid" /><div className="sun-shape" /><div className="panel-shape panel-a" /><div className="panel-shape panel-b" /><div className="graphic-label">FIELD 04 <span>•</span> 31.52°N</div></div>
            <h3>A clearer read on your roofscape</h3>
            <p>Run both engines to see dense coverage and individual detections side by side.</p>
            <div className="insight-list"><div><Check size={14} /> Local-first processing</div><div><Check size={14} /> Batch-ready workflow</div><div><Check size={14} /> Research-grade outputs</div></div>
          </aside>
        </section>

        {uploads.some((upload) => upload.status === "ready") && <section className="results-section">
          <div className="section-heading"><div><span className="section-kicker">03 / Results</span><h2>Comparison ready</h2></div><span className="selection-count green">Preview overlays</span></div>
          <div className="result-grid">{uploads.filter((upload) => upload.status === "ready").map((upload) => <article className="result-card" key={upload.id}><div className="result-image"><img src={upload.url} alt={`Preview of ${upload.file.name}`} /><div className="result-overlay"><span>ANALYSIS PREVIEW</span><span className="crosshair">+</span></div></div><div className="result-info"><strong>{upload.file.name}</strong><span>Ready for model inference · {selectedModels.length} selected</span></div></article>)}</div>
          <p className="api-note"><ShieldCheck size={15} /> Preview mode is active. Set <code>VITE_API_URL</code> to connect this interface to the Python inference service.</p>
        </section>}

        <footer id="about"><span>SOLARSIGHT / LOCAL INFERENCE</span><span>Built for solar-panel segmentation research</span></footer>
      </main>
    </div>
  );
}

function Metric({ icon, value, label, note }: { icon: React.ReactNode; value: string; label: string; note: string }) {
  return <div className="metric-card"><div className="metric-icon">{icon}</div><div><span>{label}</span><strong>{value}</strong><small>{note}</small></div></div>;
}

function formatBytes(bytes: number) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default App;
