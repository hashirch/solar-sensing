import { useState } from "react";
import {
  X,
  Download,
  Layers,
  Cpu,
  Eye,
  Sliders,
  Maximize2,
  FileSpreadsheet,
  Check,
} from "lucide-react";
import { AnalysisOutput } from "../utils/inference";

interface Props {
  output: AnalysisOutput;
  fileName: string;
  onClose: () => void;
}
type ViewTab = "split" | "side-by-side" | "combined" | "cnn" | "rcnn" | "original";

export default function ImageComparisonModal({ output, fileName, onClose }: Props) {
  const [activeTab, setActiveTab] = useState<ViewTab>("split");
  const [splitPos, setSplitPos] = useState<number>(50); // percentage 0 - 100
  const [splitOverlayMode, setSplitOverlayMode] = useState<"cnn" | "rcnn" | "combined">("cnn");
  const [downloadSuccess, setDownloadSuccess] = useState(false);

  const { metrics, imageDimensions, originalUrl, cnnResultUrl, rcnnResultUrl, combinedResultUrl } = output;

  function handleDownload(url: string, suffix: string) {
    const a = document.createElement("a");
    a.href = url;
    a.download = `${fileName.replace(/\.[^/.]+$/, "")}_${suffix}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setDownloadSuccess(true);
    setTimeout(() => setDownloadSuccess(false), 2500);
  }

  function handleExportCSV() {
    const headers = "Instance_ID,Confidence,Box_X,Box_Y,Box_Width,Box_Height,Area_m2\n";
    const rows = metrics.instances
      .map(
        (i) =>
          `${i.id},${(i.confidence * 100).toFixed(2)}%,${i.bbox[0]},${i.bbox[1]},${i.bbox[2]},${i.bbox[3]},${i.areaM2}`
      )
      .join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${fileName.replace(/\.[^/.]+$/, "")}_detections.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const activeOverlayUrl =
    splitOverlayMode === "combined" && combinedResultUrl
      ? combinedResultUrl
      : splitOverlayMode === "rcnn"
      ? rcnnResultUrl
      : cnnResultUrl;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div className="modal-title-wrap">
            <span className="step-badge">Inspection View</span>
            <h2>{fileName}</h2>
            <span className="modal-subtitle">
              {imageDimensions.width} × {imageDimensions.height} px · {metrics.inferenceTimeMs} ms inference
            </span>
          </div>
          <div className="modal-header-actions">
            <div className="tab-pill-group">
              <button
                className={`tab-pill ${activeTab === "split" ? "active" : ""}`}
                onClick={() => setActiveTab("split")}
              >
                <Sliders size={14} /> Split View
              </button>
              <button
                className={`tab-pill ${activeTab === "side-by-side" ? "active" : ""}`}
                onClick={() => setActiveTab("side-by-side")}
              >
                <Layers size={14} /> Side-by-Side
              </button>
              {combinedResultUrl && (
                <button
                  className={`tab-pill ${activeTab === "combined" ? "active" : ""}`}
                  onClick={() => setActiveTab("combined")}
                >
                  <Layers size={14} /> Combined
                </button>
              )}
              <button
                className={`tab-pill ${activeTab === "cnn" ? "active" : ""}`}
                onClick={() => setActiveTab("cnn")}
              >
                <Cpu size={14} /> CNN/U-Net
              </button>
              <button
                className={`tab-pill ${activeTab === "rcnn" ? "active" : ""}`}
                onClick={() => setActiveTab("rcnn")}
              >
                <Eye size={14} /> Mask R-CNN
              </button>
              <button
                className={`tab-pill ${activeTab === "original" ? "active" : ""}`}
                onClick={() => setActiveTab("original")}
              >
                Original
              </button>
            </div>

            <button className="btn-icon" onClick={onClose} aria-label="Close modal">
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="modal-body">
          {/* Main Viewer Area */}
          <div className="modal-viewer">
            {activeTab === "split" && (
              <div className="split-viewer-container">
                <div className="split-controls">
                  <span>Overlay Model:</span>
                  <button
                    className={`btn-subtle ${splitOverlayMode === "cnn" ? "active" : ""}`}
                    onClick={() => setSplitOverlayMode("cnn")}
                  >
                    CNN / U-Net (Dense)
                  </button>
                  <button
                    className={`btn-subtle ${splitOverlayMode === "rcnn" ? "active" : ""}`}
                    onClick={() => setSplitOverlayMode("rcnn")}
                  >
                    Mask R-CNN (Instances)
                  </button>
                  {combinedResultUrl && (
                    <button
                      className={`btn-subtle ${splitOverlayMode === "combined" ? "active" : ""}`}
                      onClick={() => setSplitOverlayMode("combined")}
                    >
                      Combined
                    </button>
                  )}
                  <span className="split-hint">Drag slider to compare raw vs detected</span>
                </div>

                <div className="split-canvas-wrapper">
                  {/* Under layer: Base original */}
                  <img src={originalUrl} alt="Original Satellite" className="split-base-img" />

                  {/* Over layer: Model segmentation (clipped by split position) */}
                  <div
                    className="split-overlay-clip"
                    style={{ clipPath: `polygon(0 0, ${splitPos}% 0, ${splitPos}% 100%, 0 100%)` }}
                  >
                    <img src={activeOverlayUrl} alt="Annotated Output" className="split-overlay-img" />
                    <span className="split-tag left">
                      {splitOverlayMode === "cnn"
                        ? "CNN/U-Net Mask"
                        : splitOverlayMode === "rcnn"
                        ? "Mask R-CNN Detections"
                        : "Combined Output"}
                    </span>
                  </div>

                  <span className="split-tag right">Original Imagery</span>

                  {/* Range Slider Overlay */}
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={splitPos}
                    onChange={(e) => setSplitPos(Number(e.target.value))}
                    className="split-range-slider"
                  />
                  {/* Visual Divider Bar */}
                  <div className="split-divider-line" style={{ left: `${splitPos}%` }}>
                    <div className="split-handle">
                      <Sliders size={16} />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "side-by-side" && (
              <div className="side-by-side-grid">
                <div className="sbs-col">
                  <div className="sbs-header">
                    <strong>Original Satellite Scene</strong>
                    <span>Raw Input</span>
                  </div>
                  <div className="sbs-img-wrap">
                    <img src={originalUrl} alt="Original Satellite" />
                  </div>
                </div>

                <div className="sbs-col">
                  <div className="sbs-header">
                    <strong>CNN / U-Net Prediction</strong>
                    <span className="tag-green">Dense Semantic Mask</span>
                  </div>
                  <div className="sbs-img-wrap">
                    <img src={cnnResultUrl} alt="CNN U-Net Prediction" />
                  </div>
                </div>

                <div className="sbs-col">
                  <div className="sbs-header">
                    <strong>Mask R-CNN Prediction</strong>
                    <span className="tag-gold">Instance Detections</span>
                  </div>
                  <div className="sbs-img-wrap">
                    <img src={rcnnResultUrl} alt="Mask R-CNN Prediction" />
                  </div>
                </div>
              </div>
            )}

            {activeTab === "combined" && combinedResultUrl && (
              <div className="single-view-container">
                <div className="single-view-banner green">
                  <Layers size={16} />
                  <span>
                    Combined · CNN/U-Net coverage <i className="legend-swatch swatch-green" /> with Mask R-CNN
                    panel outlines <i className="legend-swatch swatch-coral" /> and confidence tags
                  </span>
                </div>
                <div className="single-img-wrap">
                  <img src={combinedResultUrl} alt="Combined CNN and Mask R-CNN output" />
                </div>
              </div>
            )}

            {activeTab === "cnn" && (
              <div className="single-view-container">
                <div className="single-view-banner green">
                  <Cpu size={16} />
                  <span>
                    Dense Binary Segmentation Mask · Sigmoid threshold &gt; 0.50 · Total Area: {metrics.totalAreaM2} m²
                  </span>
                </div>
                <div className="single-img-wrap">
                  <img src={cnnResultUrl} alt="CNN / U-Net" />
                </div>
              </div>
            )}

            {activeTab === "rcnn" && (
              <div className="single-view-container">
                <div className="single-view-banner gold">
                  <Eye size={16} />
                  <span>
                    Mask R-CNN ResNet-50-FPN · {metrics.panelCount} instances detected · Avg confidence:{" "}
                    {metrics.avgConfidence}%
                  </span>
                </div>
                <div className="single-img-wrap">
                  <img src={rcnnResultUrl} alt="Mask R-CNN" />
                </div>
              </div>
            )}

            {activeTab === "original" && (
              <div className="single-view-container">
                <div className="single-view-banner">
                  <span>Raw Satellite Imagery · No overlays</span>
                </div>
                <div className="single-img-wrap">
                  <img src={originalUrl} alt="Original Satellite" />
                </div>
              </div>
            )}
          </div>

          {/* Right Metrics & Metadata Panel */}
          <div className="modal-sidebar">
            <div className="modal-sidebar-card">
              <h3>Detection Summary</h3>
              <div className="metrics-summary-grid">

                <div className="metric-box">
                  <span className="lbl">Estimated Solar Area</span>
                  <strong className="val">{metrics.totalAreaM2} m²</strong>
                  <span className="sub">{metrics.coveragePercent}% of scene</span>
                </div>


                <div className="metric-box">
                  <span className="lbl">Mean Confidence</span>
                  <strong className="val">{metrics.avgConfidence}%</strong>
                  <span className="sub">IoU threshold 0.5</span>
                </div>
              </div>
            </div>

            {/* Instance table */}
            <div className="modal-sidebar-card">
              <div className="card-heading-row">
                <h3>Detected Arrays ({metrics.instances.length})</h3>
                <button className="btn-link-sm" onClick={handleExportCSV}>
                  <FileSpreadsheet size={13} /> Export CSV
                </button>
              </div>
              <div className="instances-table-wrap">
                <table className="instances-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Confidence</th>
                      <th>Est. Area</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.instances.slice(0, 10).map((inst) => (
                      <tr key={inst.id}>
                        <td>
                          <span className="instance-idx">#{inst.id}</span>
                        </td>
                        <td>
                          <div className="conf-bar-wrap">
                            <div
                              className="conf-bar-fill"
                              style={{ width: `${inst.confidence * 100}%` }}
                            />
                            <span>{(inst.confidence * 100).toFixed(1)}%</span>
                          </div>
                        </td>
                        <td>{inst.areaM2} m²</td>
                      </tr>
                    ))}
                    {metrics.instances.length > 10 && (
                      <tr>
                        <td colSpan={3} className="more-row">
                          +{metrics.instances.length - 10} more instances in CSV
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Export & Download Buttons */}
            <div className="modal-download-actions">
              {combinedResultUrl && (
                <button
                  className="btn-secondary btn-full"
                  onClick={() => handleDownload(combinedResultUrl, "combined")}
                >
                  <Download size={14} /> Download Combined
                </button>
              )}
              <button
                className="btn-secondary btn-full"
                onClick={() => handleDownload(cnnResultUrl, "cnn_segmentation")}
              >
                <Download size={14} /> Download CNN Mask
              </button>
              <button
                className="btn-primary btn-full"
                onClick={() => handleDownload(rcnnResultUrl, "rcnn_instances")}
              >
                <Download size={14} /> Download Mask R-CNN
              </button>
              {downloadSuccess && (
                <div className="download-toast">
                  <Check size={14} /> Image downloaded successfully
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
