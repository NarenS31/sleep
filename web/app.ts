export {};

type ViewMode = "overview" | "all" | "metric";

type Metric = {
  id: string;
  label: string;
};

type ConceptNode = {
  id: string;
  label: string;
  stage: "Problem" | "Core Shift" | "Techniques" | "Outcomes";
  description: string;
};

const metrics: Metric[] = [
  { id: "total_sleep_hours", label: "Total Sleep Hours" },
  { id: "sleep_efficiency_pct", label: "Sleep Efficiency (%)" },
  { id: "interruptions", label: "Interruptions" },
  { id: "sleep_onset_latency_min", label: "Sleep Onset Latency (min)" },
  { id: "wake_variability_min", label: "Wake Variability (min)" },
  { id: "bedtime_variability_min", label: "Bedtime Variability (min)" },
  { id: "perceived_stress", label: "Perceived Stress" },
  { id: "morning_mood", label: "Morning Mood" },
  { id: "daytime_sleepiness", label: "Daytime Sleepiness" },
  { id: "screen_time_before_bed", label: "Screen Time Before Bed" },
];

const conceptNodes: ConceptNode[] = [
  {
    id: "binary",
    label: "Binary Limits",
    stage: "Problem",
    description: "Binary labels miss the range of real sleep experiences.",
  },
  {
    id: "scoring",
    label: "Multi-Dimensional Scoring",
    stage: "Core Shift",
    description: "Moves from yes/no labeling to richer continuous scoring.",
  },
  {
    id: "autoencoder",
    label: "Autoencoder",
    stage: "Techniques",
    description: "Learns structure from data without strict label dependency.",
  },
  {
    id: "latent",
    label: "Latent Representation",
    stage: "Techniques",
    description: "Compresses sleep patterns into useful internal signals.",
  },
  {
    id: "scaling",
    label: "Feature Scaling",
    stage: "Techniques",
    description: "Balances metrics so no single feature dominates.",
  },
  {
    id: "bias",
    label: "Bias Reduction",
    stage: "Outcomes",
    description: "Less rigid dependence on simplistic labels.",
  },
  {
    id: "flex",
    label: "Flexibility",
    stage: "Outcomes",
    description: "Adapts to varied sleep patterns across people.",
  },
  {
    id: "accurate",
    label: "Accurate Sleep Representation",
    stage: "Outcomes",
    description: "Produces a more realistic view of sleep quality.",
  },
];

const links = [
  "Binary Limits -> Multi-Dimensional Scoring",
  "Scoring -> Autoencoder",
  "Scoring -> Feature Scaling",
  "Autoencoder -> Latent Representation",
  "Bias Reduction -> Flexibility -> Accurate Sleep Representation",
];

declare const Plotly: any;

const viewSelect = document.getElementById("view-select") as HTMLSelectElement;
const metricSelect = document.getElementById("metric-select") as HTMLSelectElement;
const metricControl = document.getElementById("metric-control") as HTMLLabelElement;
const overview = document.getElementById("overview") as HTMLDivElement;
const analysis = document.getElementById("analysis") as HTMLDivElement;
const guideCopy = document.getElementById("guide-copy") as HTMLParagraphElement;
const kpiPeople = document.getElementById("kpi-people") as HTMLElement;
const kpiMetrics = document.getElementById("kpi-metrics") as HTMLElement;

const PEOPLE = 380;

function buildSelectors() {
  viewSelect.innerHTML = "";
  metricSelect.innerHTML = "";

  const overviewOption = new Option("Method Overview", "overview");
  const allOption = new Option("All Metrics (Cone)", "all");
  const metricOption = new Option("Focused Metric", "metric");
  viewSelect.add(overviewOption);
  viewSelect.add(allOption);
  viewSelect.add(metricOption);

  metrics.forEach((m) => metricSelect.add(new Option(m.label, m.id)));
}

function stageColumn(stage: ConceptNode["stage"]) {
  const nodes = conceptNodes.filter((n) => n.stage === stage);
  const cards = nodes
    .map(
      (n) =>
        `<article class="node"><div>${n.label}</div><small>${n.description}</small></article>`,
    )
    .join("");
  return `<section class="flow-column"><div class="flow-title">${stage}</div>${cards}</section>`;
}

function renderOverview() {
  const hero = `
    <div class="overview-hero">
      <div class="hero-card">
        <p class="eyebrow">Method Overview</p>
        <h2>From binary labels to meaningful sleep intelligence</h2>
        <p>This flow explains how the model structure improves interpretability and practical sleep analysis.</p>
      </div>
      <div class="summary-card">
        <p class="eyebrow">Takeaway</p>
        <h3>Richer modeling gives better sleep representation</h3>
        <p>Multi-dimensional scoring and latent modeling improve flexibility and reduce bias.</p>
      </div>
    </div>
  `;

  const flow = `<div class="flow-grid">${stageColumn("Problem")}${stageColumn("Core Shift")}${stageColumn("Techniques")}${stageColumn("Outcomes")}</div>`;
  const linkRow = `<div class="link-row">${links.map((t) => `<span class="link-chip">${t}</span>`).join("")}</div>`;
  const desc = `<div class="desc-grid">${conceptNodes
    .map((n) => `<article class="desc-card"><h4>${n.label}</h4><p>${n.description}</p></article>`)
    .join("")}</div>`;

  overview.innerHTML = `${hero}${flow}${linkRow}${desc}`;
}

function mulberry32(seed: number) {
  return function rng() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function drawMetric(metricId: string) {
  const metricIndex = metrics.findIndex((m) => m.id === metricId);
  const center = metricIndex * 3.4;
  const rng = mulberry32(metricIndex + 17);
  const golden = Math.PI * (3 - Math.sqrt(5));

  const x: number[] = [];
  const y: number[] = [];
  const z: number[] = [];
  const text: string[] = [];
  const color: number[] = [];
  const size: number[] = [];

  for (let i = 0; i < PEOPLE; i += 1) {
    const score = Math.round(rng() * 100);
    const norm = Math.min(1, Math.max(0, (score / 100) + (rng() - 0.5) * 0.26));
    const angle = i * golden;
    const radial = 0.026 * Math.sqrt(i + 1);

    x.push(center + Math.cos(angle) * radial * 4.2);
    y.push(norm + Math.sin(angle) * radial * 0.72);
    z.push(score + (Math.cos(angle * 0.7) * radial * 8));
    color.push(score);
    size.push(4 + (score / 100) * 3.5);
    text.push(`Person #${i + 1}<br>${metrics[metricIndex].label}<br>Sleep score: ${score}/100`);
  }

  const traces = [
    {
      type: "scatter3d",
      mode: "markers",
      x,
      y,
      z,
      text,
      hoverinfo: "text",
      marker: {
        size,
        color,
        colorscale: "Turbo",
        opacity: 0.82,
        line: { color: "rgba(255,255,255,0.85)", width: 0.6 },
        colorbar: { title: "Sleep score" },
      },
    },
  ];

  const layout = {
    margin: { l: 0, r: 0, b: 0, t: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    scene: {
      bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        title: "Selected Metric",
        tickvals: [center],
        ticktext: [metrics[metricIndex].label],
        range: [center - 11, center + 11],
        color: "#dce9f8",
      },
      yaxis: {
        title: "Relative Metric Level",
        tickvals: [-0.25, 0, 0.5, 1, 1.25],
        ticktext: ["Very Low", "Low", "Medium", "High", "Very High"],
        range: [-0.6, 1.6],
        color: "#dce9f8",
      },
      zaxis: {
        title: "Sleep Score (0-100)",
        range: [-10, 110],
        color: "#dce9f8",
      },
      camera: { eye: { x: 2.15, y: 1.45, z: 1.25 } },
    },
  };

  Plotly.react("plot", traces, layout, { responsive: true, displaylogo: false });
}

function drawAllMetricsCone() {
  const golden = Math.PI * (3 - Math.sqrt(5));

  const x: number[] = [];
  const y: number[] = [];
  const z: number[] = [];
  const text: string[] = [];
  const color: number[] = [];
  const size: number[] = [];

  for (let metricIndex = 0; metricIndex < metrics.length; metricIndex += 1) {
    const metric = metrics[metricIndex];
    const rng = mulberry32(metricIndex + 77);
    for (let i = 0; i < PEOPLE; i += 1) {
      const score = Math.round(rng() * 100);
      const angle = i * golden + metricIndex * 0.72;
      const coneRadius = ((100 - score) / 100) * 5.4 + metricIndex * 0.07;
      const drift = (metricIndex - (metrics.length - 1) / 2) * 0.16;

      x.push(Math.cos(angle) * coneRadius + drift);
      y.push(score + (rng() - 0.5) * 2.4);
      z.push(Math.sin(angle) * coneRadius + drift);
      color.push(score);
      size.push(2.8 + (score / 100) * 1.8);
      text.push(`Person #${i + 1}<br>${metric.label}<br>Sleep score: ${score}/100`);
    }
  }

  const traces = [
    {
      type: "scatter3d",
      mode: "markers",
      x,
      y,
      z,
      text,
      hoverinfo: "text",
      marker: {
        size,
        color,
        colorscale: "Turbo",
        opacity: 0.8,
        line: { color: "rgba(255,255,255,0.3)", width: 0.4 },
        colorbar: { title: "Sleep score" },
      },
    },
  ];

  const layout = {
    margin: { l: 0, r: 0, b: 0, t: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    scene: {
      bgcolor: "rgba(0,0,0,0)",
      xaxis: { title: "Cone Spread X", range: [-8.5, 8.5], color: "#dce9f8" },
      yaxis: { title: "Sleep Score (0-100)", range: [-5, 105], color: "#dce9f8" },
      zaxis: { title: "Cone Spread Z", range: [-8.5, 8.5], color: "#dce9f8" },
      camera: { eye: { x: 1.95, y: 1.05, z: 1.85 } },
    },
  };

  Plotly.react("plot", traces, layout, { responsive: true, displaylogo: false });
}

function setMode(mode: ViewMode) {
  if (mode === "overview") {
    overview.classList.remove("hidden");
    analysis.classList.add("hidden");
    metricControl.classList.add("hidden");
    guideCopy.textContent = "Read left to right: problem, core shift, techniques, and outcomes. The cards below explain each node in plain language.";
    renderOverview();
  } else if (mode === "all") {
    overview.classList.add("hidden");
    analysis.classList.remove("hidden");
    metricControl.classList.add("hidden");
    guideCopy.textContent = "All metrics are combined into a cone-shaped cloud. Wider base means lower scores; tighter top means higher scores.";
    drawAllMetricsCone();
  } else {
    overview.classList.add("hidden");
    analysis.classList.remove("hidden");
    metricControl.classList.remove("hidden");
    guideCopy.textContent = "Each point is one person. More spacing and larger axis ranges make clusters easier to read.";
    drawMetric(metricSelect.value);
  }
}

function boot() {
  buildSelectors();
  kpiPeople.textContent = String(PEOPLE);
  kpiMetrics.textContent = String(metrics.length);

  viewSelect.addEventListener("change", () => setMode(viewSelect.value as ViewMode));
  metricSelect.addEventListener("change", () => {
    if (viewSelect.value === "metric") {
      drawMetric(metricSelect.value);
    }
  });

  setMode("all");
}

boot();
