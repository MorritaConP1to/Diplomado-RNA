const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");

// Icon imports
const FaSearch = require("react-icons/fa").FaSearch;
const FaLink = require("react-icons/fa").FaLink;
const FaReceipt = require("react-icons/fa").FaReceipt;
const FaUtensils = require("react-icons/fa").FaUtensils;
const FaPhone = require("react-icons/fa").FaPhone;
const FaVolumeDown = require("react-icons/fa").FaVolumeDown;
const FaExclamationTriangle = require("react-icons/fa").FaExclamationTriangle;
const FaBalanceScale = require("react-icons/fa").FaBalanceScale;
const FaThermometerHalf = require("react-icons/fa").FaThermometerHalf;
const FaProjectDiagram = require("react-icons/fa").FaProjectDiagram;
const FaBook = require("react-icons/fa").FaBook;
const FaCheckCircle = require("react-icons/fa").FaCheckCircle;
const FaCheckDouble = require("react-icons/fa").FaCheckDouble;
const FaBrain = require("react-icons/fa").FaBrain;
const FaBullseye = require("react-icons/fa").FaBullseye;
const FaLayerGroup = require("react-icons/fa").FaLayerGroup;
const FaArrowRight = require("react-icons/fa").FaArrowRight;
const FaRuler = require("react-icons/fa").FaRuler;
const FaDna = require("react-icons/fa").FaDna;

function renderIconSvg(Icon, color = "#2563EB", size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(Icon, { color, size: String(size) })
  );
}

async function iconToBase64(Icon, color, size = 256) {
  const svg = renderIconSvg(Icon, color, size);
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ============================================================

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Diplomado Redes Neuronales";
pres.title = "Backpropagation desde Cero";

const C = {
  bg: "F3F4F6", card: "FFFFFF", cardBorder: "E5E7EB",
  accent: "2563EB", accent2: "7C3AED", green: "059669",
  amber: "B45309", red: "DC2626", text: "1F2937", muted: "6B7280",
  codeBg: "1F2937", codeText: "F9FAFB",
};
const F = { title: "Trebuchet MS", body: "Calibri", code: "Consolas" };

const makeShadow = () => ({ type: "outer", color: "000000", blur: 5, offset: 2, angle: 135, opacity: 0.07 });

function addCodeBox(slide, code, x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.codeBg }, shadow: makeShadow() });
  slide.addText(code, { x: x + 0.15, y: y + 0.08, w: w - 0.3, h: h - 0.16, fontSize: 10, fontFace: F.code, color: C.codeText, valign: "top", margin: 0 });
}

async function addIcon(slide, Icon, x, y, w, h, color) {
  const data = await iconToBase64(Icon, color || C.accent);
  slide.addImage({ data, x, y, w, h });
}

function addPausa(slide, question, x, y, w) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 1.2, h: 0.28, fill: { color: C.amber }, shadow: makeShadow() });
  slide.addText("Pausa y piensa", { x, y, w: 1.2, h: 0.28, fontSize: 9, fontFace: F.title, color: "FFFFFF", bold: true, align: "center", valign: "middle" });
  slide.addText(question, { x: x + 1.35, y, w: w - 1.35, h: 0.28, fontSize: 10, fontFace: F.body, color: C.amber, italic: true, valign: "middle" });
}

// ============================================================
// SLIDE 1 — PORTADA
// ============================================================
async function build() {
{
  const s = pres.addSlide();
  s.background = { color: C.card };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.accent } });

  // Neural network visual: 3 layers circles
  const lx = [2.5, 5.0, 7.5];
  const ly = [1.8, 2.5, 3.2];

  // Layer labels
  const llabels = ["Input", "Hidden", "Output"];
  lx.forEach((x, i) => {
    const nodes = i === 0 ? [2.5] : i === 1 ? [2.5, 3.5, 1.5] : [2.5];
    nodes.forEach((y, j) => {
      const cy = y;
      s.addShape(pres.shapes.OVAL, {
        x: x - 0.2, y: cy - 0.2, w: 0.4, h: 0.4,
        fill: { color: i === 2 ? C.green : C.accent, transparency: 20 },
        line: { color: i === 2 ? C.green : C.accent, width: 1.5 },
      });
      // connections to next layer
      if (i < 2) {
        const nextx = lx[i + 1];
        const nextNodes = i === 0 ? [2.5] : [2.5, 3.5, 1.5];
        nextNodes.forEach((ny) => {
          s.addShape(pres.shapes.LINE, {
            x: x + 0.2, y: cy, w: nextx - x - 0.4, h: ny - cy,
            line: { color: C.cardBorder, width: 0.5 },
          });
        });
      }
    });
    s.addText(llabels[i], { x: x - 0.5, y: 3.6, w: 1.0, h: 0.3, fontSize: 8, fontFace: F.body, color: C.muted, align: "center", margin: 0 });
  });

  // Title
  s.addText("Backpropagation\ndesde Cero", {
    x: 0.8, y: 0.6, w: 5.0, h: 1.2,
    fontSize: 40, fontFace: F.title, color: C.text, bold: true, margin: 0,
  });
  s.addText("El algoritmo que hace posible el aprendizaje", {
    x: 0.8, y: 1.7, w: 5.0, h: 0.4,
    fontSize: 16, fontFace: F.body, color: C.accent, italic: true, margin: 0,
  });
  s.addText("Diplomado en Redes Neuronales", {
    x: 0.8, y: 4.5, w: 4.0, h: 0.3,
    fontSize: 13, fontFace: F.body, color: C.muted, margin: 0,
  });
  await addIcon(s, FaBrain, 8.5, 0.5, 0.5, 0.5, C.accent);
}

// ============================================================
// SLIDE 2 — EL PROBLEMA
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("El Problema", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // Big stat
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 1.1, w: 3.5, h: 2.0, fill: { color: C.card }, shadow: makeShadow() });
  s.addText("2,359,296", { x: 0.8, y: 1.2, w: 3.5, h: 1.0, fontSize: 48, fontFace: F.title, color: C.accent, bold: true, align: "center", margin: 0 });
  s.addText("pesos en 1 capa", { x: 0.8, y: 2.2, w: 3.5, h: 0.3, fontSize: 13, fontFace: F.body, color: C.muted, align: "center", margin: 0 });
  s.addText("Probar 1 peso = 1 forward pass", { x: 0.8, y: 2.5, w: 3.5, h: 0.3, fontSize: 12, fontFace: F.body, color: C.muted, align: "center", margin: 0 });

  // Right visual: test each vs all at once
  s.addShape(pres.shapes.RECTANGLE, { x: 4.8, y: 1.1, w: 4.8, h: 2.0, fill: { color: C.card }, shadow: makeShadow() });
  // Tiny boxes representing weights
  for (let i = 0; i < 8; i++) {
    const bx = 5.0 + (i % 4) * 1.1;
    const by = 1.4 + Math.floor(i / 4) * 0.8;
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: by, w: 0.9, h: 0.55, fill: { color: C.accent, transparency: 80 }, line: { color: C.accent, width: 0.5 } });
  }
  s.addText("... 2.3M m\u00e1s", { x: 8.0, y: 2.3, w: 1.2, h: 0.3, fontSize: 9, fontFace: F.body, color: C.muted, align: "center", margin: 0 });
  s.addText("vs", { x: 7.0, y: 1.4, w: 0.4, h: 0.4, fontSize: 14, fontFace: F.title, color: C.amber, bold: true, align: "center", margin: 0 });

  // Bottom
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 3.4, w: 8.4, h: 0.7, fill: { color: C.card }, line: { color: C.green, width: 1.5 }, shadow: makeShadow() });
  s.addText("Backpropagation calcula TODOS los gradientes en UN solo backward pass", {
    x: 1.0, y: 3.45, w: 8.0, h: 0.55, fontSize: 14, fontFace: F.title, color: C.green, bold: true, valign: "middle", margin: 0,
  });

  await addIcon(s, FaSearch, 8.8, 0.3, 0.5, 0.5, C.accent);
  addPausa(s, "\u00bfCu\u00e1nto tardar\u00edas en probar 2.3M de pesos uno por uno?", 0.8, 4.4, 8.4);
}

// ============================================================
// SLIDE 3 — LA SOLUCIÓN
// ============================================================
async function slide3(s) {
  s.addText("La Soluci\u00f3n", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });
  await addIcon(s, FaSearch, 8.8, 0.3, 0.5, 0.5, C.accent);

  // 3-step flow as main visual
  const steps = [
    { label: "Forward", sub: "Calcula salida", color: C.green },
    { label: "Backward", sub: "Calcula gradientes", color: C.accent },
    { label: "Actualizar", sub: "Ajusta pesos", color: C.accent2 },
  ];
  steps.forEach((st, i) => {
    const cx = 0.8 + i * 3.1;
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: 1.2, w: 2.5, h: 1.6, fill: { color: C.card }, line: { color: st.color, width: 2 }, shadow: makeShadow() });
    s.addShape(pres.shapes.OVAL, { x: cx + 0.85, y: 1.3, w: 0.6, h: 0.6, fill: { color: st.color } });
    s.addText(String(i + 1), { x: cx + 0.85, y: 1.3, w: 0.6, h: 0.6, fontSize: 20, fontFace: F.title, color: "FFFFFF", bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(st.label, { x: cx, y: 2.0, w: 2.5, h: 0.35, fontSize: 16, fontFace: F.title, color: st.color, bold: true, align: "center", margin: 0 });
    s.addText(st.sub, { x: cx, y: 2.3, w: 2.5, h: 0.3, fontSize: 11, fontFace: F.body, color: C.muted, align: "center", margin: 0 });
    if (i < 2) {
      s.addShape(pres.shapes.LINE, { x: cx + 2.5, y: 2.0, w: 0.6, h: 0, line: { color: C.muted, width: 2, dashType: "dash" } });
      s.addText("\u25B6", { x: cx + 2.55, y: 1.85, w: 0.4, h: 0.3, fontSize: 14, color: C.muted, align: "center", margin: 0 });
    }
  });

  addPausa(s, "Detective: observa (forward) y reconstruye (backward) para hallar al culpable", 0.8, 3.1, 8.4);
}

// ============================================================
// SLIDE 3 — LA SOLUCIÓN (call)
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  await slide3(s);
}

// ============================================================
// SLIDE 4 — REGLA DE LA CADENA
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("Regla de la Cadena", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });
  await addIcon(s, FaLink, 8.8, 0.3, 0.5, 0.5, C.accent2);

  // Chain diagram - main visual
  const nodes = ["x", "g", "f", "y"];
  const nx = [1.0, 3.5, 6.0, 8.5];
  nodes.forEach((n, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: nx[i], y: 1.3, w: 1.3, h: 0.8, fill: { color: C.card }, line: { color: C.accent, width: 2.5 }, shadow: makeShadow() });
    s.addText(n, { x: nx[i], y: 1.3, w: 1.3, h: 0.8, fontSize: 24, fontFace: F.code, color: C.accent, bold: true, align: "center", valign: "middle", margin: 0 });
    if (i < 3) {
      s.addShape(pres.shapes.LINE, { x: nx[i] + 1.3, y: 1.7, w: 0.9, h: 0, line: { color: C.accent2, width: 2 } });
      s.addText("dy/dx", { x: nx[i] + 1.2, y: 1.0, w: 0.9, h: 0.3, fontSize: 9, fontFace: F.code, color: C.accent2, align: "center", margin: 0 });
    }
  });

  // Formula
  s.addShape(pres.shapes.RECTANGLE, { x: 2.0, y: 2.5, w: 6.0, h: 0.8, fill: { color: C.card }, shadow: makeShadow() });
  s.addText("dy/dx = f'(g(x)) \u00B7 g'(x)", { x: 2.0, y: 2.5, w: 6.0, h: 0.8, fontSize: 22, fontFace: F.code, color: C.accent2, bold: true, align: "center", valign: "middle", margin: 0 });

  // Short bullets
  s.addText([
    { text: "Multiplicas derivadas a lo largo de la cadena", options: { bullet: true, breakLine: true } },
    { text: "Cadena humana: cada persona modifica y pasa el mensaje", options: { bullet: true, breakLine: true } },
  ], { x: 0.8, y: 3.6, w: 8.4, h: 0.6, fontSize: 13, fontFace: F.body, color: C.text, valign: "top", margin: 0, paraSpaceAfter: 4 });

  addPausa(s, "Si hay 10 personas y cada una entiende el 80%, \u00bfqu\u00e9 fracci\u00f3n llega al final?", 0.8, 4.6, 8.4);
}

// ============================================================
// SLIDE 5 — GRAFO COMPUTACIONAL
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("Grafo Computacional", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // Graph as main visual
  const inputs = [
    { label: "x", x: 0.8, y: 1.8 },
    { label: "w", x: 0.8, y: 3.2 },
    { label: "b", x: 3.5, y: 3.2 },
    { label: "y", x: 8.4, y: 1.5 },
  ];
  inputs.forEach((v) => {
    s.addShape(pres.shapes.OVAL, { x: v.x, y: v.y, w: 0.8, h: 0.5, fill: { color: C.card }, line: { color: C.accent, width: 2 } });
    s.addText(v.label, { x: v.x, y: v.y, w: 0.8, h: 0.5, fontSize: 14, fontFace: F.code, color: C.accent, bold: true, align: "center", valign: "middle", margin: 0 });
  });

  const ops = [
    { label: "*", x: 2.0, y: 2.5, cx: 2.4, cy: 2.75 },
    { label: "+", x: 4.4, y: 2.5, cx: 4.8, cy: 2.75 },
    { label: "\u03C3", x: 6.4, y: 2.5, cx: 6.8, cy: 2.75 },
    { label: "L", x: 8.0, y: 2.5, cx: 8.4, cy: 2.75 },
  ];
  ops.forEach((o) => {
    s.addShape(pres.shapes.RECTANGLE, { x: o.x, y: o.y, w: 0.7, h: 0.5, fill: { color: C.accent2 } });
    s.addText(o.label, { x: o.x, y: o.y, w: 0.7, h: 0.5, fontSize: 16, fontFace: F.body, color: "FFFFFF", bold: true, align: "center", valign: "middle", margin: 0 });
  });

  await addIcon(s, FaReceipt, 8.8, 0.3, 0.5, 0.5, C.accent2);
  addPausa(s, "Cada operaci\u00F3n es un art\u00EDculo en tu recibo. El total = la p\u00E9rdida.", 0.8, 4.2, 8.4);
}

// ============================================================
// SLIDE 6 — FORWARD VS BACKWARD
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("Forward vs Backward", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  const colW = 4.0;
  // Forward
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 1.2, w: colW, h: 2.5, fill: { color: C.card }, line: { color: C.green, width: 2.5 }, shadow: makeShadow() });
  s.addShape(pres.shapes.OVAL, { x: 2.0, y: 1.35, w: 0.6, h: 0.6, fill: { color: C.green } });
  s.addText("\u2192", { x: 2.0, y: 1.35, w: 0.6, h: 0.6, fontSize: 22, fontFace: F.body, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  s.addText("Forward", { x: 0.8, y: 2.0, w: colW, h: 0.35, fontSize: 20, fontFace: F.title, color: C.green, bold: true, align: "center", margin: 0 });
  s.addText([
    { text: "Calcula salida", options: { bullet: true, breakLine: true } },
    { text: "Guarda valores intermedios", options: { bullet: true, breakLine: true } },
  ], { x: 1.0, y: 2.5, w: colW - 0.4, h: 1.0, fontSize: 13, fontFace: F.body, color: C.text, valign: "top", margin: 0, paraSpaceAfter: 4 });

  // Backward
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 1.2, w: colW, h: 2.5, fill: { color: C.card }, line: { color: C.accent, width: 2.5 }, shadow: makeShadow() });
  s.addShape(pres.shapes.OVAL, { x: 6.6, y: 1.35, w: 0.6, h: 0.6, fill: { color: C.accent } });
  s.addText("\u2190", { x: 6.6, y: 1.35, w: 0.6, h: 0.6, fontSize: 22, fontFace: F.body, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  s.addText("Backward", { x: 5.4, y: 2.0, w: colW, h: 0.35, fontSize: 20, fontFace: F.title, color: C.accent, bold: true, align: "center", margin: 0 });
  s.addText([
    { text: "Calcula gradientes", options: { bullet: true, breakLine: true } },
    { text: "Un solo pase", options: { bullet: true, breakLine: true } },
  ], { x: 5.6, y: 2.5, w: colW - 0.4, h: 1.0, fontSize: 13, fontFace: F.body, color: C.text, valign: "top", margin: 0, paraSpaceAfter: 4 });

  await addIcon(s, FaUtensils, 8.8, 0.3, 0.5, 0.5, C.accent2);
  addPausa(s, "Cocinar = forward. Probar y ajustar la receta = backward.", 0.8, 4.0, 8.4);
}

// ============================================================
// SLIDE 7 — FLUJO DEL GRADIENTE
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("Flujo del Gradiente", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // Visual: 4 colored bars diminishing
  const layers = [
    { label: "Loss", grad: "1", w: 7.5 },
    { label: "Capa 3", grad: "0.25", w: 2.0 },
    { label: "Capa 2", grad: "0.06", w: 0.5 },
    { label: "Capa 1", grad: "0.015", w: 0.12 },
  ];

  const barY = 1.2;
  layers.forEach((l, i) => {
    const by = barY + i * 0.95;
    s.addText(l.label, { x: 0.8, y: by, w: 1.2, h: 0.35, fontSize: 13, fontFace: F.title, color: C.text, bold: true, valign: "middle", margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 2.2, y: by + 0.05, w: l.w, h: 0.3,
      fill: { color: i === 0 ? C.red : i === 1 ? C.amber : i === 2 ? C.accent : C.accent2, transparency: 30 },
      line: { color: i === 0 ? C.red : i === 1 ? C.amber : i === 2 ? C.accent : C.accent2, width: 1.5 },
    });
    s.addText(l.grad, { x: 2.3, y: by + 0.05, w: 1.0, h: 0.3, fontSize: 11, fontFace: F.code, color: C.text, valign: "middle", margin: 0 });
    if (i < 3) {
      s.addText("\u2193", { x: 5.5, y: by + 0.25, w: 0.3, h: 0.3, fontSize: 12, color: C.muted, align: "center", margin: 0 });
    }
  });

  await addIcon(s, FaPhone, 8.8, 0.3, 0.5, 0.5, C.accent2);
  addPausa(s, "Como el tel\u00E9fono descompuesto: el mensaje (gradiente) se debilita en cada capa", 0.8, 4.6, 8.4);
}

// ============================================================
// SLIDE 8 — VANISHING GRADIENTS
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("Gradientes Vanishing", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // Visual: bar decreasing
  const bars = [
    { label: "1 capa", pct: 100, color: C.green },
    { label: "2 capas", pct: 25, color: C.accent },
    { label: "3 capas", pct: 6, color: C.accent2 },
    { label: "5 capas", pct: 0.4, color: C.amber },
    { label: "10 capas", pct: 0.0001, color: C.red },
  ];
  bars.forEach((b, i) => {
    const by = 1.2 + i * 0.65;
    s.addText(b.label, { x: 0.8, y: by, w: 1.2, h: 0.4, fontSize: 13, fontFace: F.title, color: C.text, bold: true, valign: "middle", margin: 0 });
    const w = Math.max(0.1, b.pct * 0.06);
    s.addShape(pres.shapes.RECTANGLE, { x: 2.2, y: by + 0.08, w, h: 0.3, fill: { color: b.color, transparency: 20 }, line: { color: b.color, width: 1.5 } });
    s.addText(b.pct >= 1 ? `${b.pct}\u00D7` : `${b.pct}\u00D7`, { x: 2.3 + w, y: by + 0.08, w: 1.2, h: 0.3, fontSize: 10, fontFace: F.code, color: C.muted, valign: "middle", margin: 0 });
  });

  await addIcon(s, FaVolumeDown, 8.8, 0.3, 0.5, 0.5, C.accent2);
  s.addText("sigmoid'(z) \u2264 0.25  |  10 capas \u2192 gradiente 0.000001x original", {
    x: 0.8, y: 4.6, w: 8.4, h: 0.3, fontSize: 14, fontFace: F.code, color: C.amber, bold: true, align: "center", margin: 0,
  });
}

// ============================================================
// SLIDE 9 — VALUE NODE
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("Value Node", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // Visual Value box
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 1.1, w: 3.5, h: 2.5, fill: { color: C.card }, line: { color: C.accent, width: 2.5 }, shadow: makeShadow() });
  s.addText("Value", { x: 0.8, y: 1.15, w: 3.5, h: 0.4, fontSize: 20, fontFace: F.title, color: C.accent, bold: true, align: "center", margin: 0 });

  const compartments = [
    { label: "data = 5.0", sub: "valor num\u00E9rico", y: 1.6, c: C.accent },
    { label: "grad = 0.0", sub: "gradiente acumulado", y: 2.2, c: C.accent2 },
    { label: "_backward()", sub: "reparte el error", y: 2.8, c: C.green },
  ];
  compartments.forEach((c) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 1.0, y: c.y, w: 3.1, h: 0.45, fill: { color: c.c, transparency: 90 }, line: { color: c.c, width: 0.5 } });
    s.addText(c.label, { x: 1.1, y: c.y, w: 2.0, h: 0.45, fontSize: 12, fontFace: F.code, color: c.c, bold: true, valign: "middle", margin: 0 });
    s.addText(c.sub, { x: 2.6, y: c.y, w: 1.4, h: 0.45, fontSize: 9, fontFace: F.body, color: C.muted, valign: "middle", margin: 0 });
  });

  await addIcon(s, FaProjectDiagram, 8.8, 0.3, 0.5, 0.5, C.accent2);
  addPausa(s, "Cada n\u00FAmero = una ficha de domin\u00F3 con valor, error acumulado y c\u00F3mo repartirlo", 0.8, 3.9, 8.4);

  addCodeBox(s,
    "class Value:\n    def __init__(self, data, children=(), op=''):\n        self.data = data\n        self.grad = 0.0\n        self._backward = lambda: None\n        self._children = set(children)\n        self._op = op",
    4.8, 1.1, 4.8, 2.5);
}

// ============================================================
// SLIDE 10 — ADD Y MUL
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("Add y Mul", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // Visual: Add vs Mul split
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 1.1, w: 4.0, h: 1.8, fill: { color: C.card }, line: { color: C.green, width: 2 }, shadow: makeShadow() });
  s.addText("SUMA", { x: 0.8, y: 1.15, w: 4.0, h: 0.35, fontSize: 18, fontFace: F.title, color: C.green, bold: true, align: "center", margin: 0 });
  // Visual: two equal halves
  s.addShape(pres.shapes.RECTANGLE, { x: 1.0, y: 1.6, w: 1.6, h: 0.5, fill: { color: C.green, transparency: 30 }, line: { color: C.green, width: 1 } });
  s.addText("a", { x: 1.0, y: 1.6, w: 1.6, h: 0.5, fontSize: 16, fontFace: F.code, color: C.green, align: "center", valign: "middle", margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 3.0, y: 1.6, w: 1.6, h: 0.5, fill: { color: C.green, transparency: 30 }, line: { color: C.green, width: 1 } });
  s.addText("b", { x: 3.0, y: 1.6, w: 1.6, h: 0.5, fontSize: 16, fontFace: F.code, color: C.green, align: "center", valign: "middle", margin: 0 });
  s.addText("da = 1  db = 1", { x: 0.8, y: 2.2, w: 4.0, h: 0.3, fontSize: 11, fontFace: F.code, color: C.muted, align: "center", margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 1.1, w: 4.0, h: 1.8, fill: { color: C.card }, line: { color: C.accent2, width: 2 }, shadow: makeShadow() });
  s.addText("MULTIPLICACI\u00D3N", { x: 5.4, y: 1.15, w: 4.0, h: 0.35, fontSize: 18, fontFace: F.title, color: C.accent2, bold: true, align: "center", margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.6, y: 1.6, w: 1.6, h: 0.5, fill: { color: C.accent2, transparency: 30 }, line: { color: C.accent2, width: 1 } });
  s.addText("a", { x: 5.6, y: 1.6, w: 1.6, h: 0.5, fontSize: 16, fontFace: F.code, color: C.accent2, align: "center", valign: "middle", margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.6, y: 1.6, w: 1.6, h: 0.5, fill: { color: C.accent2, transparency: 30 }, line: { color: C.accent2, width: 1 } });
  s.addText("b", { x: 7.6, y: 1.6, w: 1.6, h: 0.5, fontSize: 16, fontFace: F.code, color: C.accent2, align: "center", valign: "middle", margin: 0 });
  s.addText("da = b  db = a", { x: 5.4, y: 2.2, w: 4.0, h: 0.3, fontSize: 11, fontFace: F.code, color: C.muted, align: "center", margin: 0 });

  await addIcon(s, FaBalanceScale, 8.8, 0.3, 0.5, 0.5, C.accent2);
  addPausa(s, "El += es cr\u00EDtico: un Value usado varias veces acumula gradientes de todos los caminos", 0.8, 3.2, 8.4);

  addCodeBox(s,
    "# SUMA: d(a+b)/da = 1  d(a+b)/db = 1\ndef _backward():\n    self.grad += out.grad\n    other.grad += out.grad",
    0.8, 3.8, 4.0, 1.2);
  addCodeBox(s,
    "# MULT: d(a*b)/da = b  d(a*b)/db = a\ndef _backward():\n    self.grad += other.data * out.grad\n    other.grad += self.data * out.grad",
    5.4, 3.8, 4.0, 1.2);
}

// ============================================================
// SLIDE 11 — SIGMOIDE Y MSE
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("Sigmoide y Funci\u00F3n de P\u00E9rdida", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 32, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // Sigmoid curve approximation
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 1.1, w: 4.5, h: 2.3, fill: { color: C.card }, shadow: makeShadow() });
  s.addText("\u03C3(x) = 1 / (1 + e\u207B\u02E3)", { x: 0.8, y: 1.15, w: 4.5, h: 0.35, fontSize: 14, fontFace: F.code, color: C.accent2, bold: true, align: "center", margin: 0 });
  // Draw curve with rectangles
  const pts = [
    { x: 1.2, y: 2.9 }, { x: 1.6, y: 2.85 }, { x: 2.0, y: 2.7 },
    { x: 2.4, y: 2.4 }, { x: 2.8, y: 2.0 }, { x: 3.2, y: 1.65 },
    { x: 3.6, y: 1.45 }, { x: 4.0, y: 1.35 }, { x: 4.4, y: 1.3 },
  ];
  pts.forEach((p) => {
    s.addShape(pres.shapes.OVAL, { x: p.x, y: p.y, w: 0.08, h: 0.08, fill: { color: C.accent2 } });
  });
  s.addText("x", { x: 4.5, y: 3.0, w: 0.3, h: 0.2, fontSize: 9, fontFace: F.code, color: C.muted, margin: 0 });
  s.addText("\u03C3(x)", { x: 0.5, y: 1.1, w: 0.4, h: 0.2, fontSize: 9, fontFace: F.code, color: C.muted, margin: 0 });
  s.addText("0.5", { x: 0.5, y: 2.0, w: 0.4, h: 0.2, fontSize: 9, fontFace: F.code, color: C.muted, margin: 0 });

  await addIcon(s, FaThermometerHalf, 9.0, 0.3, 0.4, 0.5, C.accent2);
  addPausa(s, "MSE = (predicho - target)\u00B2  |  Como un term\u00F3metro: \u00BFqu\u00E9 tan lejos de la temperatura ideal?", 0.8, 3.7, 8.4);

  addCodeBox(s,
    "def sigmoid(self):\n    s = 1/(1 + math.exp(-x))\n    def _backward():\n        self.grad += s*(1-s) * out.grad\n    return Value(s, (self,), 'sigmoid')\n\ndef mse_loss(pred, target):\n    diff = pred + Value(-target)\n    return diff * diff",
    5.8, 1.1, 3.8, 2.8);
}

// ============================================================
// SLIDE 12 — BACKWARD PASS
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("Backward Pass", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // Topo sort visual
  const topoNodes = [
    { label: "L", n: 1, x: 7.5, y: 1.3 },
    { label: "a", n: 2, x: 6.0, y: 1.8 },
    { label: "z\u2082", n: 3, x: 4.5, y: 1.3 },
    { label: "b", n: 4, x: 3.0, y: 1.8 },
    { label: "z\u2081", n: 5, x: 3.0, y: 2.8 },
    { label: "w", n: 6, x: 1.2, y: 2.8 },
    { label: "x", n: 7, x: 1.2, y: 1.8 },
  ];
  topoNodes.forEach((node) => {
    s.addShape(pres.shapes.RECTANGLE, { x: node.x, y: node.y, w: 0.7, h: 0.5, fill: { color: C.card }, line: { color: C.accent, width: 1.5 }, shadow: makeShadow() });
    s.addText(node.label, { x: node.x, y: node.y, w: 0.7, h: 0.5, fontSize: 14, fontFace: F.code, color: C.accent, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(String(node.n), { x: node.x + 0.5, y: node.y - 0.2, w: 0.3, h: 0.2, fontSize: 9, fontFace: F.code, color: C.green, align: "center", margin: 0 });
  });
  s.addText("Orden de procesamiento inverso: 1 \u2192 7", { x: 0.8, y: 3.5, w: 4.0, h: 0.3, fontSize: 10, fontFace: F.body, color: C.muted, margin: 0 });

  await addIcon(s, FaProjectDiagram, 8.8, 0.3, 0.5, 0.5, C.accent2);
  addPausa(s, "Repartiendo culpas: el jefe (Loss) dice \"la culpa es m\u00EDa = 1\" y cada empleado reparte su parte", 0.8, 3.9, 8.4);

  addCodeBox(s,
    "def backward(self):\n    topo = []\n    visited = set()\n    def build_topo(v):\n        if v not in visited:\n            visited.add(v)\n            for child in v._children:\n                build_topo(child)\n            topo.append(v)\n    build_topo(self)\n    self.grad = 1.0\n    for v in reversed(topo):\n        v._backward()",
    0.8, 4.2, 8.8, 1.2);
}

// ============================================================
// SLIDE 13 — ARQUITECTURA
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("Arquitectura", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // Hierarchy visual
  const levels = [
    { label: "Neuron", sub: "pesos + bias + sigmoide", items: ["w\u2081, w\u2082, ...", "bias b", "\u03C3(w\u00B7x + b)"], color: C.green },
    { label: "Layer", sub: "conjunto de Neuronas", items: ["n_inputs \u2192 n_outputs", "lista de Neuronas"], color: C.accent },
    { label: "Network", sub: "conjunto de Layers", items: ["sizes = [2, 4, 1]", "parameters(), zero_grad()"], color: C.accent2 },
  ];
  levels.forEach((lv, i) => {
    const lx = 0.8 + i * 3.1;
    s.addShape(pres.shapes.RECTANGLE, { x: lx, y: 1.1, w: 2.7, h: 2.0, fill: { color: C.card }, line: { color: lv.color, width: 2.5 }, shadow: makeShadow() });
    s.addText(lv.label, { x: lx + 0.1, y: 1.15, w: 2.5, h: 0.35, fontSize: 20, fontFace: F.title, color: lv.color, bold: true, margin: 0 });
    s.addText(lv.sub, { x: lx + 0.1, y: 1.5, w: 2.5, h: 0.25, fontSize: 10, fontFace: F.body, color: C.muted, margin: 0 });
    s.addText(lv.items.map((t) => ({ text: t, options: { bullet: true, breakLine: true } })), {
      x: lx + 0.1, y: 1.85, w: 2.5, h: 1.1, fontSize: 12, fontFace: F.body, color: C.text, valign: "top", margin: 0, paraSpaceAfter: 3,
    });
    if (i < 2) {
      s.addShape(pres.shapes.LINE, { x: lx + 2.65, y: 2.0, w: 0.4, h: 0, line: { color: C.muted, width: 2 } });
      s.addText("\u25B6", { x: lx + 2.7, y: 1.85, w: 0.3, h: 0.3, fontSize: 14, color: C.muted, align: "center", margin: 0 });
    }
  });

  await addIcon(s, FaLayerGroup, 8.8, 0.3, 0.5, 0.5, C.accent2);
  addCodeBox(s,
    "net = Network([2, 4, 1])\nnet.zero_grad()\ntotal_loss.backward()\nfor p in net.parameters():\n    p.data -= lr * p.grad",
    0.8, 3.4, 8.8, 1.5);
}

// ============================================================
// SLIDE 14 — XOR
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("XOR: Ejemplo Pr\u00E1ctico", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // XOR visual: 4 points on a grid
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 1.1, w: 3.5, h: 2.5, fill: { color: C.card }, shadow: makeShadow() });

  // Grid lines
  s.addShape(pres.shapes.LINE, { x: 1.5, y: 1.3, w: 0, h: 2.1, line: { color: C.cardBorder, width: 0.5 } });
  s.addShape(pres.shapes.LINE, { x: 1.5, y: 2.35, w: 2.5, h: 0, line: { color: C.cardBorder, width: 0.5 } });

  // Points: (0,0)=0, (0,1)=1, (1,0)=1, (1,1)=0
  const pts = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
  pts.forEach((p) => {
    const px = 1.5 + p[0] * 1.8;
    const py = 1.3 + (1 - p[1]) * 1.8;
    const col = p[2] === 0 ? C.accent : C.green;
    s.addShape(pres.shapes.OVAL, { x: px - 0.15, y: py - 0.15, w: 0.3, h: 0.3, fill: { color: col } });
    s.addText(String(p[2]), { x: px - 0.1, y: py - 0.1, w: 0.3, h: 0.3, fontSize: 14, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  });

  // Axis labels
  s.addText("x\u2081", { x: 4.0, y: 3.1, w: 0.3, h: 0.2, fontSize: 9, fontFace: F.code, color: C.muted, margin: 0 });
  s.addText("x\u2082", { x: 0.5, y: 1.1, w: 0.3, h: 0.2, fontSize: 9, fontFace: F.code, color: C.muted, margin: 0 });
  s.addText("0 1", { x: 1.5, y: 3.1, w: 2.0, h: 0.2, fontSize: 8, fontFace: F.code, color: C.muted, align: "center", margin: 0 });

  // Results table
  const results = [
    { input: "[0,0]", output: "0", expected: "0" },
    { input: "[0,1]", output: "1", expected: "1" },
    { input: "[1,0]", output: "1", expected: "1" },
    { input: "[1,1]", output: "0", expected: "0" },
  ];
  const rx = 4.8;
  ["Input", "Pred", "Esp"].forEach((h, i) => {
    s.addText(h, { x: rx + i * 0.9, y: 1.1, w: 0.8, h: 0.3, fontSize: 10, fontFace: F.title, color: C.accent, bold: true, margin: 0 });
  });
  results.forEach((r, i) => {
    const ry = 1.5 + i * 0.35;
    s.addText(r.input, { x: rx, y: ry, w: 0.8, h: 0.3, fontSize: 11, fontFace: F.code, color: C.text, margin: 0 });
    s.addText(r.output, { x: rx + 0.9, y: ry, w: 0.5, h: 0.3, fontSize: 11, fontFace: F.code, color: C.green, bold: true, margin: 0 });
    s.addText(r.expected, { x: rx + 1.8, y: ry, w: 0.5, h: 0.3, fontSize: 11, fontFace: F.code, color: C.muted, margin: 0 });
  });

  await addIcon(s, FaBullseye, 8.8, 0.3, 0.5, 0.5, C.accent2);
  addPausa(s, "\u00BFPor qu\u00E9 1 neurona no puede separar XOR? \u00BFQu\u00E9 cambia con una capa oculta?", 0.8, 4.8, 8.4);
  addCodeBox(s,
    "net = Network([2, 4, 1])\nfor epoch in range(1000):\n    total_loss = Value(0.0)\n    for inputs, target in xor_data:\n        x = [Value(i) for i in inputs]\n        loss = mse_loss(net(x), target)\n        total_loss += loss\n    net.zero_grad()\n    total_loss.backward()\n    for p in net.parameters():\n        p.data -= lr * p.grad",
    0.8, 3.6, 8.8, 1.6);
}

// ============================================================
// SLIDE 15 — CÍRCULOS
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("Clasificar C\u00EDrculos", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // Circle visual
  s.addShape(pres.shapes.OVAL, { x: 1.5, y: 1.2, w: 2.5, h: 2.5, fill: { color: C.accent, transparency: 92 }, line: { color: C.accent, width: 2 } });
  s.addText("x\u00B2 + y\u00B2 < 1", { x: 2.0, y: 2.2, w: 1.5, h: 0.3, fontSize: 11, fontFace: F.code, color: C.accent, align: "center", margin: 0 });

  // Points inside/outside
  const inside = [[1.8, 1.9], [2.5, 1.6], [2.2, 2.5]];
  inside.forEach((p) => {
    s.addShape(pres.shapes.OVAL, { x: p[0], y: p[1], w: 0.12, h: 0.12, fill: { color: C.green } });
  });
  const outside = [[0.8, 1.2], [4.5, 1.9], [4.0, 3.2]];
  outside.forEach((p) => {
    s.addShape(pres.shapes.OVAL, { x: p[0], y: p[1], w: 0.12, h: 0.12, fill: { color: C.red } });
  });

  // Accuracy
  s.addShape(pres.shapes.RECTANGLE, { x: 4.8, y: 1.2, w: 2.0, h: 1.0, fill: { color: C.card }, shadow: makeShadow() });
  s.addText("~50%", { x: 4.8, y: 1.2, w: 2.0, h: 0.5, fontSize: 36, fontFace: F.title, color: C.amber, bold: true, align: "center", margin: 0 });
  s.addText("Accuracy inicial", { x: 4.8, y: 1.7, w: 2.0, h: 0.3, fontSize: 10, fontFace: F.body, color: C.muted, align: "center", margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, { x: 7.2, y: 1.2, w: 2.0, h: 1.0, fill: { color: C.card }, shadow: makeShadow() });
  s.addText(">95%", { x: 7.2, y: 1.2, w: 2.0, h: 0.5, fontSize: 36, fontFace: F.title, color: C.green, bold: true, align: "center", margin: 0 });
  s.addText("Accuracy final", { x: 7.2, y: 1.7, w: 2.0, h: 0.3, fontSize: 10, fontFace: F.body, color: C.muted, align: "center", margin: 0 });

  s.addText("\u2192", { x: 6.8, y: 1.3, w: 0.4, h: 0.5, fontSize: 28, color: C.muted, align: "center", margin: 0 });

  await addIcon(s, FaDna, 8.8, 0.3, 0.5, 0.5, C.accent2);
  s.addText("La red descubre el l\u00EDmite circular por s\u00ED sola, sin instrucciones expl\u00EDcitas", {
    x: 0.8, y: 4.0, w: 8.4, h: 0.3, fontSize: 12, fontFace: F.body, color: C.green, italic: true, align: "center", margin: 0,
  });

  addCodeBox(s,
    "circle_net = Network([2, 8, 1])\nfor epoch in range(2000):\n    random.shuffle(data)\n    for inputs, target in data:\n        x = [Value(i) for i in inputs]\n        pred = circle_net(x)\n        loss = mse_loss(pred, target)\n        circle_net.zero_grad()\n        loss.backward()\n        for p in circle_net.parameters():\n            p.data -= lr * p.grad",
    0.8, 4.4, 8.8, 1.1);
}

// ============================================================
// SLIDE 16 — PYTORCH
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("PyTorch", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // Bridge visual
  s.addText("Tu C\u00F3digo", { x: 0.8, y: 1.2, w: 2.5, h: 0.35, fontSize: 18, fontFace: F.title, color: C.accent, bold: true, align: "center", margin: 0 });
  s.addText("PyTorch", { x: 6.8, y: 1.2, w: 2.5, h: 0.35, fontSize: 18, fontFace: F.title, color: C.accent2, bold: true, align: "center", margin: 0 });

  // Bridge shape
  s.addShape(pres.shapes.RECTANGLE, { x: 3.3, y: 1.7, w: 3.4, h: 0.15, fill: { color: C.accent, transparency: 70 } });

  const pairs = [
    ["total_loss.backward()", "loss.backward()"],
    ["p.data -= lr * p.grad", "optimizer.step()"],
    ["net.zero_grad()", "optimizer.zero_grad()"],
  ];
  pairs.forEach((p, i) => {
    const py = 2.1 + i * 0.65;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: py, w: 2.5, h: 0.45, fill: { color: C.card }, line: { color: C.accent, width: 1 }, shadow: makeShadow() });
    s.addText(p[0], { x: 0.9, y: py, w: 2.3, h: 0.45, fontSize: 10, fontFace: F.code, color: C.text, valign: "middle", margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, { x: 6.8, y: py, w: 2.5, h: 0.45, fill: { color: C.card }, line: { color: C.accent2, width: 1 }, shadow: makeShadow() });
    s.addText(p[1], { x: 6.9, y: py, w: 2.3, h: 0.45, fontSize: 10, fontFace: F.code, color: C.text, valign: "middle", margin: 0 });

    s.addText("\u2194", { x: 4.6, y: py, w: 0.6, h: 0.45, fontSize: 16, color: C.muted, align: "center", valign: "middle", margin: 0 });
  });

  await addIcon(s, FaCheckDouble, 8.8, 0.3, 0.5, 0.5, C.accent2);
  s.addText("Mismo algoritmo. Escala industrial. GPUs. Terabytes de datos.", {
    x: 0.8, y: 4.4, w: 8.4, h: 0.4, fontSize: 14, fontFace: F.body, color: C.accent, bold: true, align: "center", margin: 0,
  });
}

// ============================================================
// SLIDE 17 — INFERENCIA VS TRAINING
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("Inferencia vs Entrenamiento", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  // Training
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 1.2, w: 4.0, h: 1.8, fill: { color: C.card }, line: { color: C.accent, width: 2.5 }, shadow: makeShadow() });
  s.addShape(pres.shapes.OVAL, { x: 1.8, y: 1.3, w: 0.5, h: 0.5, fill: { color: C.accent } });
  s.addText("\u21BB", { x: 1.8, y: 1.3, w: 0.5, h: 0.5, fontSize: 20, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  s.addText("Entrenamiento", { x: 0.8, y: 1.85, w: 4.0, h: 0.35, fontSize: 18, fontFace: F.title, color: C.accent, bold: true, align: "center", margin: 0 });
  s.addText("Forward + Backward + Update", { x: 0.8, y: 2.2, w: 4.0, h: 0.3, fontSize: 13, fontFace: F.body, color: C.text, align: "center", margin: 0 });
  s.addText("Los pesos cambian", { x: 0.8, y: 2.45, w: 4.0, h: 0.3, fontSize: 12, fontFace: F.body, color: C.muted, align: "center", margin: 0 });

  // Inference
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 1.2, w: 4.0, h: 1.8, fill: { color: C.card }, line: { color: C.green, width: 2.5 }, shadow: makeShadow() });
  s.addShape(pres.shapes.OVAL, { x: 6.4, y: 1.3, w: 0.5, h: 0.5, fill: { color: C.green } });
  s.addText("\u2192", { x: 6.4, y: 1.3, w: 0.5, h: 0.5, fontSize: 20, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  s.addText("Inferencia", { x: 5.4, y: 1.85, w: 4.0, h: 0.35, fontSize: 18, fontFace: F.title, color: C.green, bold: true, align: "center", margin: 0 });
  s.addText("Forward \u00FAnicamente", { x: 5.4, y: 2.2, w: 4.0, h: 0.3, fontSize: 13, fontFace: F.body, color: C.text, align: "center", margin: 0 });
  s.addText("Sin cambios en pesos", { x: 5.4, y: 2.45, w: 4.0, h: 0.3, fontSize: 12, fontFace: F.body, color: C.muted, align: "center", margin: 0 });

  s.addText("Estudiar = entrenamiento. Presentar examen = inferencia.", {
    x: 0.8, y: 3.3, w: 8.4, h: 0.3, fontSize: 12, fontFace: F.body, color: C.amber, italic: true, align: "center", margin: 0,
  });
  s.addText("ChatGPT/Claude usan inferencia: tu prompt fluye forward, ning\u00FAn peso cambia", {
    x: 0.8, y: 3.7, w: 8.4, h: 0.3, fontSize: 12, fontFace: F.body, color: C.text, align: "center", margin: 0,
  });
  await addIcon(s, FaArrowRight, 8.8, 0.3, 0.5, 0.5, C.accent2);
  s.addText("Entender backpropagation importa: molde\u00F3 cada peso de esa red", {
    x: 0.8, y: 4.1, w: 8.4, h: 0.3, fontSize: 13, fontFace: F.body, color: C.accent, italic: true, bold: true, align: "center", margin: 0,
  });
}

// ============================================================
// SLIDE 18 — TÉRMINOS CLAVE
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("T\u00E9rminos Clave", { x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 36, fontFace: F.title, color: C.text, bold: true, margin: 0 });

  const terms = [
    { term: "Backpropagation", def: "Asigna culpa (gradientes) a cada peso v\u00EDa chain rule" },
    { term: "Grafo Computacional", def: "Mapa de operaciones: nodos = c\u00E1lculos" },
    { term: "Regla de la Cadena", def: "Multiplicar derivadas a lo largo de la cadena" },
    { term: "Gradiente", def: "C\u00F3mo cambiar\u00EDa el error si ajustas un peso" },
    { term: "Vanishing Gradient", def: "Gradiente se encoge exponencialmente con sigmoide" },
    { term: "Forward Pass", def: "Calcular salida, guardar valores intermedios" },
    { term: "Backward Pass", def: "Recorrer el grafo al rev\u00E9s repartiendo gradientes" },
    { term: "Learning Rate", def: "Tama\u00F1o del ajuste: lr peque\u00F1o = lento" },
    { term: "Topological Sort", def: "Orden: hijos procesados antes que padres" },
  ];

  await addIcon(s, FaBook, 8.8, 0.3, 0.5, 0.5, C.accent2);
  terms.forEach((t, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const cx = 0.8 + col * 3.0;
    const cy = 1.1 + row * 1.35;

    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: cy, w: 2.6, h: 1.15, fill: { color: C.card }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: cy, w: 0.05, h: 1.15, fill: { color: C.accent } });
    s.addText(t.term, { x: cx + 0.15, y: cy + 0.08, w: 2.3, h: 0.3, fontSize: 13, fontFace: F.title, color: C.accent, bold: true, margin: 0 });
    s.addText(t.def, { x: cx + 0.15, y: cy + 0.4, w: 2.3, h: 0.65, fontSize: 10, fontFace: F.body, color: C.text, valign: "top", margin: 0 });
  });
}

// ============================================================
// SLIDE 19 — CIERRE
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.card };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.accent } });

  s.addText("Lo que aprendimos", { x: 0.8, y: 0.4, w: 8.4, h: 0.6, fontSize: 32, fontFace: F.title, color: C.text, bold: true, align: "center", margin: 0 });

  const items = [
    "Backprop = 1 backward pass calcula TODOS los gradientes",
    "Regla de la cadena: multiplicar derivadas hacia atr\u00E1s",
    "Value: almacena data, grad y c\u00F3mo repartir el error",
    "Suma reparte igual, Multiplicaci\u00F3n reparte seg\u00FAn el otro valor",
    "Sigmoide profundo = gradientes que desaparecen",
    "Construiste el mismo algoritmo que usa PyTorch",
  ];

  await addIcon(s, FaCheckCircle, 8.8, 0.3, 0.5, 0.5, C.accent2);
  items.forEach((item, i) => {
    const iy = 1.2 + i * 0.5;
    s.addShape(pres.shapes.OVAL, { x: 1.0, y: iy + 0.05, w: 0.25, h: 0.25, fill: { color: C.green } });
    s.addText(item, { x: 1.4, y: iy, w: 7.5, h: 0.35, fontSize: 13, fontFace: F.body, color: C.text, valign: "middle", margin: 0 });
  });

  s.addText("Gracias", { x: 0.8, y: 4.3, w: 8.4, h: 0.6, fontSize: 32, fontFace: F.title, color: C.accent, bold: true, align: "center", margin: 0 });
  s.addText("Pr\u00F3ximo: Lesson 04 \u2014 ReLU y variantes", {
    x: 0.8, y: 4.9, w: 8.4, h: 0.3, fontSize: 12, fontFace: F.body, color: C.muted, align: "center", margin: 0,
  });
}

// ============================================================
// WRITE
// ============================================================
  pres.writeFile({ fileName: "Backpropagation_from_Scratch.pptx" }).then(() => {
    console.log("Presentation created: Backpropagation_from_Scratch.pptx");
  });
}

build().catch((err) => console.error(err));
