// The pilot's neural network in 3D: sensors on the left, two hidden layers as grids,
// buttons on the right. Connections glow with the signal flowing through them.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { ACTIONS, INPUTS, THROTTLE_OUTPUTS } from "./nn.js";

const NEGATIVE = new THREE.Color("#3987e5");
const POSITIVE = new THREE.Color("#d95926");
const IDLE = new THREE.Color("#262624");
const WHITE = new THREE.Color("#ffffff");

export function createBrain({ canvas, labels, view }) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
  camera.position.set(4, 4, 24);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.enableZoom = false;
  controls.enablePan = false;
  controls.minAzimuthAngle = -1.2; // stay in front, where sensors are on the left
  controls.maxAzimuthAngle = 1.2;
  controls.target.set(0, -0.3, 0);
  let userMoved = false; // until the user drags, the view sways gently on its own
  let hovering = false; // ...and holds still while the pointer is over it
  let swayTime = 0;
  let lastRender = performance.now();
  controls.addEventListener("start", () => { userMoved = true; });

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.7, 0.45, 0.08);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  const labelRenderer = new CSS2DRenderer({ element: labels });

  let net = null; // layout + geometry for the current network shape
  let mode = "signal";
  let focus = null; // {layer, index}
  let selected = null;
  let last = null; // last forward() result
  let onSelect = () => {};
  const group = new THREE.Group();
  scene.add(group);

  function layout(sizes) {
    const L = sizes.length;
    return sizes.map((n, l) => {
      const x = -10.5 + (21 * l) / (L - 1);
      if (l === 0 || l === L - 1) {
        const gap = l === 0 ? 1.0 : 1.35;
        return Array.from({ length: n }, (_, i) => new THREE.Vector3(x, ((n - 1) / 2 - i) * gap, 0));
      }
      const cols = Math.ceil(Math.sqrt(n));
      const rows = Math.ceil(n / cols);
      const gap = Math.min(0.95, 7 / Math.max(cols, rows));
      return Array.from({ length: n }, (_, i) => {
        const r = Math.floor(i / cols), c = i % cols;
        return new THREE.Vector3(x, ((rows - 1) / 2 - r) * gap, (c - (cols - 1) / 2) * gap);
      });
    });
  }

  function setNetwork(weights) {
    const sizes = [weights.actor[0].w[0].length, ...weights.actor.map((l) => l.w.length)];
    const key = `${sizes.join("x")}:${weights.control}`;
    if (!net || net.key !== key) {
      build(sizes, weights.control === "throttle" ? THROTTLE_OUTPUTS : ACTIONS);
      net.key = key;
      last = null; // activations from a different network shape no longer fit
    }
    net.weights = weights;
    // Weights are fixed per network; signals are recomputed on every frame.
    let s = 0;
    weights.actor.forEach(({ w }) => {
      for (let j = 0; j < w.length; j++) for (let i = 0; i < w[j].length; i++) net.segmentWeight[s++] = w[j][i];
    });
    if (mode === "weights") paintWeights();
    else if (last) update(last);
  }

  function build(sizes, outputNames) {
    group.clear();
    labels.querySelectorAll(".node-label, .layer-label").forEach((el) => el.remove());
    const positions = layout(sizes);
    const offsets = [];
    let count = 0;
    sizes.forEach((n) => { offsets.push(count); count += n; });

    const nodes = new THREE.InstancedMesh(
      new THREE.SphereGeometry(1, 18, 14),
      new THREE.MeshBasicMaterial({ toneMapped: false }),
      count,
    );
    const base = [];
    const m = new THREE.Matrix4();
    positions.forEach((layer, l) => layer.forEach((p, i) => {
      const r = l === 0 || l === sizes.length - 1 ? 0.2 : 0.13;
      base.push(r);
      m.makeScale(r, r, r).setPosition(p);
      nodes.setMatrixAt(offsets[l] + i, m);
      nodes.setColorAt(offsets[l] + i, IDLE);
    }));
    group.add(nodes);

    let segments = 0;
    for (let l = 0; l < sizes.length - 1; l++) segments += sizes[l] * sizes[l + 1];
    const linePositions = new Float32Array(segments * 6);
    const segmentFrom = new Uint32Array(segments), segmentTo = new Uint32Array(segments);
    const segmentLayer = new Uint8Array(segments);
    let s = 0;
    for (let l = 0; l < sizes.length - 1; l++) {
      for (let j = 0; j < sizes[l + 1]; j++) {
        for (let i = 0; i < sizes[l]; i++) {
          const a = positions[l][i], b = positions[l + 1][j];
          linePositions.set([a.x, a.y, a.z, b.x, b.y, b.z], s * 6);
          segmentFrom[s] = offsets[l] + i;
          segmentTo[s] = offsets[l + 1] + j;
          segmentLayer[s] = l;
          s++;
        }
      }
    }
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
    lineGeometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(segments * 6), 3));
    const lines = new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
    }));
    group.add(lines);

    // Labels: sensor names on the left, buttons on the right, layer titles on top.
    const label = (text, cls, position, anchorX = 0.5) => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      const obj = new CSS2DObject(el);
      obj.position.copy(position);
      obj.center.set(anchorX, 0.5); // which point of the label sits on the position
      group.add(obj);
      return el;
    };
    const inputLabels = positions[0].map((p, i) => {
      const el = label("", "node-label left", p.clone().add(new THREE.Vector3(-0.45, 0, 0)), 1);
      el.dataset.name = INPUTS[i] || `input ${i + 1}`;
      return el;
    });
    const outputLabels = positions[sizes.length - 1].map((p, i) => {
      const el = label("", "node-label", p.clone().add(new THREE.Vector3(0.5, 0, 0)), 0);
      el.dataset.name = outputNames[i] || `output ${i + 1}`;
      return el;
    });
    const top = Math.max(...positions.flat().map((p) => p.y)) + 1.4;
    sizes.forEach((n, l) => {
      const name = l === 0 ? "Sensors" : l === sizes.length - 1 ? (outputNames === ACTIONS ? "Buttons" : "Throttle") : `Hidden ${l}`;
      label(`${name} · ${n}`, "layer-label", new THREE.Vector3(positions[l][0].x, top, 0));
    });

    net = {
      key: sizes.join("x"), sizes, positions, offsets, count, nodes, base, lines,
      segmentFrom, segmentTo, segmentLayer, segmentWeight: new Float32Array(segments),
      segmentSignal: new Float32Array(segments), inputLabels, outputLabels, weights: null,
    };
  }

  function nodeColor(value, out) {
    const v = Math.max(-1, Math.min(1, value));
    return out.copy(IDLE).lerp(v >= 0 ? POSITIVE : NEGATIVE, Math.pow(Math.abs(v), 0.8)).multiplyScalar(1 + Math.abs(v) * 1.4);
  }

  function focusSet() {
    const target = selected || focus;
    if (!target) return null;
    return net.offsets[target.layer] + target.index;
  }

  function paintSegments(valueOf) {
    const colors = net.lines.geometry.attributes.color.array;
    const focusId = focusSet();
    const maxPerLayer = new Float32Array(net.sizes.length).fill(1e-6);
    const n = net.segmentWeight.length;
    for (let s = 0; s < n; s++) {
      const v = Math.abs(valueOf(s));
      if (v > maxPerLayer[net.segmentLayer[s]]) maxPerLayer[net.segmentLayer[s]] = v;
    }
    for (let s = 0; s < n; s++) {
      const v = valueOf(s);
      // Only the strongest signals light up, so individual paths stay readable.
      const share = Math.abs(v) / maxPerLayer[net.segmentLayer[s]];
      let k = Math.pow(Math.max(0, (share - 0.18) / 0.82), 1.7) * 0.75;
      if (focusId !== null) {
        const touches = net.segmentFrom[s] === focusId || net.segmentTo[s] === focusId;
        k = touches ? Math.max(k, 0.12) * 1.8 : k * 0.06;
      }
      const c = v >= 0 ? POSITIVE : NEGATIVE;
      const o = s * 6;
      colors[o] = colors[o + 3] = c.r * k;
      colors[o + 1] = colors[o + 4] = c.g * k;
      colors[o + 2] = colors[o + 5] = c.b * k;
    }
    net.lines.geometry.attributes.color.needsUpdate = true;
  }

  function paintWeights() {
    if (!net?.weights) return;
    paintSegments((s) => net.segmentWeight[s]);
    const c = new THREE.Color();
    for (let k = 0; k < net.count; k++) net.nodes.setColorAt(k, c.copy(IDLE).multiplyScalar(2.2));
    net.nodes.instanceColor.needsUpdate = true;
  }

  function update(result) {
    last = result;
    if (!net?.weights) return;
    const acts = result.layers; // [inputs, hidden..., logits]
    if (mode === "signal") {
      let s = 0;
      for (let l = 0; l < net.sizes.length - 1; l++) {
        const src = acts[l];
        const w = net.weights.actor[l].w;
        for (let j = 0; j < w.length; j++) for (let i = 0; i < src.length; i++) net.segmentSignal[s++] = w[j][i] * src[i];
      }
      paintSegments((k) => net.segmentSignal[k]);
    } else {
      paintWeights();
    }

    const c = new THREE.Color(), m = new THREE.Matrix4();
    const relu = net.weights.activation === "relu";
    const throttle = net.weights.control === "throttle";
    const L = net.sizes.length;
    for (let l = 0; l < L; l++) {
      const values = l === L - 1 ? result.outputs : acts[l];
      const scaleTo = relu && l > 0 && l < L - 1 ? Math.max(...values, 1e-6) : 1;
      for (let i = 0; i < values.length; i++) {
        const id = net.offsets[l] + i;
        let v = values[i] / scaleTo;
        if (l === L - 1 && !throttle) {
          c.copy(IDLE).lerp(WHITE, v).multiplyScalar(i === result.action ? 2.2 : 1);
        } else if (l === L - 1 || mode === "signal") {
          nodeColor(v, c);
        } else {
          c.copy(IDLE).multiplyScalar(2.2);
          v = 0;
        }
        const isTarget = selected && selected.layer === l && selected.index === i;
        if (isTarget) c.copy(WHITE).multiplyScalar(2.5);
        net.nodes.setColorAt(id, c);
        const r = net.base[id] * (0.8 + Math.min(Math.abs(v), 1) * 0.55) * (isTarget ? 1.5 : 1);
        m.makeScale(r, r, r).setPosition(net.positions[l][i]);
        net.nodes.setMatrixAt(id, m);
      }
    }
    net.nodes.instanceColor.needsUpdate = true;
    net.nodes.instanceMatrix.needsUpdate = true;

    net.inputLabels.forEach((el, i) => {
      el.innerHTML = "";
      el.append(`${el.dataset.name} `);
      const b = document.createElement("b");
      b.textContent = (acts[0][i] >= 0 ? "+" : "") + acts[0][i].toFixed(2);
      el.append(b);
    });
    net.outputLabels.forEach((el, i) => {
      el.innerHTML = "";
      const b = document.createElement("b");
      if (throttle) { // the raw output, then the engine power it gives
        const v = result.outputs[i];
        b.textContent = `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
        const power = i === 0 ? result.engines.main : result.engines.side;
        const side = i === 1 && power ? (power < 0 ? "left " : "right ") : "";
        el.append(b, ` ${el.dataset.name} → ${power ? `${side}${Math.round(Math.abs(power) * 100)}%` : "off"}`);
      } else {
        b.textContent = `${Math.round(result.probs[i] * 100)}%`;
        el.append(b, ` ${el.dataset.name}${i === result.action ? "  ◀" : ""}`);
      }
    });
  }

  // Hover and click on neurons: pick the one nearest the pointer on screen.
  const projected = new THREE.Vector3();
  let downAt = null;
  function pick(event) {
    if (!net) return null;
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left, py = event.clientY - rect.top;
    let best = null, bestDistance = 16 * 16;
    net.positions.forEach((layer, l) => layer.forEach((position, index) => {
      projected.copy(position).project(camera);
      const dx = ((projected.x + 1) / 2) * rect.width - px;
      const dy = ((1 - projected.y) / 2) * rect.height - py;
      const d = dx * dx + dy * dy;
      if (d < bestDistance) { bestDistance = d; best = { layer: l, index }; }
    }));
    return best;
  }
  canvas.addEventListener("pointermove", (event) => {
    const hit = pick(event);
    canvas.style.cursor = hit ? "pointer" : "grab";
    const changed = JSON.stringify(hit) !== JSON.stringify(focus);
    focus = hit;
    if (changed && last) update(last);
  });
  canvas.addEventListener("pointerdown", (event) => { downAt = [event.clientX, event.clientY]; });
  canvas.addEventListener("pointerup", (event) => {
    if (!downAt || Math.hypot(event.clientX - downAt[0], event.clientY - downAt[1]) > 5) return;
    const hit = pick(event);
    selected = hit && !(selected && hit.layer === selected.layer && hit.index === selected.index) ? hit : null;
    onSelect(selected);
    if (last) update(last);
  });
  canvas.addEventListener("pointerenter", () => { hovering = true; });
  canvas.addEventListener("pointerleave", () => { hovering = false; focus = null; if (last) update(last); });

  function resize() {
    const w = view.clientWidth, h = view.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    labelRenderer.setSize(w, h);
    camera.aspect = w / h;
    // Back off far enough that the network plus its labels (about 15 units each side) fit.
    const halfWidth = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.aspect;
    const distance = Math.min(Math.max(15.5 / halfWidth, 26), 80);
    camera.position.sub(controls.target).setLength(distance).add(controls.target);
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(view);
  resize();

  return {
    setNetwork,
    update,
    setMode(next) { mode = next; if (last) update(last); else paintWeights(); },
    onSelect(fn) { onSelect = fn; },
    clearSelection() { selected = null; if (last) update(last); },
    render() {
      const now = performance.now();
      const dt = Math.min((now - lastRender) / 1000, 0.05);
      lastRender = now;
      if (!userMoved && !hovering) {
        swayTime += dt;
        const r = camera.position.distanceTo(controls.target);
        const azimuth = Math.sin(swayTime * 0.22) * 0.55;
        const polar = 1.36;
        camera.position.set(
          controls.target.x + r * Math.sin(polar) * Math.sin(azimuth),
          controls.target.y + r * Math.cos(polar),
          controls.target.z + r * Math.sin(polar) * Math.cos(azimuth),
        );
      }
      controls.update();
      composer.render();
      labelRenderer.render(scene, camera);
    },
    get ready() { return !!net?.weights; },
  };
}
