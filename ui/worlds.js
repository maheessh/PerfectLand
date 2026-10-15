// What each world looks like: its dotted globe (in the hero, and hanging in other worlds'
// skies) and the sky, ground and weather of the flight deck. Physics lives in lab.py.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Procedural noise for planet surfaces ---------------------------------------------

function hash(x, y, z) {
  let h = Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ Math.imul(z, 1440662683);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function noise(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const fx = x - xi, fy = y - yi, fz = z - zi;
  const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy), w = fz * fz * (3 - 2 * fz);
  const c = (dx, dy, dz) => hash(xi + dx, yi + dy, zi + dz);
  const l = (a, b, t) => a + (b - a) * t;
  return l(
    l(l(c(0, 0, 0), c(1, 0, 0), u), l(c(0, 1, 0), c(1, 1, 0), u), v),
    l(l(c(0, 0, 1), c(1, 0, 1), u), l(c(0, 1, 1), c(1, 1, 1), u), v),
    w,
  );
}

function fbm(x, y, z, octaves = 4) {
  let sum = 0, amp = 0.5, freq = 1, norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amp * noise(x * freq, y * freq, z * freq);
    norm += amp;
    amp *= 0.5;
    freq *= 2.03;
  }
  return sum / norm; // about 0.5 on average
}

const C = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
const smooth = (a, b, v) => {
  const t = Math.min(Math.max((v - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a)); // angle difference in -π..π
const oval = (lon, lat, lon0, lat0, w, h) => (wrap(lon - lon0) / w) ** 2 + ((lat - lat0) / h) ** 2;

// Surface color at a point (x, y, z) on the unit sphere.
const SURFACES = {
  classic(x, y, z) {
    const n = fbm(x * 1.5 + 4, y * 1.5, z * 1.5, 4);
    return mix(C("#3d3d3a"), C("#8d8d88"), smooth(0.35, 0.65, n));
  },
  earth(x, y, z, lat) {
    const n = fbm(x * 1.7 + 11, y * 1.7 + 3, z * 1.7 - 7, 5);
    let c;
    if (n > 0.55) {
      const dry = fbm(x * 3 + 2, y * 3, z * 3 + 5, 3);
      c = Math.abs(lat) < 0.55 && dry > 0.56 ? C("#c4a46a") : mix(C("#3e6b2f"), C("#7b7148"), smooth(0.55, 0.7, n));
    } else {
      c = mix(C("#163f7a"), C("#2f72b8"), smooth(0.35, 0.55, n));
    }
    if (Math.abs(lat) > 1.2 + (n - 0.5) * 0.4) c = C("#eef2f6");
    const cloud = fbm(x * 2.6 - 4, y * 2.6 + 9, z * 2.6, 4);
    return mix(c, C("#ffffff"), smooth(0.58, 0.72, cloud) * 0.85);
  },
  moon(x, y, z) {
    let c = mix(C("#5f5d59"), C("#a8a6a0"), smooth(0.38, 0.56, fbm(x * 1.4 + 5, y * 1.4, z * 1.4 + 2, 4)));
    if (noise(x * 18, y * 18, z * 18) > 0.82) c = mix(c, C("#d4d2cc"), 0.6);
    return c;
  },
  mercury(x, y, z) {
    let c = mix(C("#6e655d"), C("#a2968a"), smooth(0.35, 0.65, fbm(x * 2 + 1, y * 2 + 7, z * 2, 4)));
    if (noise(x * 16 + 3, y * 16, z * 16) > 0.8) c = mix(c, C("#cfc4b8"), 0.55);
    return c;
  },
  venus(x, y, z, lat) {
    return mix(C("#cfa463"), C("#f0dcaa"), 0.5 + 0.5 * Math.sin(lat * 5 + fbm(x * 2, y * 2, z * 2, 4) * 6));
  },
  mars(x, y, z, lat) {
    const n = fbm(x * 1.6 - 3, y * 1.6 + 4, z * 1.6, 5);
    let c = mix(C("#6d3520"), C("#c0663a"), smooth(0.36, 0.55, n));
    c = mix(c, C("#d99a64"), smooth(0.6, 0.72, n));
    if (Math.abs(lat) > 1.28 + (n - 0.5) * 0.3) c = C("#f0ebe3");
    return c;
  },
  jupiter(x, y, z, lat, lon) {
    const band = Math.sin(lat * 13 + fbm(x * 3, y * 3, z * 3, 4) * 2.2);
    let c = mix(C("#a8744a"), C("#efe3c6"), 0.5 + 0.5 * band);
    if (Math.abs(lat) < 0.25) c = mix(c, C("#e9d9b5"), 0.4);
    const spot = oval(lon, lat, 1.0, -0.39, 0.34, 0.13); // the Great Red Spot
    return spot < 1 ? mix(C("#b9533a"), c, smooth(0.4, 1, spot)) : c;
  },
  saturn(x, y, z, lat) {
    return mix(C("#c9a96b"), C("#ecdcb0"), 0.5 + 0.5 * Math.sin(lat * 11 + fbm(x * 2.5, y * 2.5, z * 2.5, 3) * 1.2));
  },
  uranus(x, y, z, lat) {
    return mix(C("#93cfd6"), C("#b9e4e8"), 0.5 + 0.5 * Math.sin(lat * 6));
  },
  neptune(x, y, z, lat, lon) {
    let c = mix(C("#2c4aa6"), C("#4d73d6"), 0.5 + 0.5 * Math.sin(lat * 8 + fbm(x * 2, y * 2, z * 2, 3) * 1.5));
    if (oval(lon, lat, 2.0, -0.36, 0.3, 0.12) < 1) c = C("#1d2f6e"); // the Great Dark Spot
    if (Math.abs(lat + 0.5) < 0.12 && fbm(x * 6, y * 20, z * 6, 2) > 0.62) c = mix(c, C("#eef2ff"), 0.8);
    return c;
  },
  pluto(x, y, z, lat, lon) {
    let c = mix(C("#8d6b52"), C("#cdb89b"), smooth(0.38, 0.6, fbm(x * 1.8 + 2, y * 1.8, z * 1.8 - 1, 4)));
    const heart = oval(lon, lat, 0.9, 0.25, 0.55, 0.5); // Tombaugh Regio, Pluto's bright "heart"
    if (heart < 1) c = mix(C("#f2ebdd"), c, smooth(0.6, 1, heart));
    if (Math.abs(lat) < 0.35 && oval(lon, 0, -0.7, 0, 0.75, 1) < 1) c = mix(c, C("#5e3b2c"), 0.65); // dark Cthulhu
    return c;
  },
  titan(x, y, z, lat) {
    return mix(C("#c8873a"), C("#e3ad5f"), 0.5 + 0.5 * Math.sin(lat * 4 + fbm(x * 2, y * 2, z * 2, 3) * 2));
  },
  europa(x, y, z) {
    let c = mix(C("#cfc3b0"), C("#efe9df"), smooth(0.4, 0.6, fbm(x * 3 + 7, y * 3, z * 3 - 2, 4)));
    if (Math.abs(fbm(x * 4.5, y * 4.5 + 3, z * 4.5, 3) - 0.5) < 0.014) c = C("#9a6744"); // cracks in the ice
    return c;
  },
  io(x, y, z) {
    let c = mix(C("#d9c45a"), C("#efe39a"), smooth(0.4, 0.6, fbm(x * 3, y * 3, z * 3, 4)));
    if (noise(x * 9 + 2, y * 9, z * 9) > 0.8) c = C("#a8532a");
    return c;
  },
  phobos(x, y, z) {
    return mix(C("#5a4e45"), C("#8a7c70"), fbm(x * 4, y * 4, z * 4, 3));
  },
  charon(x, y, z, lat) {
    return lat > 1.1 ? C("#6b3f32") : mix(C("#7d7a76"), C("#aaa6a0"), fbm(x * 3, y * 3, z * 3, 3));
  },
};

const TILT = { earth: 23.4, mars: 25.2, saturn: 26.7, uranus: 97.8, neptune: 28.3, pluto: 119.5, moon: 6.7, jupiter: 3.1 };
const RIM = { earth: "#8fc1ff", venus: "#f2d39a", titan: "#e9a95a", mars: "#e8a57c", uranus: "#c4f2f6", neptune: "#7f9fff" };

// The flight deck's look for each world.
export const LOOKS = {
  classic: { sky: ["#d8d8d4", "#d8d8d4"], ground: "#2f2f2c", line: "#0b0b0a", dark: false, fog: [34, 78] },
  mercury: { sky: ["#000000", "#16130f"], ground: "#a0948a", line: "#e6dcd0", dark: true, stars: true, fog: [40, 90],
    sun: { size: 9, color: "#fff4d6", at: [24, 20, -80] } },
  venus: { sky: ["#a87a3e", "#e8c88a"], ground: "#6e4f2a", line: "#3a2914", dark: false, fog: [14, 46],
    weather: { color: "#f3d49a", size: 0.35, alpha: 0.16, count: 500 } },
  earth: { sky: ["#3a78c9", "#c4dcf2"], ground: "#4b5b38", line: "#1f2a14", dark: false, fog: [30, 85],
    sun: { size: 3, color: "#fffdf4", at: [30, 30, -90] }, clouds: true,
    bodies: [{ world: "moon", size: 1.6, at: [-34, 26, -95], night: 0.6 }] },
  moon: { sky: ["#000000", "#0b0b0f"], ground: "#b4b2ac", line: "#f0eee8", dark: true, stars: true, fog: [40, 95],
    sun: { size: 2.4, color: "#fffaf0", at: [32, 24, -95] },
    bodies: [{ world: "earth", size: 4.5, at: [-24, 19, -72], night: 0.15 }] },
  mars: { sky: ["#b9805a", "#e9c49c"], ground: "#8a4426", line: "#4a1f0e", dark: false, fog: [26, 80],
    sun: { size: 1.6, color: "#fff2dc", at: [28, 22, -90] }, weather: { color: "#d99a66", size: 0.09, alpha: 0.4, count: 600 },
    bodies: [{ world: "phobos", size: 0.9, at: [-22, 22, -80], night: 0.3 }] },
  jupiter: { sky: ["#3a2618", "#c8935e"], ground: "#e3c69b", line: "#7a5634", dark: true, cloudFloor: true, fog: [26, 80],
    sun: { size: 1, color: "#fff4dc", at: [30, 26, -95] }, weather: { color: "#f4e6cf", size: 0.32, alpha: 0.2, count: 450 },
    bodies: [{ world: "io", size: 1.4, at: [-26, 21, -82], night: 0.25 }] },
  europa: { sky: ["#000000", "#08080e"], ground: "#ddd6ca", line: "#f6f2ea", dark: true, stars: true, fog: [40, 95],
    sun: { size: 0.9, color: "#fff4dc", at: [34, 26, -95] },
    bodies: [{ world: "jupiter", size: 11, at: [-30, 16, -88], night: 0.08 }] },
  saturn: { sky: ["#3a3122", "#d3b97f"], ground: "#e9d6a7", line: "#7a6740", dark: true, cloudFloor: true, fog: [26, 80],
    sun: { size: 0.8, color: "#fff4dc", at: [30, 26, -95] }, weather: { color: "#f2e5c4", size: 0.32, alpha: 0.18, count: 450 },
    skyRing: true },
  titan: { sky: ["#7d4e18", "#d89a46"], ground: "#6d4a24", line: "#2e1d0b", dark: false, fog: [12, 42],
    weather: { color: "#e6a95a", size: 0.3, alpha: 0.14, count: 500 },
    bodies: [{ world: "saturn", size: 6, at: [-28, 20, -80], night: 0.2, rings: true, opacity: 0.35 }] },
  uranus: { sky: ["#3c7d8c", "#b3e3e8"], ground: "#cdeef0", line: "#3c6f78", dark: false, cloudFloor: true, fog: [26, 80],
    weather: { color: "#e6fbfc", size: 0.3, alpha: 0.18, count: 400 } },
  neptune: { sky: ["#081a52", "#3b62c6"], ground: "#a3bcf0", line: "#d7e3ff", dark: true, cloudFloor: true, fog: [26, 80],
    weather: { color: "#dbe6ff", size: 0.3, alpha: 0.2, count: 500 } },
  pluto: { sky: ["#000000", "#0c0b0e"], ground: "#cbb69a", line: "#f2e8d8", dark: true, stars: true, fog: [40, 95],
    sun: { size: 0.5, color: "#fff8ea", at: [34, 26, -95] },
    bodies: [{ world: "charon", size: 5.5, at: [-24, 18, -76], night: 0.12 }] },
};

// Dotted globes ------------------------------------------------------------------

const GLOBE_VERTEX = /* glsl */ `
  attribute vec3 aColor;
  uniform vec3 uLight;
  uniform float uSize;
  uniform float uScale;
  uniform float uNight;
  uniform float uMode; // 0 = planet surface, 1 = atmosphere rim, 2 = rings
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    if (uMode > 1.5) {
      // Ring dots are all visible, except where the planet sits in front of them.
      vec4 center = modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0);
      float radius = length((modelViewMatrix * vec4(1.0, 0.0, 0.0, 0.0)).xyz);
      bool hidden = mv.z < center.z && length(mv.xy - center.xy) < radius;
      vColor = aColor;
      vAlpha = hidden ? 0.0 : 0.85;
    } else {
      vec3 n = normalize(mat3(modelMatrix) * position);
      float light = max(dot(n, normalize(uLight)), 0.0);
      vColor = aColor * (uNight + (1.0 - uNight) * light);
      float facing = normalize(normalMatrix * position).z;
      // A planet hides its far side; the atmosphere rim only shows near the edge.
      vAlpha = uMode > 0.5 ? pow(1.0 - abs(facing), 3.0) * step(0.0, facing) * (0.3 + 0.7 * light)
                           : smoothstep(-0.05, 0.2, facing);
    }
    gl_PointSize = uSize * uScale / -mv.z;
    gl_Position = projectionMatrix * mv;
  }`;
const GLOBE_FRAGMENT = /* glsl */ `
  uniform float uOpacity;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    gl_FragColor = vec4(vColor, vAlpha * uOpacity * smoothstep(0.5, 0.32, d));
    #include <colorspace_fragment>
  }`;

function globeMaterial({ size, night, mode = 0, opacity = 1, light }) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uLight: { value: light }, uSize: { value: size }, uScale: { value: 500 },
      uNight: { value: night }, uMode: { value: mode }, uOpacity: { value: opacity },
    },
    vertexShader: GLOBE_VERTEX,
    fragmentShader: GLOBE_FRAGMENT,
    transparent: true,
    depthWrite: false,
  });
}

function fibonacciSphere(count, radius, colorAt) {
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const color = new THREE.Color();
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = 1 - (2 * (i + 0.5)) / count;
    const r = Math.sqrt(1 - y * y);
    const x = Math.cos(golden * i) * r, z = Math.sin(golden * i) * r;
    positions.set([x * radius, y * radius, z * radius], i * 3);
    const [cr, cg, cb] = colorAt(x, y, z, Math.asin(y), Math.atan2(z, x));
    color.setRGB(cr, cg, cb, THREE.SRGBColorSpace);
    colors.set([color.r, color.g, color.b], i * 3);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  return geometry;
}

function saturnRings(count) {
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const color = new THREE.Color();
  let k = 0;
  while (k < count) {
    const r = 1.24 + Math.random() * 1.03;
    if (r > 1.95 && r < 2.03) continue; // the Cassini division
    const density = r < 1.53 ? 0.35 : r < 1.95 ? 1 : 0.7; // faint C ring, bright B ring, A ring
    if (Math.random() > density) continue;
    const a = Math.random() * Math.PI * 2;
    positions.set([Math.cos(a) * r, (Math.random() - 0.5) * 0.01, Math.sin(a) * r], k * 3);
    const [cr, cg, cb] = C(r < 1.53 ? "#a99a7c" : r < 1.95 ? "#e0cfa6" : "#cbb991");
    color.setRGB(cr, cg, cb, THREE.SRGBColorSpace);
    colors.set([color.r, color.g, color.b], k * 3);
    k++;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  return geometry;
}

const geometryCache = new Map();
function cached(key, make) {
  if (!geometryCache.has(key)) geometryCache.set(key, make());
  return geometryCache.get(key);
}

/**
 * A planet as a sphere of colored dots. Returns { group, spin, materials }:
 * rotate `spin` to turn the planet on its (tilted) axis.
 */
export function dotGlobe(world, { points = 16000, size = 0.03, night = 0.18, light = new THREE.Vector3(1, 0.4, 0.8), rings, opacity = 1 } = {}) {
  const surface = SURFACES[world] || SURFACES.classic;
  const group = new THREE.Group();
  const spin = new THREE.Group();
  group.add(spin);
  group.rotation.z = THREE.MathUtils.degToRad(TILT[world] || 0);
  const materials = [];
  const body = new THREE.Points(cached(`${world}:${points}`, () => fibonacciSphere(points, 1, surface)),
    globeMaterial({ size, night, light, opacity }));
  spin.add(body);
  materials.push(body.material);
  if (RIM[world]) {
    const tint = RIM[world];
    const rim = new THREE.Points(cached(`rim:${world}`, () => fibonacciSphere(4000, 1.045, () => C(tint))),
      globeMaterial({ size: size * 1.1, night: 0.4, light, mode: 1, opacity }));
    group.add(rim);
    materials.push(rim.material);
  }
  if (rings ?? world === "saturn") {
    const ring = new THREE.Points(cached(`rings:${points}`, () => saturnRings(Math.round(points * 0.7))),
      globeMaterial({ size: size * 0.8, night: 1, light, mode: 2, opacity }));
    group.add(ring);
    materials.push(ring.material);
  }
  group.traverse((o) => { o.frustumCulled = false; });
  return { group, spin, materials };
}

/** Keep dot sizes right when the canvas size changes. */
export function setGlobeScale(materials, renderer, camera) {
  const scale = (renderer.domElement.height) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
  for (const m of materials) m.uniforms.uScale.value = scale;
}

// The hero globe ----------------------------------------------------------------------

export function createGlobe(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  camera.position.set(0, 0.6, 7.6);
  const controls = new OrbitControls(camera, canvas);
  controls.enableZoom = false;
  controls.enablePan = false;
  controls.enableDamping = true;
  controls.rotateSpeed = 0.6;
  const light = new THREE.Vector3(-1, 0.5, 0.9);
  let current = null, next = null, fade = 1, world = null;

  function build(id) {
    const globe = dotGlobe(id, { points: 22000, size: 0.019, night: 0.3, light });
    scene.add(globe.group);
    return globe;
  }

  function setWorld(id) {
    if (id === world) return;
    world = id;
    if (next) { scene.remove(next.group); }
    next = build(id);
    next.materials.forEach((m) => { m.uniforms.uOpacity.value = 0; });
    fade = 0;
    resize();
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    for (const g of [current, next]) if (g) setGlobeScale(g.materials, renderer, camera);
  }
  new ResizeObserver(resize).observe(canvas);

  let last = performance.now();
  return {
    setWorld,
    render() {
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      if (next) {
        fade = Math.min(fade + dt * 2.2, 1);
        next.materials.forEach((m) => { m.uniforms.uOpacity.value = fade; });
        if (current) current.materials.forEach((m) => { m.uniforms.uOpacity.value = 1 - fade; });
        if (fade >= 1) {
          if (current) scene.remove(current.group);
          current = next;
          next = null;
        }
      }
      for (const g of [current, next]) if (g) g.spin.rotation.y += dt * 0.12;
      controls.update();
      renderer.render(scene, camera);
    },
  };
}
