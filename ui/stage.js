// The flight deck's 3D scene: a dotted landscape where flights are replayed, dressed as
// whichever world the flight happened on. World units match the simulator, with the pad
// center at (0, 0) and z pointing at you.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { F } from "./nn.js";
import { dotGlobe, LOOKS, setGlobeScale } from "./worlds.js";

const S = 1 / 30; // simulator pixels per world unit
const LANDER_POLY = [[-14, 17], [-17, 0], [-17, -10], [17, -10], [17, 0], [14, 17]];
const LEG_HALF = { w: 2 * S, h: 8 * S };
const TRAIL_COLORS = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#898781", "#898781"],
  dark: ["#3987e5", "#d95926", "#199e70", "#8a8984", "#8a8984"],
};

const POINT_VERTEX = /* glsl */ `
  attribute vec3 aColor;
  attribute float aAlpha;
  attribute float aSize;
  uniform float uScale;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vColor = aColor;
    vAlpha = aAlpha;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * uScale / -mv.z;
    gl_Position = projectionMatrix * mv;
  }`;
const POINT_FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    gl_FragColor = vec4(vColor, vAlpha * smoothstep(0.5, 0.36, d));
    #include <colorspace_fragment>
  }`;
const SKY_VERTEX = /* glsl */ `
  varying vec3 vWorld;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }`;
const SKY_FRAGMENT = /* glsl */ `
  uniform vec3 uTop;
  uniform vec3 uHorizon;
  varying vec3 vWorld;
  void main() {
    float h = normalize(vWorld - cameraPosition).y;
    gl_FragColor = vec4(mix(uHorizon, uTop, pow(clamp(h, 0.0, 1.0), 0.55)), 1.0);
    #include <colorspace_fragment>
  }`;

function pointCloud(capacity) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
  geometry.setAttribute("aAlpha", new THREE.BufferAttribute(new Float32Array(capacity), 1));
  geometry.setAttribute("aSize", new THREE.BufferAttribute(new Float32Array(capacity), 1));
  geometry.setDrawRange(0, 0);
  const points = new THREE.Points(geometry, new THREE.ShaderMaterial({
    uniforms: { uScale: { value: 400 } },
    vertexShader: POINT_VERTEX,
    fragmentShader: POINT_FRAGMENT,
    transparent: true,
    depthWrite: false,
  }));
  points.frustumCulled = false;
  return points;
}

function touch(points, ...names) {
  for (const name of names) points.geometry.attributes[name].needsUpdate = true;
}

function dotTexture(soft = false) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const gradient = g.createRadialGradient(32, 32, 0, 32, 32, 30);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(soft ? 0.2 : 0.9, "rgba(255,255,255,1)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = gradient;
  g.fillRect(0, 0, 64, 64);
  const texture = new THREE.CanvasTexture(c);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

const smooth = (a, b, v) => {
  const t = Math.min(Math.max((v - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};

// Rolling hills behind the physics slice, and beyond the simulator's 20-unit world.
function backdrop(x, z) {
  return (
    0.55 * Math.sin(0.23 * x + 0.17 * z + 0.6) +
    0.45 * Math.sin(0.11 * x - 0.31 * z + 1.7) +
    0.28 * Math.sin(0.57 * x + 0.43 * z + 0.3) +
    Math.max(0, -z - 9) * 0.075 * (1.3 + Math.sin(0.15 * x + 0.21 * z))
  );
}

function profileAt(profile, x) {
  const first = profile[0], last = profile[profile.length - 1];
  if (x <= first[0]) return first[1] + (backdrop(x, 0) - backdrop(first[0], 0)) * smooth(0, 4, first[0] - x);
  if (x >= last[0]) return last[1] + (backdrop(x, 0) - backdrop(last[0], 0)) * smooth(0, 4, x - last[0]);
  for (let i = 1; i < profile.length; i++) {
    if (x <= profile[i][0]) {
      const [x0, y0] = profile[i - 1], [x1, y1] = profile[i];
      return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
    }
  }
  return last[1];
}

function buildLander() {
  const group = new THREE.Group();
  const shape = new THREE.Shape(LANDER_POLY.map(([x, y]) => new THREE.Vector2(x * S, y * S)));
  const bodyGeometry = new THREE.ExtrudeGeometry(shape, {
    depth: 0.82, bevelEnabled: true, bevelSize: 0.035, bevelThickness: 0.04, bevelSegments: 3,
  });
  bodyGeometry.translate(0, 0, -0.41);
  group.add(new THREE.Mesh(bodyGeometry, new THREE.MeshStandardMaterial({ color: 0x1f1f1d, roughness: 0.45, metalness: 0.35 })));
  const porthole = new THREE.Mesh(new THREE.CircleGeometry(0.13, 32), new THREE.MeshBasicMaterial({ color: 0xe9e9e4 }));
  porthole.position.set(0, 0.2, 0.46);
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.13, 0.17, 32), new THREE.MeshBasicMaterial({ color: 0x5c5c57 }));
  ring.position.set(0, 0.2, 0.455);
  const stripe = new THREE.Mesh(new THREE.BoxGeometry(1.02, 0.045, 0.02), new THREE.MeshBasicMaterial({ color: 0xeb6834 }));
  stripe.position.set(0, -0.12, 0.455);
  group.add(porthole, ring, stripe);

  const metal = new THREE.MeshStandardMaterial({ color: 0x4a4a46, roughness: 0.35, metalness: 0.6 });
  const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.19, 0.2, 24, 1, true), metal);
  nozzle.position.set(0, -0.43, 0);
  group.add(nozzle);
  for (const side of [-1, 1]) {
    const pod = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.075, 0.14, 16), metal);
    pod.rotation.z = (side * Math.PI) / 2;
    pod.position.set(side * 0.6, 0.47, 0);
    group.add(pod);
  }

  const cone = (radius, length, color, opacity) => {
    const geometry = new THREE.ConeGeometry(radius, length, 24, 1, true);
    geometry.translate(0, -length / 2, 0);
    geometry.rotateX(Math.PI); // open end at the nozzle, tip pointing away
    geometry.translate(0, -length, 0);
    return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
  };
  const main = new THREE.Group();
  main.add(cone(0.17, 1.1, 0xeb6834, 0.78), cone(0.09, 0.6, 0xffd27a, 0.95));
  main.position.set(0, -0.53, 0);
  group.add(main);
  // The left engine (pushes the rocket left) vents on the right side, and vice versa.
  const sides = { left: null, right: null };
  for (const [name, x, dir] of [["left", 0.67, 1], ["right", -0.67, -1]]) {
    const flame = new THREE.Group();
    flame.add(cone(0.075, 0.5, 0xeb6834, 0.8), cone(0.04, 0.28, 0xffd27a, 0.95));
    flame.rotation.z = dir * (Math.PI / 2);
    flame.position.set(x, 0.47, 0);
    group.add(flame);
    sides[name] = flame;
  }
  for (const flame of [main, sides.left, sides.right]) flame.scale.setScalar(0.001);

  const legMaterial = new THREE.MeshStandardMaterial({ color: 0x33332f, roughness: 0.5, metalness: 0.4 });
  const footMaterials = [0, 1].map(() => new THREE.MeshStandardMaterial({ color: 0x33332f, roughness: 0.5 }));
  const legs = footMaterials.map((footMaterial) => {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(LEG_HALF.w * 2, LEG_HALF.h * 2, 0.14), legMaterial);
    const foot = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.05, 0.34), footMaterial);
    foot.position.y = -LEG_HALF.h;
    leg.add(foot);
    return leg;
  });
  return { group, main, sides, legs, footMaterials };
}

function buildPad(scene) {
  const pad = new THREE.Group();
  const slab = new THREE.Mesh(new THREE.BoxGeometry(4.0, 0.07, 2.6), new THREE.MeshStandardMaterial({ color: 0x141413, roughness: 0.7 }));
  slab.position.y = -0.035;
  const lights = new THREE.Points(
    new THREE.BufferGeometry().setFromPoints(
      Array.from({ length: 22 }, (_, i) => new THREE.Vector3(-1.9 + (i % 11) * 0.38, 0.01, i < 11 ? 1.2 : -1.2)),
    ),
    new THREE.PointsMaterial({ color: 0xeb6834, size: 0.09 }),
  );
  pad.add(slab, lights);
  const flags = [];
  for (const x of [-2, 2]) {
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 1.67, 8), new THREE.MeshStandardMaterial({ color: 0x2a2a27 }));
    pole.position.set(x, 0.835, 0);
    const shape = new THREE.Shape([new THREE.Vector2(0, 0), new THREE.Vector2(0.55, -0.16), new THREE.Vector2(0, -0.33)]);
    const flag = new THREE.Mesh(new THREE.ShapeGeometry(shape, 6), new THREE.MeshBasicMaterial({ color: 0xc8a02a, side: THREE.DoubleSide }));
    flag.position.set(x + 0.02, 1.67, 0);
    pad.add(pole, flag);
    flags.push(flag);
  }
  scene.add(pad);
  return { pad, flags };
}

export function createStage(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xd8d8d4, 34, 78);
  const camera = new THREE.PerspectiveCamera(34, 16 / 9, 0.1, 400);
  camera.position.set(0, 6.5, 26);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 4.2, 0);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.enableZoom = false; // the page scrolls with the wheel; distance lives in the view panel
  controls.maxPolarAngle = Math.PI * 0.53;
  controls.minAzimuthAngle = -1.2;
  controls.maxAzimuthAngle = 1.2;

  const hemi = new THREE.HemisphereLight(0xffffff, 0x8f8f89, 2.2);
  const sun = new THREE.DirectionalLight(0xffffff, 2.4);
  sun.position.set(-6, 12, 9);
  const rim = new THREE.DirectionalLight(0xffffff, 0);
  rim.position.set(4, 6, -12);
  scene.add(hemi, sun, rim);

  // Sky: a gradient dome, stars, a sun, and other worlds hanging in it.
  const sky = new THREE.Mesh(new THREE.SphereGeometry(300, 32, 16), new THREE.ShaderMaterial({
    uniforms: { uTop: { value: new THREE.Color() }, uHorizon: { value: new THREE.Color() } },
    vertexShader: SKY_VERTEX, fragmentShader: SKY_FRAGMENT, side: THREE.BackSide, depthWrite: false, fog: false,
  }));
  const stars = new THREE.Points(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 1800 }, () => {
    const a = Math.random() * Math.PI * 2, h = Math.random() * 0.95 + 0.02;
    const r = Math.sqrt(1 - h * h);
    return new THREE.Vector3(Math.cos(a) * r * 240, h * 240, Math.sin(a) * r * 240 - 40);
  })), new THREE.PointsMaterial({ color: 0xffffff, size: 1.5, sizeAttenuation: false, transparent: true, opacity: 0.8, fog: false }));
  const sunSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: dotTexture(true), blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  const skyThings = new THREE.Group();
  sky.renderOrder = -1;
  scene.add(sky, stars, sunSprite, skyThings);
  let bodies = [];

  // Terrain: a field of dots whose front row is the simulator's exact ground.
  const xs = [], zs = [];
  for (let z = -48; z <= 7; z += 0.34) zs.push(z);
  for (let x = -44; x <= 44; x += 0.26) xs.push(x);
  const count = xs.length * zs.length;
  const terrainGeometry = new THREE.BufferGeometry();
  const terrainPositions = new Float32Array(count * 3);
  terrainGeometry.setAttribute("position", new THREE.BufferAttribute(terrainPositions, 3));
  terrainGeometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  const dot = dotTexture(), softDot = dotTexture(true);
  const terrainMaterial = new THREE.PointsMaterial({ size: 0.085, map: dot, alphaTest: 0.5, transparent: true, vertexColors: true });
  const terrain = new THREE.Points(terrainGeometry, terrainMaterial);
  scene.add(terrain);
  const groundXs = Array.from({ length: 1760 }, (_, i) => -44 + i * 0.05);
  const groundPositions = new Float32Array(groundXs.length * 3);
  const groundGeometry = new THREE.BufferGeometry();
  groundGeometry.setAttribute("position", new THREE.BufferAttribute(groundPositions, 3));
  const groundLine = new THREE.Points(groundGeometry, new THREE.PointsMaterial({ color: 0x0b0b0a, size: 0.07, map: dot, alphaTest: 0.5, transparent: true }));
  scene.add(groundLine);
  let fromHeights = null, toHeights = null, morph = 1;

  function heightsFor(profile) {
    const heights = new Float32Array(count);
    let k = 0;
    for (const z of zs) {
      const w = 1 - smooth(0.7, 6, Math.abs(z));
      for (const x of xs) heights[k++] = profileAt(profile, x) * w + (backdrop(x, z) - 0.6) * (1 - w) - (z > 1.5 ? (z - 1.5) * 0.25 : 0);
    }
    return heights;
  }

  function applyTerrain(t) {
    const e = t * t * (3 - 2 * t);
    let k = 0, p = 0;
    for (const z of zs) {
      for (const x of xs) {
        terrainPositions[p++] = x;
        terrainPositions[p++] = fromHeights ? fromHeights[k] + (toHeights[k] - fromHeights[k]) * e : toHeights[k];
        terrainPositions[p++] = z;
        k++;
      }
    }
    terrainGeometry.attributes.position.needsUpdate = true;
  }

  const { pad, flags } = buildPad(scene);
  const lander = buildLander();
  scene.add(lander.group, ...lander.legs);

  const exhaust = pointCloud(900);
  const particles = Array.from({ length: 900 }, () => ({ life: 0, age: 1, p: new THREE.Vector3(), v: new THREE.Vector3() }));
  let nextParticle = 0;
  const hot = new THREE.Color("#ff8a3d"), smoke = new THREE.Color("#5e5e59"), tmp = new THREE.Color();
  const trails = pointCloud(16000);
  const weather = pointCloud(700);
  let weatherSpec = null;
  scene.add(exhaust, trails, weather);

  const state = {
    flight: null, look: null, world: null, cameraMode: "follow", distance: 26, showTrails: true,
    pose: null, velocity: new THREE.Vector3(), main: 0, side: 0, wind: 0, reframe: 0,
    top: 10, // the highest point of the current flight, for the wide camera
  };
  let trailList = [];

  /** Dress the scene as a world: sky, stars, sun, ground colors, light, weather. */
  function setWorld(world) {
    const look = LOOKS[world] || LOOKS.classic;
    if (state.world === world) return;
    state.world = world;
    state.look = look;
    const [top, horizon] = look.sky.map((c) => new THREE.Color(c));
    sky.material.uniforms.uTop.value.copy(top);
    sky.material.uniforms.uHorizon.value.copy(horizon);
    scene.fog.color.copy(horizon);
    [scene.fog.near, scene.fog.far] = look.fog;
    renderer.setClearColor(horizon);
    stars.visible = !!look.stars;

    const base = new THREE.Color(look.ground);
    const colors = terrainGeometry.attributes.color.array;
    const c = new THREE.Color();
    let k = 0;
    for (const z of zs) {
      for (const x of xs) {
        const n = 0.5 + 0.5 * Math.sin(x * 1.7 + Math.cos(z * 1.3) * 2) * Math.sin(z * 0.9 + x * 0.3);
        const bands = look.cloudFloor ? 0.12 * Math.sin(z * 0.9 + Math.sin(x * 0.2) * 2) : 0;
        c.copy(base).multiplyScalar(0.82 + 0.3 * n + bands);
        colors.set([c.r, c.g, c.b], k * 3);
        k++;
      }
    }
    terrainGeometry.attributes.color.needsUpdate = true;
    terrainMaterial.size = look.cloudFloor ? 0.17 : 0.085;
    terrainMaterial.map = look.cloudFloor ? softDot : dot;
    terrainMaterial.alphaTest = look.cloudFloor ? 0.05 : 0.5;
    terrainMaterial.opacity = look.cloudFloor ? 0.75 : 1;
    terrainMaterial.needsUpdate = true;
    groundLine.material.color.set(look.line);

    hemi.color.set(look.dark ? "#cfd6e6" : "#ffffff");
    hemi.groundColor.set(look.ground);
    hemi.intensity = look.dark ? 1.3 : 2.2;
    sun.intensity = look.dark ? 3.2 : 2.4;
    rim.intensity = look.dark ? 1.8 : 0; // outline the dark rocket against a dark sky

    if (look.sun) {
      sunSprite.visible = true;
      sunSprite.position.set(...look.sun.at);
      sunSprite.scale.setScalar(look.sun.size * 3.2);
      sunSprite.material.color.set(look.sun.color);
      sun.position.copy(sunSprite.position).normalize().multiplyScalar(20);
    } else {
      sunSprite.visible = false;
      sun.position.set(-6, 12, 9);
    }

    skyThings.clear();
    bodies = [];
    for (const b of look.bodies || []) {
      const at = new THREE.Vector3(...b.at);
      const light = look.sun ? new THREE.Vector3(...look.sun.at).sub(at).normalize() : new THREE.Vector3(1, 0.5, 0.6);
      // Dot size is in world units, so it scales with the globe's size.
      const globe = dotGlobe(b.world, { points: 9000, size: 0.032 * b.size, night: b.night ?? 0.15, light, rings: b.rings, opacity: b.opacity ?? 1 });
      globe.group.position.copy(at);
      globe.group.scale.setScalar(b.size);
      skyThings.add(globe.group);
      bodies.push(globe);
    }
    if (look.skyRing) skyThings.add(skyRing());
    if (look.clouds) skyThings.add(cloudBank());
    weatherSpec = look.weather || null;
    seedWeather();
    resize();
    if (trailList.length) setTrails(trailList);
  }

  function skyRing() { // Saturn's rings seen from inside its atmosphere: an arc across the sky
    const points = [];
    for (let i = 0; i < 5000; i++) {
      const t = Math.random() * Math.PI, band = 1 + (Math.random() - 0.5) * 0.08;
      points.push(new THREE.Vector3(Math.cos(t) * 170 * band, 12 + Math.sin(t) * 46 * band, -150));
    }
    return new THREE.Points(new THREE.BufferGeometry().setFromPoints(points),
      new THREE.PointsMaterial({ color: 0xf1e2bd, size: 1.4, sizeAttenuation: false, transparent: true, opacity: 0.7, fog: false }));
  }

  function cloudBank() { // soft clouds high in Earth's sky
    const points = [];
    for (let c = 0; c < 14; c++) {
      const cx = (Math.random() - 0.5) * 120, cy = 16 + Math.random() * 12, cz = -50 - Math.random() * 40;
      for (let i = 0; i < 60; i++) points.push(new THREE.Vector3(cx + (Math.random() - 0.5) * 14, cy + (Math.random() - 0.5) * 2.5, cz + (Math.random() - 0.5) * 6));
    }
    return new THREE.Points(new THREE.BufferGeometry().setFromPoints(points),
      new THREE.PointsMaterial({ color: 0xffffff, size: 2.2, map: softDot, transparent: true, opacity: 0.55, depthWrite: false, fog: false }));
  }

  function seedWeather() {
    const n = weatherSpec?.count || 0;
    const pos = weather.geometry.attributes.position.array;
    const col = weather.geometry.attributes.aColor.array;
    const alpha = weather.geometry.attributes.aAlpha.array;
    const size = weather.geometry.attributes.aSize.array;
    const color = new THREE.Color(weatherSpec?.color || "#ffffff");
    for (let i = 0; i < n; i++) {
      pos.set([(Math.random() - 0.5) * 50, Math.random() * 14 - 1, -18 + Math.random() * 24], i * 3);
      col.set([color.r, color.g, color.b], i * 3);
      alpha[i] = weatherSpec.alpha * (0.4 + Math.random() * 0.6);
      size[i] = weatherSpec.size * (0.6 + Math.random() * 0.8);
    }
    weather.geometry.setDrawRange(0, n);
    touch(weather, "position", "aColor", "aAlpha", "aSize");
  }

  function setFlight(flight) {
    state.flight = flight;
    state.wind = flight.conditions?.enable_wind ? flight.conditions.wind_power : 0;
    setWorld(flight.conditions?.planet in LOOKS ? flight.conditions.planet : state.world || "classic");
    const profile = flight.terrain;
    fromHeights = toHeights ? terrainHeightsNow() : null;
    toHeights = heightsFor(profile);
    morph = fromHeights ? 0 : 1;
    applyTerrain(morph);
    for (let i = 0; i < groundXs.length; i++) {
      groundPositions.set([groundXs[i], profileAt(profile, groundXs[i]) + 0.012, 0], i * 3);
    }
    groundGeometry.attributes.position.needsUpdate = true;
    pad.position.y = profile[Math.floor(profile.length / 2)][1];
    state.top = flight.frames.reduce((top, f) => Math.max(top, f[F.y]), 0);
    state.reframe = 1; // swing the camera to the new flight quickly
  }

  function terrainHeightsNow() {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) out[i] = terrainPositions[i * 3 + 1];
    return out;
  }

  /** Place the lander between two recorded frames (t in 0..1). */
  function setPose(a, b, t, final) {
    const lerp = (i) => a[i] + (b[i] - a[i]) * t;
    lander.group.position.set(lerp(F.x), lerp(F.y), 0);
    lander.group.rotation.z = lerp(F.angle);
    lander.legs.forEach((leg, i) => {
      const o = i === 0 ? F.leg1 : F.leg2;
      leg.position.set(lerp(o), lerp(o + 1), 0);
      leg.rotation.z = lerp(o + 2);
      const touching = a[F.obs + 6 + i] > 0.5;
      lander.footMaterials[i].color.set(touching ? 0x3987e5 : 0x33332f);
    });
    state.velocity.set((b[F.x] - a[F.x]) * 50, (b[F.y] - a[F.y]) * 50, 0);
    state.main = final ? 0 : a[F.main];
    state.side = final ? 0 : a[F.side];
    state.pose = a;
  }

  function emit(dt) {
    const group = lander.group;
    group.updateMatrixWorld();
    const spawn = (local, dir, speed, count) => {
      for (let n = 0; n < count; n++) {
        const q = particles[nextParticle];
        nextParticle = (nextParticle + 1) % particles.length;
        q.p.copy(local).applyMatrix4(group.matrixWorld);
        q.p.z += (Math.random() - 0.5) * 0.25;
        q.v.copy(dir).applyQuaternion(group.quaternion).multiplyScalar(speed * (0.7 + Math.random() * 0.6));
        q.v.x += (Math.random() - 0.5) * 1.2;
        q.v.z += (Math.random() - 0.5) * 1.2;
        q.v.add(state.velocity);
        q.age = 0;
        q.life = 0.5 + Math.random() * 0.6;
      }
    };
    if (state.main > 0) spawn(new THREE.Vector3(0, -0.6, 0), new THREE.Vector3(0, -1, 0), 7 * state.main, Math.round(6 * state.main));
    if (state.side < 0) spawn(new THREE.Vector3(0.75, 0.47, 0), new THREE.Vector3(1, 0, 0), 4, Math.round(3 * -state.side));
    if (state.side > 0) spawn(new THREE.Vector3(-0.75, 0.47, 0), new THREE.Vector3(-1, 0, 0), 4, Math.round(3 * state.side));

    const pos = exhaust.geometry.attributes.position.array;
    const col = exhaust.geometry.attributes.aColor.array;
    const alpha = exhaust.geometry.attributes.aAlpha.array;
    const size = exhaust.geometry.attributes.aSize.array;
    particles.forEach((q, i) => {
      if (q.age < q.life) {
        q.age += dt;
        q.v.y -= 1.5 * dt;
        q.v.x += state.wind * 0.08 * dt;
        q.v.multiplyScalar(1 - 1.8 * dt);
        q.p.addScaledVector(q.v, dt);
      }
      const k = Math.min(q.age / q.life, 1);
      tmp.copy(hot).lerp(smoke, Math.min(k * 1.6, 1));
      pos.set([q.p.x, q.p.y, q.p.z], i * 3);
      col.set([tmp.r, tmp.g, tmp.b], i * 3);
      alpha[i] = k >= 1 ? 0 : (1 - k) * 0.9;
      size[i] = 0.07 + k * 0.16;
    });
    touch(exhaust, "position", "aColor", "aAlpha", "aSize");
    exhaust.geometry.setDrawRange(0, particles.length);
  }

  function driftWeather(dt, time) {
    if (!weatherSpec) return;
    const pos = weather.geometry.attributes.position.array;
    const speed = 0.4 + state.wind * 0.35;
    for (let i = 0; i < weatherSpec.count; i++) {
      let x = pos[i * 3] + speed * dt * (0.6 + (i % 7) * 0.08);
      if (x > 25) x -= 50;
      pos[i * 3] = x;
      pos[i * 3 + 1] += Math.sin(time * 0.7 + i) * 0.004;
    }
    touch(weather, "position");
  }

  /** Practice attempts as fading dotted paths: [[attempt, outcome code, [[x, y], ...]], ...]. */
  function setTrails(list) {
    trailList = list;
    const recent = list.slice(-110);
    const palette = TRAIL_COLORS[state.look?.dark ? "dark" : "light"];
    const pos = trails.geometry.attributes.position.array;
    const col = trails.geometry.attributes.aColor.array;
    const alpha = trails.geometry.attributes.aAlpha.array;
    const size = trails.geometry.attributes.aSize.array;
    const color = new THREE.Color();
    let k = 0;
    recent.forEach(([n, code, path], rank) => {
      const fade = 0.12 + 0.75 * ((rank + 1) / recent.length);
      const z = (((n * 7919) % 97) / 97 - 0.5) * 2.2;
      color.set(palette[code] || palette[3]);
      path.forEach(([x, y], j) => {
        if (k >= 16000) return;
        pos.set([x, y, z], k * 3);
        col.set([color.r, color.g, color.b], k * 3);
        const end = j === path.length - 1;
        alpha[k] = end ? Math.min(fade + 0.15, 1) : fade * 0.8;
        size[k] = end ? 0.2 : 0.075;
        k++;
      });
    });
    trails.geometry.setDrawRange(0, k);
    touch(trails, "position", "aColor", "aAlpha", "aSize");
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    const scale = (h * renderer.getPixelRatio()) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
    for (const points of [exhaust, trails, weather]) points.material.uniforms.uScale.value = scale;
    for (const b of bodies) setGlobeScale(b.materials, renderer, camera);
  }
  new ResizeObserver(resize).observe(canvas);
  setWorld("classic");
  toHeights = heightsFor([[-10, 1.2], [-8, 0.9], [-6, 0.4], [-4, 0.3], [-2, -0.03], [0, -0.03], [2, -0.03], [4, 0.5], [6, 0.8], [8, 1.1], [10, 1.4]]);
  applyTerrain(1);

  const desired = new THREE.Vector3();
  let time = 0;
  function render(dt) {
    time += dt;
    if (morph < 1) {
      morph = Math.min(morph + dt / 0.9, 1);
      applyTerrain(morph);
    }
    const flame = (power, group) => {
      const target = power > 0 ? (0.55 + 0.45 * power) * (0.88 + Math.sin(time * 61) * 0.1 + Math.random() * 0.06) : 0.001;
      const s = group.scale.x + (target - group.scale.x) * Math.min(dt * 30, 1);
      group.scale.setScalar(Math.max(s, 0.001));
      group.visible = s > 0.02;
    };
    flame(state.main, lander.main);
    flame(Math.max(-state.side, 0), lander.sides.left);
    flame(Math.max(state.side, 0), lander.sides.right);
    emit(dt);
    driftWeather(dt, time);
    trails.visible = state.showTrails;
    flags.forEach((f, i) => { f.rotation.y = Math.sin(time * 2.2 + i) * (0.15 + state.wind * 0.03); });
    for (const b of bodies) b.spin.rotation.y += dt * 0.02;

    if (state.cameraMode !== "free") {
      // Frame the rocket (follow) or the whole flight (wide) with the ground below it, keeping
      // the rocket out of the top fifth of the view, where the caption sits. Flights that start
      // high need the camera to back away: follow backs up to 2.6 times the chosen distance and
      // then keeps just the rocket in view; wide backs up to 3.2 times.
      const p = lander.group.position;
      const follow = state.cameraMode === "follow" && state.pose;
      const y = follow ? p.y : state.top, room = 0.4, ground = -1.5; // room: share of the upper half kept clear
      const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
      const fit = (y - ground) / ((2 - room) * tan);
      const distance = Math.min(Math.max(state.distance, fit), state.distance * (follow ? 2.6 : 3.2));
      const half = distance * tan, lo = y - (1 - room) * half, hi = ground + half;
      const prefer = follow ? Math.max(1.8, p.y * 0.62 + 0.5) : 4.2;
      desired.set(follow ? p.x * 0.55 : 0, lo <= hi ? Math.min(Math.max(prefer, lo), hi) : lo, 0);
      state.reframe = Math.max(0, state.reframe - dt * 1.5);
      const delta = desired.sub(controls.target).multiplyScalar(Math.min(dt * (2.2 + state.reframe * 7), 1));
      controls.target.add(delta);
      camera.position.add(delta);
      const offset = camera.position.clone().sub(controls.target);
      offset.setLength(offset.length() + (distance - offset.length()) * Math.min(dt * 3, 1));
      camera.position.copy(controls.target).add(offset);
    }
    // Push the fog back as far as the camera has backed away, so the rocket never fades out.
    const away = Math.max(0, camera.position.distanceTo(controls.target) - state.distance);
    scene.fog.near = state.look.fog[0] + away;
    scene.fog.far = state.look.fog[1] + away;
    controls.update();
    sky.position.copy(camera.position);
    renderer.render(scene, camera);
  }

  return {
    setWorld,
    setFlight,
    setPose,
    setTrails,
    render,
    options: state,
    resetView() {
      controls.target.set(0, 4.2, 0);
      camera.position.set(0, 6.5, state.distance);
    },
  };
}
