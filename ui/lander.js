// The 3D lander for the flight deck and the game: the same machine as the hand-drawn figure
// (ui/figures/src/lander.html), in flat toon shading with ink outlines so it reads like the drawing.
//
// It is built in the simulator's body frame (meters, origin at the Box2D body origin) so it fits the
// physics: the hull fills the collision polygon (1.13 m wide, from 0.33 m below the origin to 0.57 m
// above), the engine bell ends 0.47 m below the origin (the pad is 0.55 m below at rest), the thruster
// pods sit where the side engines push (0.4 m out, 0.47 m up), and the two physics legs drive the left
// and right legs every frame. The front and back legs copy their stroke, so all four touch down together.

import * as THREE from "three";
import { mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";

/** Where the main engine's exhaust leaves the bell, in the hull's frame. */
export const NOZZLE_EXIT = new THREE.Vector3(0, -0.47, 0);
/** The side thrusters' outlets: [0] on the +x side (fires when side < 0), [1] on the -x side. */
export const RCS_PORTS = [new THREE.Vector3(0.5, 0.47, 0), new THREE.Vector3(-0.5, 0.47, 0)];

const INK = 0x231e16;
const FOOT = 8 / 30; // from a physics leg's centre to its foot
const DS = { r: 0.6, y0: -0.33, y1: 0.07 }; // descent stage: an octagon in gold foil
const AV = { r0: 0.5, r1: 0.41, y0: 0.07, y1: 0.47 }; // avionics deck on top
const OUTLINE = 0.016; // ink width, in meters

// Three flat tones, like the figure's light / mid / dark bands.
const tones = new THREE.DataTexture(new Uint8Array([95, 175, 255]), 3, 1, THREE.RedFormat);
tones.minFilter = tones.magFilter = THREE.NearestFilter;
tones.needsUpdate = true;
const toon = (color, extra = {}) => new THREE.MeshToonMaterial({ color, gradientMap: tones, ...extra });

const MAT = {
  gold: toon(0xf2c652),
  deck: toon(0xef6a4c),
  band: toon(0xeb6834),
  panel: toon(0x2b3d8f),
  metal: toon(0xdfe4e6),
  piston: toon(0xf7f8f8),
  engine: toon(0x5c6369, { emissive: 0x000000, side: THREE.DoubleSide }),
  dark: toon(0x3d3935),
  dish: toon(0xfbfaf3, { side: THREE.DoubleSide }),
  lens: new THREE.MeshBasicMaterial({ color: 0x1d2a26 }),
  lampOff: new THREE.MeshBasicMaterial({ color: 0x3a3632 }),
  lampOn: new THREE.MeshBasicMaterial({ color: 0x3987e5 }),
};
const GHOST = new THREE.MeshBasicMaterial({ color: 0x4f97ee, transparent: true, opacity: 0.36, depthWrite: false });
const outlineMaterial = new THREE.ShaderMaterial({
  uniforms: { uWidth: { value: OUTLINE } },
  vertexShader: /* glsl */ `
    uniform float uWidth;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vec3 n = normalize(normalMatrix * normal);
      mv.xyz += n * uWidth;
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `void main() { gl_FragColor = vec4(${[(INK >> 16) / 255, ((INK >> 8) & 255) / 255, (INK & 255) / 255].map((v) => v.toFixed(3)).join(", ")}, 1.0); }`,
  side: THREE.BackSide,
});

// Geometries, shared by every lander. An octagon is a cylinder with 8 sides, turned so a face looks forward.
const oct = (rTop, rBottom, h) => {
  const g = new THREE.CylinderGeometry(rTop, rBottom, h, 8, 1);
  g.rotateY(Math.PI / 8);
  return g;
};
const GEO = {
  ds: oct(DS.r, DS.r, DS.y1 - DS.y0),
  band: oct(DS.r + 0.006, DS.r + 0.006, 0.035),
  deck: oct(AV.r1, AV.r0, AV.y1 - AV.y0),
  hatch: new THREE.CylinderGeometry(0.12, 0.12, 0.04, 18),
  gimbal: new THREE.CylinderGeometry(0.11, 0.11, 0.04, 18),
  bell: new THREE.CylinderGeometry(0.075, 0.17, 0.12, 22, 1, true),
  bellInside: new THREE.CircleGeometry(0.165, 22),
  rod: new THREE.CylinderGeometry(1, 1, 1, 10), // unit rod along y, scaled per strut
  pad: new THREE.CylinderGeometry(0.13, 0.15, 0.035, 18),
  lamp: new THREE.SphereGeometry(0.03, 10, 8),
  quad: new THREE.BoxGeometry(0.09, 0.09, 0.09),
  nozzle: new THREE.CylinderGeometry(0.018, 0.032, 0.05, 10),
  panel: new THREE.PlaneGeometry(0.34, 0.26),
  lens: new THREE.CircleGeometry(0.038, 14),
  dish: new THREE.CylinderGeometry(0.15, 0.03, 0.06, 18, 1, true),
  plumeOuter: new THREE.ConeGeometry(1, 1, 22, 1, true),
  plumeCore: new THREE.ConeGeometry(1, 1, 16, 1, true),
  puff: new THREE.ConeGeometry(1, 1, 10, 1, true),
};
// Ink outlines need smooth normals, or they crack open at the hard edges.
const OUTLINE_GEO = new Map();
const outlineOf = (geometry) => {
  if (!OUTLINE_GEO.has(geometry)) {
    const g = geometry.clone();
    g.deleteAttribute("normal");
    g.deleteAttribute("uv");
    const merged = mergeVertices(g, 1e-4);
    merged.computeVertexNormals();
    OUTLINE_GEO.set(geometry, merged);
  }
  return OUTLINE_GEO.get(geometry);
};
// The plume's cones hang from their tip at the origin, pointing down -y.
for (const g of [GEO.plumeOuter, GEO.plumeCore, GEO.puff]) g.translate(0, -0.5, 0); // tip at the origin, widening down

export function createLander({ ghost = false } = {}) {
  const root = new THREE.Group();
  const body = new THREE.Group();
  root.add(body);

  /** A part: a mesh with its ink outline (none on the ghost). */
  const part = (geometry, material, parent = body, ink = true) => {
    const mesh = new THREE.Mesh(geometry, ghost ? GHOST : material);
    parent.add(mesh);
    if (ink && !ghost) {
      const line = new THREE.Mesh(outlineOf(geometry), outlineMaterial);
      mesh.add(line);
    }
    return mesh;
  };

  // Hull: the gold descent stage with an orange band, the avionics deck, a hatch.
  part(GEO.ds, MAT.gold).position.y = (DS.y0 + DS.y1) / 2;
  part(GEO.band, MAT.band, body, false).position.y = DS.y1 - 0.06;
  part(GEO.deck, MAT.deck).position.y = (AV.y0 + AV.y1) / 2;
  part(GEO.hatch, MAT.metal).position.set(-0.08, AV.y1 + 0.02, -0.08);
  // Solar panels on the four diagonal faces of the descent stage.
  const apothem = DS.r * Math.cos(Math.PI / 8);
  for (let k = 0; k < 4; k++) {
    const a = Math.PI / 4 + (k * Math.PI) / 2;
    const p = part(GEO.panel, MAT.panel, body, false);
    p.position.set(Math.sin(a) * (apothem + 0.004), (DS.y0 + DS.y1) / 2 - 0.01, Math.cos(a) * (apothem + 0.004));
    p.rotation.y = a;
  }
  // Two hazard cameras on the deck's front face.
  for (const x of [-0.12, 0.12]) {
    const lens = part(GEO.lens, MAT.lens, body, false);
    lens.position.set(x, 0.3, AV.r0 * Math.cos(Math.PI / 8) - 0.06);
    lens.rotation.x = -0.21;
  }
  // High-gain antenna on a mast.
  const mast = part(GEO.rod, MAT.metal);
  mast.scale.set(0.014, 0.26, 0.014);
  mast.position.set(-0.18, AV.y1 + 0.13, 0.12);
  const dish = part(GEO.dish, MAT.dish);
  dish.position.set(-0.18, AV.y1 + 0.28, 0.12);
  dish.rotation.set(0.5, 0, 0.35);

  // Engine: gimbal ring and bell; the bell's inside glows as it heats.
  part(GEO.gimbal, MAT.dark).position.y = DS.y0 - 0.02;
  const bellMat = ghost ? GHOST : MAT.engine.clone();
  const bell = new THREE.Mesh(GEO.bell, bellMat);
  bell.position.y = NOZZLE_EXIT.y + 0.06;
  body.add(bell);
  if (!ghost) bell.add(new THREE.Mesh(outlineOf(GEO.bell), outlineMaterial));
  const mouth = new THREE.Mesh(GEO.bellInside, ghost ? GHOST : new THREE.MeshBasicMaterial({ color: 0x1a1816 }));
  mouth.rotation.x = Math.PI / 2;
  mouth.position.y = NOZZLE_EXIT.y + 0.002;
  body.add(mouth);

  // Thruster pods on booms at the deck's four diagonal corners, at the side engines' height.
  const pods = [];
  for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const at = new THREE.Vector3(sx * 0.42, RCS_PORTS[0].y, sz * 0.42);
    alignRod(part(GEO.rod, MAT.metal, body, false), new THREE.Vector3(sx * 0.3, at.y, sz * 0.3), at, 0.012);
    part(GEO.quad, MAT.dark).position.copy(at);
    for (const d of [[0, 1, 0], [0, -1, 0], [sx, 0, 0]]) {
      const n = part(GEO.nozzle, MAT.engine, body, false);
      n.position.copy(at).addScaledVector(new THREE.Vector3(...d), 0.065);
      n.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...d));
    }
    const puff = new THREE.Mesh(GEO.puff, new THREE.MeshBasicMaterial({ color: 0xf4f6f8, transparent: true, opacity: ghost ? 0.4 : 0.85, depthWrite: false }));
    puff.position.copy(at).add(new THREE.Vector3(sx * 0.09, 0, 0));
    puff.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), new THREE.Vector3(sx, 0, 0));
    puff.visible = false;
    body.add(puff);
    pods.push({ side: sx, puff });
  }

  // Legs: struts rebuilt every frame from the hull to each footpad.
  const legs = [];
  for (const dir of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    const leg = {
      dir,
      main: part(GEO.rod, MAT.metal),
      piston: part(GEO.rod, MAT.piston),
      brace: part(GEO.rod, MAT.metal),
      pad: part(GEO.pad, MAT.metal),
      probe: part(GEO.rod, MAT.metal, body, false),
      lamp: new THREE.Mesh(GEO.lamp, ghost ? GHOST : MAT.lampOff),
      foot: new THREE.Vector3(),
    };
    body.add(leg.lamp);
    legs.push(leg);
  }

  // Plume: an orange cone with a bright core, hanging from the bell; it widens in vacuum.
  const plume = new THREE.Group();
  plume.position.y = NOZZLE_EXIT.y;
  const outer = new THREE.Mesh(GEO.plumeOuter, new THREE.MeshBasicMaterial({ color: 0xff9a4a, transparent: true, opacity: ghost ? 0.3 : 0.62, depthWrite: false }));
  const core = new THREE.Mesh(GEO.plumeCore, new THREE.MeshBasicMaterial({ color: 0xfff1b8, transparent: true, opacity: ghost ? 0.45 : 0.95, depthWrite: false }));
  plume.add(outer, core);
  plume.visible = false;
  body.add(plume);
  const light = ghost ? null : new THREE.PointLight(0xff8a3d, 0, 5, 1.6);
  if (light) {
    light.position.y = NOZZLE_EXIT.y - 0.2;
    body.add(light);
  }

  const inv = new THREE.Matrix4(), v = new THREE.Vector3(), hinge = new THREE.Vector3(), low = new THREE.Vector3(), mid = new THREE.Vector3(), brace = new THREE.Vector3();
  const footBody = [new THREE.Vector3(), new THREE.Vector3()];
  let heat = 0, flicker = 1, plumeLen = 0;

  return {
    root,
    body,
    update(pose) {
      body.position.set(pose.x, pose.y, 0);
      body.rotation.set(0, 0, pose.angle);
      body.updateMatrixWorld();
      inv.copy(body.matrixWorld).invert();
      // The physics feet, in the hull's frame.
      pose.legs.forEach((l, i) => {
        v.set(l.x + Math.sin(l.angle) * FOOT, l.y - Math.cos(l.angle) * FOOT, 0).applyMatrix4(inv);
        v.z = 0;
        footBody[i].copy(v);
      });
      const reach = (Math.abs(footBody[0].x) + Math.abs(footBody[1].x)) / 2;
      const drop = (footBody[0].y + footBody[1].y) / 2;
      const touching = pose.contact[0] || pose.contact[1];
      legs.forEach((leg, i) => {
        const [dx, dz] = leg.dir;
        if (i < 2) leg.foot.copy(footBody[i]);
        else leg.foot.set(0, drop, dz * reach);
        hinge.set(dx, 0, dz).multiplyScalar(apothem - 0.03).setY(DS.y1 - 0.09);
        low.set(dx, 0, dz).multiplyScalar(apothem - 0.02).setY(DS.y0 + 0.05);
        const padTop = v.copy(leg.foot).setY(leg.foot.y + 0.035);
        mid.lerpVectors(hinge, padTop, 0.55);
        alignRod(leg.main, hinge, mid, 0.034);
        alignRod(leg.piston, mid, padTop, 0.022);
        brace.lerpVectors(hinge, padTop, 0.72);
        alignRod(leg.brace, low, brace, 0.016);
        leg.pad.position.copy(leg.foot).setY(leg.foot.y + 0.017);
        leg.lamp.position.copy(leg.foot).setY(leg.foot.y + 0.045);
        const contact = i < 2 ? pose.contact[i] : touching;
        if (!ghost) leg.lamp.material = contact ? MAT.lampOn : MAT.lampOff;
        // Contact probe: hangs 14 cm under the pad until it touches.
        leg.probe.visible = !contact;
        if (!contact) alignRod(leg.probe, leg.foot, v.copy(leg.foot).add(mid.set(dx * 0.03, -0.14, dz * 0.03)), 0.007);
      });
    },
    engines({ main, side, air, dt, time }) {
      // Heat builds while the engine burns and fades after.
      heat = Math.min(1, Math.max(0, heat + (main > 0 ? dt * 0.5 : -dt * 0.25)));
      if (!ghost) bellMat.emissive.setRGB(heat * 0.9, heat * 0.32, heat * 0.06);
      // Plume length eases with the throttle; in vacuum it spreads wide and thin, in thick air it stays tight.
      const target = main > 0 ? 0.55 + 0.65 * main : 0;
      plumeLen += (target - plumeLen) * Math.min(1, dt * 18);
      flicker = 0.9 + 0.08 * Math.sin(time * 61) + 0.04 * Math.sin(time * 23);
      plume.visible = plumeLen > 0.03;
      if (plume.visible) {
        const spread = air < 0.005 ? 0.42 : air < 0.5 ? 0.26 : 0.18;
        const len = plumeLen * flicker * (air >= 1 ? 0.7 : 1);
        outer.scale.set(spread * len + 0.12, len, spread * len + 0.12);
        core.scale.set(0.07, len * 0.55, 0.07);
      }
      if (light) light.intensity = main > 0 ? 2.2 * main * flicker : 0;
      // Side thrusters: the "left engine" (side < 0) fires from the +x pods and pushes the lander left.
      for (const pod of pods) {
        const firing = side !== 0 && Math.sign(side) === -pod.side;
        pod.puff.visible = firing;
        if (firing) {
          const k = Math.abs(side) * (0.85 + 0.15 * Math.sin(time * 47 + pod.side));
          pod.puff.scale.set(0.05 * k, 0.24 * k, 0.05 * k);
        }
      }
    },
    dispose() {
      root.removeFromParent();
      bellMat.dispose?.();
    },
  };
}

const UP = new THREE.Vector3(0, 1, 0), dirTmp = new THREE.Vector3();
/** Stretch a unit rod (radius 1, length 1 along y) between two points. */
function alignRod(mesh, a, b, radius) {
  dirTmp.subVectors(b, a);
  const length = dirTmp.length() || 1e-4;
  mesh.position.addVectors(a, b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(UP, dirTmp.divideScalar(length));
  mesh.scale.set(radius, length, radius);
}
