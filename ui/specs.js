// Facts about flying the lander on a world, worked out from the simulator's own numbers.

// The simulator's lander (Gymnasium's LunarLander, as wrapped by lander.py). Box2D sizes are
// pixels at 30 per meter; masses come from the shapes' areas and densities.
const HULL_MASS = 5 * (867 / 900); // density 5 on the hull polygon's 0.96 m²
const LEG_MASS = 1 * (64 / 900); // density 1 on each 0.13 x 0.53 m leg
const MASS = HULL_MASS + 2 * LEG_MASS; // about 4.96 kg per meter of depth (a 2D simulator)
const MAIN_IMPULSE = 13 * (4 / 30); // N·s per frame at full power, before the world's engine factor
const SIDE_IMPULSE = 0.6 * (12 / 30);
const FPS = 50;
export const MAIN_ACCEL = (MAIN_IMPULSE * FPS) / MASS; // about 17.5 m/s² at full power
const SIDE_ACCEL = (SIDE_IMPULSE * FPS) / MASS;

/** Facts about flying the lander on a world, from its gravity, engine factor and air. */
export function specs(world) {
  const g = Math.abs(world.gravity);
  const accel = MAIN_ACCEL * world.engine_power;
  const twr = accel / g;
  const hover = 1 / twr;
  return [
    ["Thrust to weight", `${twr.toFixed(2)} : 1`, twr < 1.3 ? "barely more than its weight: brake early" : twr > 3 ? "plenty: short, gentle pulses" : "comfortable margin"],
    ["Full-power braking", `${(accel - g).toFixed(1)} m/s²`, "main engine against gravity"],
    ["Hover throttle", hover <= 1 ? `${Math.round(hover * 100)}%` : "can't hover", hover < 0.5 ? "below the throttle's 50% floor: pulse it" : "throttle pilots can hold a hover"],
    ["Side thrusters", `${SIDE_ACCEL.toFixed(1)} m/s²`, "sideways nudge, plus a turn"],
    ["Air", world.atmosphere ? `drag ${world.atmosphere}` : "vacuum", world.atmosphere >= 1 ? "thick: it slows every move" : world.atmosphere ? "some drag" : "no drag, no smoke"],
    ["Wind", world.enable_wind ? `${world.wind_power} · gusts ${world.turbulence_power}` : "none", world.enable_wind ? "pushes and twists the rocket" : "still"],
  ];
}
