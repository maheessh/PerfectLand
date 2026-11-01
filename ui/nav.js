// The top bar, shared by the game and mission control: highlights the section you're reading,
// and on smaller screens folds the section links into a Menu.

export function setupNav() {
  const links = [...document.querySelectorAll(".nav-links a[href^='#'], .nav-panel a[href^='#']")];
  const sections = [...new Set(links.map((a) => a.getAttribute("href")))]
    .map((hash) => document.querySelector(hash))
    .filter(Boolean);

  // Scrollspy: the section that covers the line a third of the way down the screen is "here".
  const mark = () => {
    const line = innerHeight / 3;
    let here = null; // above the first section (the hero), nothing is highlighted
    for (const s of sections) if (s.getBoundingClientRect().top <= line) here = s;
    for (const a of links) {
      const on = here && a.getAttribute("href") === `#${here.id}`;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "location");
      else a.removeAttribute("aria-current");
    }
  };
  // The bar is see-through over the game's hero and turns solid once you scroll.
  const bar = document.querySelector(".nav");
  const solid = () => bar?.classList.toggle("scrolled", scrollY > 24);
  addEventListener("scroll", solid, { passive: true });
  solid();
  addEventListener("scroll", mark, { passive: true });
  addEventListener("resize", mark);
  mark();

  // Menu: the section links and the other page, for screens too narrow for the full bar.
  const button = document.getElementById("nav-menu");
  const panel = document.getElementById("nav-panel");
  if (!button || !panel) return;
  const open = (yes) => {
    panel.hidden = !yes;
    button.setAttribute("aria-expanded", String(yes));
    button.classList.toggle("on", yes);
  };
  button.addEventListener("click", (e) => {
    e.stopPropagation();
    open(panel.hidden);
  });
  panel.addEventListener("click", (e) => { if (e.target.closest("a")) open(false); });
  document.addEventListener("click", (e) => { if (!panel.hidden && !e.target.closest(".nav-panel")) open(false); });
  addEventListener("keydown", (e) => { if (e.key === "Escape" && !panel.hidden) { open(false); button.focus(); } });
}
