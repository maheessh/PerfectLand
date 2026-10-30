// Cockpit-style status lamps over the 3D view (main engine, side thrusters, leg contact) and
// short callouts, the way crews called out a landing: "Contact light." ... "Engine stop."

const LAMPS = [
  ["main", "MAIN ENG"],
  ["rcs-l", "RCS ◀"],
  ["rcs-r", "RCS ▶"],
  ["contact", "CONTACT"],
];

export function createAnnunciator(lamps, callout) {
  lamps.classList.add("annunciator");
  lamps.replaceChildren(...LAMPS.map(([key, text]) => {
    const lamp = document.createElement("span");
    lamp.dataset.lamp = key;
    lamp.textContent = text;
    return lamp;
  }));
  const el = Object.fromEntries(LAMPS.map(([key]) => [key, lamps.querySelector(`[data-lamp="${key}"]`)]));
  let contact = false, burned = false, timer = null, lastText = "";
  const events = { contact: false, stop: false };

  function say(text, kind = "") {
    if (!callout || text === lastText) return;
    lastText = text;
    callout.textContent = text;
    callout.className = `callout show ${kind}`;
    clearTimeout(timer);
    timer = setTimeout(() => { callout.classList.remove("show"); lastText = ""; }, 2200);
  }

  return {
    /**
     * One frame: main 0..1, side -1..1 (negative: the left engine, which pushes left),
     * legs: [left touching, right touching]. Returns what just happened: { contact, stop }.
     */
    update(main, side, legs) {
      el.main.classList.toggle("on", main > 0);
      el["rcs-l"].classList.toggle("on", side < 0);
      el["rcs-r"].classList.toggle("on", side > 0);
      const touching = !!(legs[0] || legs[1]);
      el.contact.classList.toggle("on", touching);
      events.contact = touching && !contact;
      if (events.contact) say("Contact light", "blue");
      events.stop = touching && burned && main === 0 && !!(legs[0] && legs[1]);
      if (events.stop) say("Engine stop");
      burned = main > 0 ? true : touching ? false : burned;
      contact = touching;
      return events;
    },
    say,
    reset() {
      contact = burned = false;
      for (const lamp of Object.values(el)) lamp.classList.remove("on");
      callout?.classList.remove("show");
      lastText = "";
    },
  };
}
