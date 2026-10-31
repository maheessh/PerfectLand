// "My AIs": pilots trained in this browser, saved in this browser (IndexedDB). Nothing is uploaded:
// each person's AIs stay on their own device. Export/import moves one to another device as a file.

const DB = "perfectland", STORE = "pilots", VERSION = 1;
let dbPromise = null;

function open() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB, VERSION);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "id" });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}

async function run(mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode), req = fn(tx.objectStore(STORE));
    tx.oncomplete = () => resolve(req?.result);
    tx.onerror = () => reject(tx.error);
  });
}

export const myAIs = {
  /** Every saved pilot, newest first. Empty (not an error) where storage is blocked. */
  async list() {
    try {
      const all = (await run("readonly", (s) => s.getAll())) || [];
      return all.sort((a, b) => b.updated - a.updated);
    } catch { return []; }
  },
  async get(id) { return run("readonly", (s) => s.get(id)); },
  async save(pilot) { pilot.updated = Date.now(); await run("readwrite", (s) => s.put(pilot)); return pilot; },
  async remove(id) { await run("readwrite", (s) => s.delete(id)); },
  /** A new pilot record (not saved yet). */
  create({ name, world }) {
    const id = `ai-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
    return { id, name, world, created: Date.now(), updated: Date.now(), frames: 0, sessions: 0, best: null, weights: null, latest: null, history: [] };
  },
  /** Download a pilot as a .json file. */
  download(pilot) {
    const blob = new Blob([JSON.stringify({ perfectland: 1, ...pilot })], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `${pilot.name.replace(/[^\w-]+/g, "-")}.perfectland.json` });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  },
  /** Read a file made by download(); returns the saved pilot. */
  async upload(file) {
    const data = JSON.parse(await file.text());
    if (!data.perfectland || !data.weights?.actor) throw new Error("That isn't a PerfectLand AI file.");
    delete data.perfectland;
    const mine = await this.list();
    if (mine.some((p) => p.id === data.id)) data.id = `${data.id}-copy`;
    return this.save(data);
  },
};

/** The player's name, for "Mahesh's AI" and shared results. Kept in this browser only. */
export const profile = {
  get name() { try { return localStorage.getItem("perfectland.name") || ""; } catch { return ""; } },
  set name(v) { try { localStorage.setItem("perfectland.name", v.trim().slice(0, 30)); } catch { /* fine without */ } },
};
