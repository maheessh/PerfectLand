"""The game's static build: GitHub Pages must ship every module play.html loads."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"


def local_imports(path):
    """Modules a file loads: static imports and workers (new URL("./x.js", import.meta.url))."""
    text = path.read_text()
    return set(re.findall(r'from\s+"\./([\w.-]+\.js)"', text)) | set(re.findall(r'new URL\("\./([\w.-]+\.js)"', text))


def test_pages_build_ships_every_module_the_game_imports():
    needed, todo = set(), ["play.js"]
    while todo:
        name = todo.pop()
        if name in needed:
            continue
        needed.add(name)
        todo.extend(local_imports(UI / name))
    workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text()
    shipped = set(re.findall(r"ui/([\w.-]+\.js)", workflow))
    assert needed <= shipped, f"pages.yml is missing {sorted(needed - shipped)}"
    assert "cp -r ui/vendor" in workflow, "pages.yml must ship ui/vendor (planck.js, used by sim.js)"


def test_game_page_uses_relative_paths():
    """On GitHub Pages the game lives under /PerfectLand/, so absolute paths would break."""
    html = (UI / "play.html").read_text()
    assert 'src="play.js"' in html and 'href="styles.css"' in html
    assert not re.search(r'(src|href)="/(?!/)', html)


def test_built_figures_match_their_sources():
    """ui/figures/<name>.html must be rebuilt (python ui/figures/build.py) after editing src/<name>.html."""
    src_dir = UI / "figures" / "src"
    for src in src_dir.glob("*.html"):
        built = (UI / "figures" / src.name).read_text()
        figure_script = src.read_text().split('<script src="core.js"></script>', 1)[1].split("</body>")[0].strip()
        assert figure_script in built, f"{src.name} changed since it was built: run python ui/figures/build.py"
    core = (src_dir / "core.js").read_text()
    for built in (UI / "figures").glob("*.html"):
        assert core.strip()[:200] in built.read_text(), f"{built.name} was built with another core.js"
