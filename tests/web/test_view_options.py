"""The view options of the plots (#452): the probability scale from linear to logarithmic (Box-Cox),
the colour maps and the stored options. The functions are plain ES modules, run in Node."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

COMPONENTS = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app" / "components"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")


def node(script: str, tmp_path: Path) -> object:
    path = tmp_path / "check.mjs"
    path.write_text(script, encoding="utf-8")
    return json.loads(subprocess.run(["node", str(path)], capture_output=True, text=True, timeout=60,
                                     check=True).stdout)


SCALE_CHECK = """
const m = await import(%(module)r);
const values = [0.4, 0.2, 0.05, 0.003, 1e-6, 0, null];
const grid = Array.from({ length: 400 }, (_, i) => 10 ** (-8 + (8 * i) / 399) * 0.4);
const lambdas = [1, 0.9, 0.75, 0.5, 0.3, 0.1, 0.05, 0];
const result = { monotone: [], ticksOrdered: [], ticksInside: [], ends: [], continuous: [] };
for (const decades of [false, true]) {
  for (const lambda of lambdas) {
    const s = m.probabilityScale(values, { lambda, decades });
    const ts = grid.map((p) => s.t(p));
    result.monotone.push(ts.every((t, i) => i === 0 || t >= ts[i - 1] - 1e-12)
      && ts.every((t) => t >= 0 && t <= 1));
    result.ticksOrdered.push(s.ticks.every((p, i) => i === 0 || p > s.ticks[i - 1]));
    result.ticksInside.push(s.ticks.every((p) => p >= s.min - 1e-15 && p <= s.max + 1e-15)
      && s.ticks.map((p) => s.t(p) ?? 0).every((t, i, all) => i === 0 || t > all[i - 1]));
    result.ends.push([s.t(s.max), s.ticks.at(-1) === s.max]);
  }
  // the scale depends continuously on λ: λ = 0.05 is close to the logarithm, 0.95 close to linear
  const near = (a, b) => Math.max(...grid.map((p) => Math.abs((a.t(p) ?? 0) - (b.t(p) ?? 0))));
  result.continuous.push([
    near(m.probabilityScale(values, { lambda: 0.05, decades }), m.probabilityScale(values, { lambda: 0, decades })),
    near(m.probabilityScale(values, { lambda: 0.95, decades }), m.probabilityScale(values, { lambda: 1, decades }))]);
}
const lin = m.probabilityScale([0.5, 0.25, 0.1, 0]);
const log = m.probabilityScale([0.5, 0.005, 0], { lambda: 0 });
const above = m.probabilityScale([0.5, 0.25], { lambda: 0.5 });
console.log(JSON.stringify({
  ...result,
  linear: [lin.min, lin.max, lin.t(0.25), lin.t(0.1), lin.t(0), lin.t(null), lin.ticks],
  log: [log.min, log.max, log.t(0.05), log.t(0.005)],
  boxCoxLimit: Math.abs(m.boxCox(0.3, 1e-7) - Math.log(0.3)),
  boxCoxLinear: m.boxCox(0.3, 1),
  tallerThanMax: [above.t(0.8), above.height(0.8) > 1],
  empty: [m.probabilityScale([0, null]).t(0.5), m.probabilityScale([0, null]).max],
  decadeLog: m.probabilityScale([0.5, 0.25, 0.001, 0], { lambda: 0, decades: true }).ticks,
  snap: [m.snapLambda(0.97), m.snapLambda(0.52), m.snapLambda(0.02), m.snapLambda(0.333), m.snapLambda(-1), m.snapLambda(NaN)],
  names: [m.scaleName(1), m.scaleName(0.5), m.scaleName(0), m.scaleName(0.3)],
}));
"""


@needs_node
def test_probability_scale_from_linear_to_log(tmp_path):
    """The Box-Cox scale (#452) is monotone in p and maps into [0, 1] for every λ, its ticks lie in
    order on the scale, λ = 1 is the linear scale from 0 and λ = 0 the logarithmic scale from the
    smallest value; λ near 0 is close to the logarithm, λ near 1 close to linear (continuity)."""
    out = node(SCALE_CHECK % {"module": (COMPONENTS / "plot-scale.js").as_uri()}, tmp_path)
    assert all(out["monotone"]) and all(out["ticksOrdered"]) and all(out["ticksInside"])
    assert all(t == 1 and last for t, last in out["ends"])
    for near_log, near_linear in out["continuous"]:
        assert near_log < 0.08 and near_linear < 0.05
    assert out["linear"] == [0, 0.5, 0.5, pytest.approx(0.2), None, None, [0, 0.25, 0.5]]
    lo, hi, middle, bottom = out["log"]
    assert lo == pytest.approx(0.005) and hi == 0.5 and middle == pytest.approx(0.5) and bottom == pytest.approx(0, abs=1e-9)
    assert out["boxCoxLimit"] < 1e-6 and out["boxCoxLinear"] == pytest.approx(-0.7)
    # a bar above the largest value (the "?" tower) keeps its height; colours stay clamped
    assert out["tallerThanMax"] == [1, True]
    assert out["empty"] == [None, 0]
    assert out["decadeLog"] == pytest.approx([1e-4, 1e-3, 1e-2, 1e-1, 1])
    assert out["snap"] == [1, 0.5, 0, 0.33, 0, 1]
    assert out["names"] == ["Linear (λ = 1)", "Square root (λ = 0.50)", "Logarithmic (λ → 0)", "Power (λ = 0.30)"]


COLOUR_CHECK = """
const m = await import(%(module)r);
const contrast = (a, b) => { const [x, y] = [m.luminance(a), m.luminance(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
const worst = {};
for (const name of Object.keys(m.COLOUR_MAPS)) {
  let low = Infinity;
  for (let i = 0; i <= 100; i++) {
    const bg = m.colourAt(name, i / 100);
    low = Math.min(low, contrast(bg, m.textOn(bg)));
  }
  worst[name] = low;
}
const lum = (name) => Array.from({ length: 11 }, (_, i) => m.luminance(m.colourAt(name, i / 10)));
console.log(JSON.stringify({ worst, ends: [m.colourAt("viridis", 0), m.colourAt("viridis", 1), m.colourAt("unknown", 1)],
  monotone: Object.fromEntries(Object.keys(m.COLOUR_MAPS).map((n) => { const l = lum(n);
    return [n, l.every((v, i) => i === 0 || v > l[i - 1]) || l.every((v, i) => i === 0 || v < l[i - 1])]; })) }));
"""


@needs_node
def test_colour_maps_and_text_on_cells(tmp_path):
    """Viridis, cividis and greyscale (#452): each runs monotonically in lightness, so that it reads
    in greyscale too, and the text on a cell (black or white, whichever contrasts more) has at least
    4.5:1 on every colour of every map."""
    out = node(COLOUR_CHECK % {"module": (COMPONENTS / "plot-scale.js").as_uri()}, tmp_path)
    assert set(out["worst"]) == {"viridis", "cividis", "greyscale"}
    assert all(value >= 4.5 for value in out["worst"].values()), out["worst"]
    assert out["ends"] == ["#440154", "#fde725", "#fde725"]
    assert all(out["monotone"].values())


OPTIONS_CHECK = """
const store = new Map();
globalThis.localStorage = { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, String(v)) };
let contrast = false;
globalThis.matchMedia = (query) => ({ matches: contrast && query.includes("contrast") });
const v = await import(%(module)r);
const fresh = v.loadViewOptions();
store.set("prodigy.gui.chartScale", "log");
const migrated = v.loadViewOptions();
contrast = true;
const outline = v.loadViewOptions();
contrast = false;
const bad = v.normaliseOptions({ lambda: "x", colours: "__proto__", intervals: "toString", unaccounted: "yes", open: 1 });
const saved = v.saveViewOptions({ lambda: 0.52, colours: "cividis", open: true });
const reloaded = v.loadViewOptions();
globalThis.localStorage = { getItem: () => { throw new Error("denied"); }, setItem: () => { throw new Error("denied"); } };
const denied = v.normaliseOptions(null);
const kept = v.saveViewOptions({ colours: "greyscale" });
console.log(JSON.stringify({ fresh, migrated: migrated.lambda, outline: outline.intervals, bad, saved, reloaded,
  denied, kept: [kept.colours, v.viewOptions().colours] }));
"""


@needs_node
def test_view_options_are_validated_and_stored_per_viewer(tmp_path):
    """The options are stored per viewer in localStorage (#452): checked against their allowed
    values, the old linear/log choice taken over, Outline the default with increased contrast,
    whether the inspector is open remembered, and the page keeps working when storage throws."""
    out = node(OPTIONS_CHECK % {"module": (COMPONENTS / "view-options.js").as_uri()}, tmp_path)
    assert out["fresh"] == {"lambda": 1, "colours": "viridis", "intervals": "glass", "unaccounted": True, "open": False}
    assert out["migrated"] == 0 and out["outline"] == "outline"
    assert out["bad"] == {"lambda": 1, "colours": "viridis", "intervals": "glass", "unaccounted": True, "open": False}
    assert out["saved"]["lambda"] == 0.5 and out["saved"]["colours"] == "cividis" and out["saved"]["open"] is True
    assert out["reloaded"] == out["saved"]
    assert out["denied"]["colours"] == "viridis" and out["kept"] == ["greyscale", "greyscale"]


def test_view_options_replace_the_scale_switch():
    """One "View options" toggle (#452) replaces the linear/log switch next to the view switch of the
    analysis card and of a marginal figure. It shows an inline inspector, a labelled region beside or
    above the plot that never covers it (no dialog, no overlay); Escape inside it closes it and
    returns focus. The plots take scale and colours from it."""
    control = (COMPONENTS / "view-options.js").read_text(encoding="utf-8")
    for needle in ('"aria-expanded"', '"aria-controls"', 'role: "region"', '"Escape"', "button.focus()",
                   "aria-valuetext", "prefers-reduced-transparency", "prefers-contrast: more", "try {"):
        assert needle in control, needle
    assert 'role: "dialog"' not in control and "pointerdown" not in control
    css = (COMPONENTS / "result-view.css").read_text(encoding="utf-8")
    assert "container-type: inline-size" in css and "@container (min-width: 1000px)" in css
    assert "vo-popover" not in css
    for name in ("analysis-card.js", "parts.js"):
        source = (COMPONENTS / name).read_text(encoding="utf-8")
        assert "viewOptionsControl(" in source and ".frame(" in source and "onViewOptionsChange(" in source, name
        assert "loadScale" not in source and '["log", "Log"]]' not in source.split("sizeScale")[0], name
    joint = "".join((COMPONENTS / name).read_text(encoding="utf-8") for name in ("joint-view.js", "joint-bars3d.js"))
    assert "ownUpperEnds" in joint and '"tower"' in joint and "colours.colour(" in joint
    charts = (COMPONENTS / "charts.js").read_text(encoding="utf-8")
    assert "view.unaccounted" in charts and "mc-ext-hover" in charts and "probabilityScale(" in charts
