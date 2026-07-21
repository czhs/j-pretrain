"""Regenerate the Figure-3a replication plot directly from ``results.csv``.

Two panels mirroring the paper's Figure 3a conceptual structure:

* Left:  x=lambda, y=L_im  (immediate MusicPile val loss after Stage-2 convergence).
* Right: x=lambda, y=L_ret (retained MusicPile val loss after Stage-3 ChemPile FT).

A single 300M-subset curve per panel (this reduced reproduction uses only 300M).
No manual data entry: everything is read from results.csv, so the figure is fully
reproducible. Missing conditions are simply absent points (never silently faked).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def _read(csv_path: str | Path) -> list[tuple[float, float | None, float | None]]:
    rows = []
    for r in csv.DictReader(Path(csv_path).open()):
        def f(k):
            v = r.get(k)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        rows.append((float(r["lambda"]), f("L_im"), f("L_ret")))
    return sorted(rows, key=lambda t: t[0])


def make_figure(csv_path: str | Path, out_dir: str | Path,
                subset_label: str = "300M MusicPile subset",
                title: str = "Figure 3a replication (reduced: 300M subset, 1 seed)") -> list[Path]:
    rows = _read(csv_path)
    lam = [r[0] for r in rows]
    im = [(l, v) for l, _, v in ((r[0], None, r[1]) for r in rows) if v is not None]
    ret = [(l, v) for l, v in ((r[0], r[2]) for r in rows) if v is not None]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.2))
    if im:
        axL.plot([l for l, _ in im], [v for _, v in im], "o-", color="#1f77b4", label=subset_label)
    axL.set_xlabel("lambda (fraction of 300M MusicPile shown in Stage 1)")
    axL.set_ylabel("L_im  (MusicPile val loss, post Stage-2)")
    axL.set_title("Immediate: L_im vs lambda")
    axL.grid(True, alpha=0.3)
    axL.legend()

    if ret:
        axR.plot([l for l, _ in ret], [v for _, v in ret], "s-", color="#d62728", label=subset_label)
    axR.set_xlabel("lambda (fraction of 300M MusicPile shown in Stage 1)")
    axR.set_ylabel("L_ret  (MusicPile val loss, post Stage-3)")
    axR.set_title("Retained: L_ret vs lambda")
    axR.grid(True, alpha=0.3)
    axR.legend()

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"figure3a_replication.{ext}"
        fig.savefig(p, dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths
