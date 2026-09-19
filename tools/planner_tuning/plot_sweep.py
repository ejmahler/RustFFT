#!/usr/bin/env python3
"""Plot the output of `planner_tuning sweep`.

The sweep times the shipping planner's pick against the cost model's at every length in a
range. This draws the two views worth having:

  top     cost per transform normalised by N log2 N, both planners, so the shape of the
          whole population is visible at once
  bottom  the ratio planner/model per length, one trace per input file, so several cost
          model stages or several machines can be compared directly

Usage:
    plot_sweep.py sweep.tsv                          # one run
    plot_sweep.py a.tsv b.tsv c.tsv                  # compare runs in the ratio panel
    plot_sweep.py *.tsv --out fig.png                # write instead of showing
    plot_sweep.py sweep.tsv --lo 4 --hi 128          # zoom a length range

Needs matplotlib, which is the only third-party dependency in this directory:
    python3 -m venv .venv && .venv/bin/pip install matplotlib
    .venv/bin/python plot_sweep.py sweep_neon_f64.tsv
"""
import argparse
import math
import os
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(__doc__.rsplit("Needs matplotlib", 1)[0] +
             "matplotlib is not installed. From this directory:\n"
             "    python3 -m venv .venv && .venv/bin/pip install matplotlib\n"
             "    .venv/bin/python plot_sweep.py <sweep.tsv>")


def load(path):
    """Rows of a sweep TSV, plus whatever the header recorded about the run."""
    head, rows = {}, []
    for line in open(path):
        if line.startswith("#"):
            parts = line[1:].strip().split("\t")
            if len(parts) >= 2:
                head[parts[0]] = "\t".join(parts[1:])
            continue
        if line.startswith("len\t"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 10:
            continue
        rows.append(dict(n=int(f[0]), agree=f[2] == "1",
                         planner=float(f[3]), model=float(f[4]), ratio=float(f[5]),
                         pnorm=float(f[6]), mnorm=float(f[7]),
                         pspec=f[8], mspec=f[9]))
    if not rows:
        sys.exit(f"{path}: no data rows")
    return head, rows


def gmean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def summarise(path, head, rows):
    """The same figures the plot shows, as text, because they are what gets quoted."""
    dis = [r for r in rows if not r["agree"]]
    loss = [r for r in dis if r["ratio"] < 1 / 1.02]
    win = [r for r in dis if r["ratio"] > 1.02]
    worst = min(dis, key=lambda r: r["ratio"]) if dis else None
    best = max(dis, key=lambda r: r["ratio"]) if dis else None
    print(f"{os.path.basename(path)}  [{head.get('planner', '?')}]")
    print(f"  {len(rows)} lengths, {len(rows) - len(dis)} where both planners agree")
    print(f"  geometric mean planner/model {gmean([r['ratio'] for r in rows]):.4f}")
    print(f"  model wins beyond 2% at {len(win)}, loses beyond 2% at {len(loss)}")
    if worst:
        print(f"  best  {best['ratio']:.3f}x at N={best['n']}  model {best['mspec']}")
        print(f"  worst {worst['ratio']:.3f}x at N={worst['n']}  model {worst['mspec']}"
              f"  planner {worst['pspec']}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Plot planner_tuning sweep output.")
    ap.add_argument("files", nargs="+", help="sweep TSVs")
    ap.add_argument("--lo", type=int, default=4, help="lowest length to plot (default 4)")
    ap.add_argument("--hi", type=int, default=None, help="highest length to plot")
    ap.add_argument("--out", help="write the figure here instead of showing it")
    ap.add_argument("--quiet", action="store_true", help="skip the text summary")
    args = ap.parse_args()

    loaded = []
    for path in args.files:
        head, rows = load(path)
        rows = [r for r in rows
                if r["n"] >= args.lo and (args.hi is None or r["n"] <= args.hi)]
        if rows:
            loaded.append((path, head, rows))
    if not loaded:
        sys.exit("nothing left after the length filter")

    if not args.quiet:
        for path, head, rows in loaded:
            summarise(path, head, rows)

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(13, 8), height_ratios=[2, 1], sharex=True,
        gridspec_kw=dict(hspace=0.12))

    # Top: the population, from the first file only. Overlaying several runs here just
    # produces mud, and the runs differ in the model's pick rather than the planner's.
    path, head, rows = loaded[0]
    ax.scatter([r["n"] for r in rows], [r["pnorm"] for r in rows],
               s=7, alpha=0.65, label="fixed planner", color="#2a78d6")
    ax.scatter([r["n"] for r in rows], [r["mnorm"] for r in rows],
               s=7, alpha=0.65, label="estimating planner", color="#eb6834")
    ax.set_ylabel("ns per N log2 N")
    ax.set_title(f"{os.path.basename(path)}   planner {head.get('planner', '?')}")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.25)

    # Bottom: every file, so stages and machines line up against the same 1.0.
    for path, head, rows in loaded:
        bx.plot([r["n"] for r in rows], [r["ratio"] for r in rows],
                lw=0.9, alpha=0.85, label=os.path.basename(path))
    bx.axhline(1.0, color="0.4", lw=1)
    # Log scale so 1.25x up and 1.25x down are the same distance from 1.0, but with
    # explicit ticks: the default log locator labels only the decade and 1.0 is the
    # only decade anywhere near this data.
    bx.set_yscale("log")
    ticks = [0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    lo_r = min(r["ratio"] for _, _, rows in loaded for r in rows)
    hi_r = max(r["ratio"] for _, _, rows in loaded for r in rows)
    ticks = [t for t in ticks if lo_r / 1.05 <= t <= hi_r * 1.05] or [1.0]
    bx.set_yticks(ticks)
    bx.set_yticklabels([f"{t:g}x" for t in ticks])
    bx.minorticks_off()
    bx.set_ylabel("planner / model")
    bx.set_xlabel("transform length N")
    bx.grid(alpha=0.25, which="both")
    if len(loaded) > 1:
        bx.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    bx.annotate("estimate faster", xy=(0.005, 0.92), xycoords="axes fraction", fontsize=8)
    bx.annotate("planner faster", xy=(0.005, 0.04), xycoords="axes fraction", fontsize=8)

    if args.out:
        fig.savefig(args.out, dpi=140, bbox_inches="tight")
        print(f"wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
