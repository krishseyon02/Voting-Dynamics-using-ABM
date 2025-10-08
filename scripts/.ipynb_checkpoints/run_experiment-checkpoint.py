import argparse, os, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.voting_model import VotingABM

def sweep(influences, noises, repeats=5, steps=200, graph="ws", n=200, p_edge=0.05, seed=42):
    rows = []
    for inf in influences:
        for noi in noises:
            for r in range(repeats):
                m = VotingABM(n=n, p_edge=p_edge, influence=inf, noise=noi,
                              steps=steps, graph=graph, seed=seed + r)
                metrics = m.run()
                rows.append({
                    "influence": inf,
                    "noise": noi,
                    "repeat": r,
                    "consensus": int(metrics["consensus"]),
                    "final_variance": metrics["final_variance"],
                    "minority_share": metrics["minority_share"]
                })
    return pd.DataFrame(rows)

def heatmap(df, value, x="influence", y="noise", out="figures/heatmap.png"):
    pivot = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    plt.figure()
    plt.imshow(pivot.values, origin="lower", aspect="auto")
    plt.xticks(range(pivot.shape[1]), [f"{c:.2f}" for c in pivot.columns])
    plt.yticks(range(pivot.shape[0]), [f"{r:.2f}" for r in pivot.index])
    plt.xlabel(x); plt.ylabel(y); plt.title(value)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.colorbar()
    plt.savefig(out, bbox_inches="tight")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--influences", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--noises", default="0.00,0.02,0.05,0.10")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--graph", choices=["erdos","ws","ba"], default="ws")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--p_edge", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    influences = [float(x) for x in args.influences.split(",")]
    noises     = [float(x) for x in args.noises.split(",")]

    Path("results").mkdir(exist_ok=True)
    df = sweep(influences, noises, args.repeats, args.steps, args.graph, args.n, args.p_edge, args.seed)
    out_csv = "results/param_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")

    heatmap(df, "consensus", out="figures/consensus_rate.png")
    heatmap(df, "minority_share", out="figures/minority_share.png")
    print("Figures written to figures/ folder")
