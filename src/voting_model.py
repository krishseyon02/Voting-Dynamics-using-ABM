# src/voting_model.py
from dataclasses import dataclass
import numpy as np, networkx as nx, json

@dataclass
class VotingABM:
    n:int=200
    p_edge:float=0.05
    influence:float=0.8
    noise:float=0.02
    zealot_frac:float=0.0
    steps:int=100
    seed:int=42
    graph:str="erdos"  # erdos | ws | ba

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        if self.graph == "erdos":
            self.G = nx.erdos_renyi_graph(self.n, self.p_edge, seed=self.seed)
        elif self.graph == "ws":
            k = max(2, int(self.p_edge * self.n))
            self.G = nx.watts_strogatz_graph(self.n, k, 0.1, seed=self.seed)
        else:
            m = max(1, int(self.p_edge * self.n // 2))
            self.G = nx.barabasi_albert_graph(self.n, m, seed=self.seed)

        self.opinions = self.rng.integers(0, 2, size=self.n)  # 0/1
        self.zealots = self.rng.random(self.n) < self.zealot_frac
        self.history = [self.opinions.copy()]

    def step(self):
        new = self.opinions.copy()
        current = self.history[-1]
        for i in range(self.n):
            if self.zealots[i]:  # zealots never change
                continue
            neigh = list(self.G.neighbors(i))
            if not neigh:
                continue
            majority = int(round(current[neigh].mean()))
            r = self.rng.random()
            if r < self.noise:
                new[i] = 1 - current[i]
            elif r < self.noise + self.influence:
                new[i] = majority
        self.opinions = new
        self.history.append(new.copy())

    def run(self):
        for _ in range(self.steps):
            self.step()
        return self.metrics()

    def metrics(self):
        hist = np.array(self.history)
        final = hist[-1]
        avg_over_time = hist.mean(axis=1)
        final_variance = float(final.var())
        minority_share = float(min(final.mean(), 1 - final.mean()))
        consensus = bool(final.mean() in (0.0, 1.0))
        return {
            "consensus": consensus,
            "final_variance": final_variance,
            "minority_share": minority_share,
            "avg_over_time": avg_over_time.tolist()
        }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--p_edge", type=float, default=0.05)
    p.add_argument("--influence", type=float, default=0.8)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--zealot_frac", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--graph", choices=["erdos","ws","ba"], default="erdos")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = VotingABM(**vars(args)).run()
    print(json.dumps(out, indent=2))
