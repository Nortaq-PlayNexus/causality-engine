"""Structural Causal Model: variables, equations, noise, interventions, counterfactuals."""
import random, math, copy
from typing import Dict, List, Any, Callable, Optional
from .causal_graph import CausalGraph, Node

class SCM:
    def __init__(self, graph: CausalGraph, equations: Optional[Dict[str, Callable]] = None,
                 noise_std: float = 0.1, seed: int = 0):
        self.graph = graph
        self.equations = equations or {}
        self.noise_std = noise_std
        self.rng = random.Random(seed)
        self.values: Dict[str, float] = {}
        self.noise: Dict[str, float] = {}

    def _default_eq(self, nid, parent_vals):
        # linear sum of parents
        w = 0.0
        for p, val in parent_vals.items():
            w += 0.5 * val
        return w

    def sample(self, n_samples=1, interventions=None, seed=None):
        interventions = interventions or {}
        rng = random.Random(seed) if seed is not None else self.rng
        samples = []
        for _ in range(n_samples):
            self.noise = {nid: rng.gauss(0, self.noise_std) for nid in self.graph.nodes}
            self.values = {}
            # topological order
            order = self._topo_order()
            for nid in order:
                if nid in interventions:
                    self.values[nid] = interventions[nid]
                    continue
                parents = self.graph.parents(nid)
                pv = {p: self.values[p] for p in parents}
                eq = self.equations.get(nid)
                if eq is None:
                    base = self._default_eq(nid, pv)
                else:
                    base = eq(pv, self.noise[nid])
                self.values[nid] = base + self.noise[nid]
            samples.append(dict(self.values))
        return samples

    def _topo_order(self):
        indeg = {n: 0 for n in self.graph.nodes}
        for e in self.graph.edges:
            indeg[e.target] += 1
        from collections import deque
        q = deque([n for n in self.graph.nodes if indeg[n] == 0])
        order = []
        indeg = dict(indeg)
        while q:
            u = q.popleft()
            order.append(u)
            for c in self.graph.children(u):
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        return order

    def do(self, interventions: Dict[str, float], n_samples=1, seed=None):
        """Interventional distribution P(V | do(X=x))."""
        return self.sample(n_samples=n_samples, interventions=interventions, seed=seed)

    def counterfactual(self, observed: Dict[str, float], interventions: Dict[str, float], seed=None):
        """Abduction + action + prediction. Returns counterfactual outcome."""
        rng = random.Random(seed) if seed is not None else self.rng
        # abduction: infer noise from observed
        # We approximate by computing noise residuals given observed and equations
        noise_est = {}
        order = self._topo_order()
        vals = dict(observed)
        for nid in order:
            if nid in observed:
                parents = self.graph.parents(nid)
                pv = {p: vals.get(p, 0.0) for p in parents}
                eq = self.equations.get(nid)
                base = eq(pv, 0.0) if eq else self._default_eq(nid, pv)
                noise_est[nid] = observed[nid] - base
            else:
                noise_est[nid] = rng.gauss(0, self.noise_std)
        # action + prediction with inferred noise, applying intervention
        self.noise = noise_est
        self.values = {}
        for nid in order:
            if nid in interventions:
                self.values[nid] = interventions[nid]
                continue
            parents = self.graph.parents(nid)
            pv = {p: self.values.get(p, 0.0) for p in parents}
            eq = self.equations.get(nid)
            base = eq(pv, self.noise.get(nid, 0.0)) if eq else self._default_eq(nid, pv)
            self.values[nid] = base + (0.0 if nid in observed else self.noise.get(nid, 0.0))
            # keep noise contribution
            if nid not in observed:
                self.values[nid] = base + self.noise.get(nid, 0.0)
        # simpler: recompute properly
        self.values = {}
        for nid in order:
            if nid in interventions:
                self.values[nid] = interventions[nid]
                continue
            parents = self.graph.parents(nid)
            pv = {p: self.values.get(p, 0.0) for p in parents}
            eq = self.equations.get(nid)
            base = eq(pv, self.noise.get(nid, 0.0)) if eq else self._default_eq(nid, pv)
            # base already includes noise? equations receive noise separately; default ignores noise
            self.values[nid] = base + self.noise.get(nid, 0.0) if eq is None else base
        return dict(self.values)

    def to_dict(self):
        return {"graph": self.graph.to_dict(), "noise_std": self.noise_std}

    @classmethod
    def from_dict(cls, d, equations=None, seed=0):
        from .causal_graph import CausalGraph
        g = CausalGraph.from_dict(d["graph"])
        return cls(g, equations=equations, noise_std=d.get("noise_std", 0.1), seed=seed)
