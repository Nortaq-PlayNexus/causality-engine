from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
import random, math, copy, json


@dataclass
class Node:
    id: str
    name: str = ""
    observed: bool = True
    latent: bool = False
    domain: str = "continuous"  # continuous, discrete, binary
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {"id": self.id, "name": self.name, "observed": self.observed,
                "latent": self.latent, "domain": self.domain, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class Edge:
    source: str
    target: str
    weight: float = 1.0
    relation: str = "causal"  # causal, confounding
    lag: int = 0  # time lag for time-series

    def to_dict(self):
        return {"source": self.source, "target": self.target, "weight": self.weight,
                "relation": self.relation, "lag": self.lag}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class CausalGraph:
    """Directed acyclic graph (DAG) representing causal structure. Ground truth."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, e: Edge):
        if e.source in self.nodes and e.target in self.nodes:
            self.edges.append(e)

    def parents(self, nid):
        return [e.source for e in self.edges if e.target == nid]

    def children(self, nid):
        return [e.target for e in self.edges if e.source == nid]

    def ancestors(self, nid, seen=None):
        if seen is None: seen = set()
        for p in self.parents(nid):
            if p not in seen:
                seen.add(p)
                self.ancestors(p, seen)
        return seen

    def descendants(self, nid, seen=None):
        if seen is None: seen = set()
        for c in self.children(nid):
            if c not in seen:
                seen.add(c)
                self.descendants(c, seen)
        return seen

    def is_dag(self):
        # Kahn
        indeg = {n: 0 for n in self.nodes}
        for e in self.edges:
            indeg[e.target] += 1
        from collections import deque
        q = deque([n for n in self.nodes if indeg[n] == 0])
        cnt = 0
        while q:
            u = q.popleft()
            cnt += 1
            for c in self.children(u):
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        return cnt == len(self.nodes)

    def to_adjacency(self):
        ids = list(self.nodes.keys())
        idx = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)
        M = [[0.0]*n for _ in range(n)]
        for e in self.edges:
            M[idx[e.source]][idx[e.target]] = e.weight
        return ids, M

    def to_dict(self):
        return {"nodes": {k: v.to_dict() for k, v in self.nodes.items()},
                "edges": [e.to_dict() for e in self.edges]}

    @classmethod
    def from_dict(cls, d):
        g = cls()
        for k, v in d.get("nodes", {}).items():
            g.add_node(Node.from_dict(v))
        for e in d.get("edges", []):
            g.add_edge(Edge.from_dict(e))
        return g

    def structural_hamming_distance(self, other):
        """SHD between two graphs (same node set assumed)."""
        # edges as set of (src,tgt)
        a = set((e.source, e.target) for e in self.edges)
        b = set((e.source, e.target) for e in other.edges)
        # count missing/extra edges (undirected orientation errors count 2? standard: 1 per edge diff + 2 for reversed)
        shd = 0
        # For each unordered pair
        pairs = set()
        for n in self.nodes:
            for m in self.nodes:
                if n < m:
                    pairs.add((n, m))
        for (u, v) in pairs:
            e1_uv = (u, v) in a
            e1_vu = (v, u) in a
            e2_uv = (u, v) in b
            e2_vu = (v, u) in b
            if (e1_uv, e1_vu) != (e2_uv, e2_vu):
                # reversed counts 2, else 1
                if e1_uv and e2_vu or e1_vu and e2_uv:
                    shd += 2
                else:
                    shd += 1
        return shd

    def induced_subgraph(self, node_ids):
        g = CausalGraph()
        for nid in node_ids:
            if nid in self.nodes:
                g.add_node(self.nodes[nid])
        for e in self.edges:
            if e.source in g.nodes and e.target in g.nodes:
                g.add_edge(e)
        return g
