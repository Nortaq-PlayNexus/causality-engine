"""Hidden / latent variables and confounding."""
import random
from typing import Dict, List, Any
from .causal_graph import CausalGraph, Node, Edge

def add_latent_confounder(graph: CausalGraph, targets: List[str], name="U"):
    g = graph
    if name not in g.nodes:
        g.add_node(Node(id=name, name=name, observed=False, latent=True))
    for t in targets:
        g.add_edge(Edge(name, t, 0.4, "confounding"))
    return g

def latent_variables(graph):
    return [n for n in graph.nodes.values() if n.latent]

class HiddenVariableModel:
    def __init__(self, observed_graph, seed=0):
        self.rng=random.Random(seed)
        self.observed_graph=observed_graph
        self.latents=[]
    def inject(self, targets, name=None):
        name = name or f"U{len(self.latents)}"
        self.latents.append(name)
        add_latent_confounder(self.observed_graph, targets, name)
        return name
