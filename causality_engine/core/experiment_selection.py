"""Automated experiment selection (active learning for causal discovery)."""
import random, math
from typing import List, Dict, Any

def select_next_experiment(candidate_interventions, current_graph, uncertainty_fn, budget=1):
    # rank by expected information gain approximated by uncertainty
    ranked = sorted(candidate_interventions, key=lambda iv: uncertainty_fn(iv, current_graph), reverse=True)
    return ranked[:budget]

def candidate_interventions_from_nodes(nodes):
    out=[]
    for n in nodes:
        out.append({"target":n,"value":1.0})
        out.append({"target":n,"value":0.0})
    return out

def uncertainty_proxy(intervention, graph):
    # higher if node has many descendants (intervening reveals more)
    # find node
    tgt=intervention["target"]
    desc=graph.descendants(tgt) if hasattr(graph,'descendants') else set()
    return len(desc)+1

def expected_info_gain(graph_before, graph_after, data):
    # proxy: reduction in number of ambiguous edges
    return abs(len(graph_before.edges)-len(graph_after.edges))
