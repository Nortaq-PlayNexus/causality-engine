"""Metrics for causal discovery benchmarks."""
import math
from typing import Dict, List

def shd(predicted_graph, true_graph):
    return true_graph.structural_hamming_distance(predicted_graph)

def false_discovery_rate(predicted_graph, true_graph):
    true_edges=set((e.source,e.target) for e in true_graph.edges)
    pred_edges=set((e.source,e.target) for e in predicted_graph.edges)
    if not pred_edges: return 0.0
    fp=len(pred_edges-true_edges)
    return fp/len(pred_edges)

def true_positive_rate(predicted_graph, true_graph):
    true_edges=set((e.source,e.target) for e in true_graph.edges)
    pred_edges=set((e.source,e.target) for e in predicted_graph.edges)
    if not true_edges: return 1.0
    tp=len(pred_edges & true_edges)
    return tp/len(true_edges)

def noise_robustness(scores_by_noise):
    # scores_by_noise: dict noise->metric; return slope (degradation)
    noises=sorted(scores_by_noise.keys())
    if len(noises)<2: return 0.0
    return scores_by_noise[noises[-1]]-scores_by_noise[noises[0]]

def generalization_score(train_metric, test_metric):
    return test_metric/train_metric if train_metric>0 else 0.0
