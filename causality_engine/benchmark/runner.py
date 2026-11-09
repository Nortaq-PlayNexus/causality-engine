"""Benchmark suite."""
import random
from typing import Dict, Any
from ..core.causal_graph import CausalGraph, Node, Edge
from ..core.structural_causal_model import SCM
from ..core.ecosystems import Ecosystem
from ..core.graph_recovery import pc_skeleton, score_based_hill_climbing
from ..core.metrics import shd, false_discovery_rate, true_positive_rate
from ..core.hidden_variables import add_latent_confounder

def make_ground_truth(seed=0, kind="ecosystem"):
    if kind=="ecosystem":
        eco=Ecosystem(seed=seed)
        return eco.graph, eco
    # chain
    g=CausalGraph()
    for i in range(5):
        g.add_node(Node(id=f"X{i}"))
    for i in range(4):
        g.add_edge(Edge(f"X{i}", f"X{i+1}", 0.5))
    return g, None

def run_graph_recovery_benchmark(seeds=10, n_samples=200):
    results={"pc":{"shd":[],"fdr":[],"tpr":[]},"hc":{"shd":[],"fdr":[],"tpr":[]}}
    for s in range(seeds):
        g,_=make_ground_truth(seed=s)
        scm=SCM(g, noise_std=0.1, seed=s)
        data=scm.sample(n_samples=n_samples, seed=s)
        keys=list(g.nodes.keys())
        # flatten
        rows=[{k:d[k] for k in keys} for d in data]
        # pc
        gpc=pc_skeleton(rows, keys, alpha=0.15)
        gpc_nodes=set(gpc.nodes.keys())
        # ensure same nodes
        ghc=score_based_hill_climbing(rows, keys)
        for name,gg in [("pc",gpc),("hc",ghc)]:
            results[name]["shd"].append(shd(gg,g))
            results[name]["fdr"].append(false_discovery_rate(gg,g))
            results[name]["tpr"].append(true_positive_rate(gg,g))
    out={}
    for name in ["pc","hc"]:
        out[name]={
            "shd_mean": round(sum(results[name]["shd"])/len(results[name]["shd"]),3),
            "fdr_mean": round(sum(results[name]["fdr"])/len(results[name]["fdr"]),3),
            "tpr_mean": round(sum(results[name]["tpr"])/len(results[name]["tpr"]),3),
        }
    return out

def run_intervention_prediction_benchmark(seeds=5, n_samples=100):
    from ..core.graph_recovery import intervention_predict_error
    errs=[]
    for s in range(seeds):
        g,_=make_ground_truth(seed=s)
        scm=SCM(g, noise_std=0.1, seed=s)
        data=scm.sample(n_samples=n_samples, seed=s)
        keys=list(g.nodes.keys())
        rows=[{k:d[k] for k in keys} for d in data]
        gr=score_based_hill_climbing(rows, keys)
        ivs=[{keys[0]:2.0},{keys[2]:-1.0} if len(keys)>2 else {keys[0]:0.0}]
        e=intervention_predict_error(scm, gr, ivs, n=20)
        errs.append(e)
    return {"intervention_error_mean": round(sum(errs)/len(errs),3)}

def run_noise_robustness_benchmark(seeds=5):
    out={}
    for noise in [0.05,0.2,0.5]:
        shds=[]
        for s in range(seeds):
            g,_=make_ground_truth(seed=s)
            scm=SCM(g, noise_std=noise, seed=s)
            data=scm.sample(n_samples=200, seed=s)
            keys=list(g.nodes.keys())
            rows=[{k:d[k] for k in keys} for d in data]
            gr=score_based_hill_climbing(rows, keys)
            shds.append(shd(gr,g))
        out[noise]=round(sum(shds)/len(shds),3)
    return {"shd_by_noise": out}

def run_negative_controls(seeds=5):
    # Random data with no causal structure: expect high FDR / no edges
    import random
    res=[]
    for s in range(seeds):
        rng=random.Random(s)
        keys=[f"R{i}" for i in range(4)]
        data=[{k:rng.gauss(0,1) for k in keys} for _ in range(200)]
        g=CausalGraph()
        for k in keys: g.add_node(Node(id=k))
        gr=score_based_hill_climbing(data, keys)
        res.append(false_discovery_rate(gr, g))
    return {"negative_control_fdr_mean": round(sum(res)/len(res),3)}

def run_all():
    return {
        "graph_recovery": run_graph_recovery_benchmark(),
        "intervention_prediction": run_intervention_prediction_benchmark(),
        "noise_robustness": run_noise_robustness_benchmark(),
        "negative_controls": run_negative_controls(),
    }
