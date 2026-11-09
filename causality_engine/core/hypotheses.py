"""Alternative hypotheses: competing causal graphs."""
import copy, random, math
from typing import List, Dict, Any
from .causal_graph import CausalGraph

class HypothesisSet:
    def __init__(self):
        self.hypotheses: List[Dict[str,Any]] = []

    def add(self, graph: CausalGraph, name: str, prior=0.5):
        self.hypotheses.append({"name":name,"graph":graph,"prior":prior,"posterior":prior,"score":0.0})

    def update_scores(self, data_score_fn):
        for h in self.hypotheses:
            h["score"]=data_score_fn(h["graph"])
            # bayes-ish
        total=sum(math.exp(h["score"]) for h in self.hypotheses) if False else None
        import math
        # softmax over scores
        mx=max(h["score"] for h in self.hypotheses)
        exps=[math.exp(h["score"]-mx) for h in self.hypotheses]
        Z=sum(exps)
        for h,e in zip(self.hypotheses,exps):
            h["posterior"]=h["prior"]*e/Z
        # normalize
        s=sum(h["posterior"] for h in self.hypotheses)
        for h in self.hypotheses:
            h["posterior"]/=s

    def best(self):
        return max(self.hypotheses, key=lambda h: h["posterior"]) if self.hypotheses else None

    def to_dict(self):
        return [{"name":h["name"],"prior":h["prior"],"posterior":h["posterior"],"score":h["score"],
                 "graph":h["graph"].to_dict()} for h in self.hypotheses]
