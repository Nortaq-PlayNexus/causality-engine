"""Configurable synthetic ecosystems with known ground-truth causal structures."""
import random, math
from typing import Dict, List, Any
from .causal_graph import CausalGraph, Node, Edge

class Ecosystem:
    def __init__(self, name="ecosystem", seed=0, n_species=5):
        self.name = name
        self.rng = random.Random(seed)
        self.n_species = n_species
        self.species = [f"S{i}" for i in range(n_species)]
        self.graph = self._build_graph()
        self.state = {s: self.rng.uniform(10, 50) for s in self.species}
        self.state["resource"] = 100.0
        self.state["temperature"] = 20.0
        self.state["predation_pressure"] = 1.0

    def _build_graph(self):
        g = CausalGraph()
        for s in self.species:
            g.add_node(Node(id=s, name=s))
        for extra in ["resource", "temperature", "predation_pressure"]:
            g.add_node(Node(id=extra, name=extra))
        # temperature -> resource, resource -> species, predation -> species
        g.add_edge(Edge("temperature", "resource", 0.3))
        g.add_edge(Edge("resource", self.species[0], 0.4))
        for i in range(1, len(self.species)):
            g.add_edge(Edge(self.species[i-1], self.species[i], 0.2))
            g.add_edge(Edge("resource", self.species[i], 0.15))
        g.add_edge(Edge("predation_pressure", self.species[-1], -0.25))
        return g

    def step(self, interventions=None):
        interventions = interventions or {}
        rng = self.rng
        # update temperature
        if "temperature" not in interventions:
            self.state["temperature"] += rng.gauss(0, 0.5)
        else:
            self.state["temperature"] = interventions["temperature"]
        if "resource" not in interventions:
            self.state["resource"] = max(0, self.state["resource"] + 0.3*self.state["temperature"] - 0.1*sum(self.state[s] for s in self.species) + rng.gauss(0, 2))
        else:
            self.state["resource"] = interventions["resource"]
        for s in self.species:
            if s in interventions:
                self.state[s] = interventions[s]
                continue
            parents_contrib = 0.4*self.state["resource"]/max(1,len(self.species))
            if s != self.species[0]:
                parents_contrib += 0.2*self.state[self.species[self.species.index(s)-1]]
            self.state[s] = max(0, self.state[s] + 0.1*parents_contrib - 0.05*self.state[s] + ( -0.25*self.state["predation_pressure"] if s==self.species[-1] else 0) + rng.gauss(0,1))
        if "predation_pressure" in interventions:
            self.state["predation_pressure"] = interventions["predation_pressure"]
        return dict(self.state)

    def run(self, T=50, interventions=None):
        interventions = interventions or {}
        out = []
        for t in range(T):
            iv = interventions.get(t, {})
            out.append(self.step(iv))
        return out

    def ground_truth_graph(self):
        return self.graph


class PopulationDynamics:
    def __init__(self, seed=0, n=4):
        self.rng = random.Random(seed)
        self.n = n
        self.names = [f"P{i}" for i in range(n)]
        self.graph = CausalGraph()
        for nm in self.names:
            self.graph.add_node(Node(id=nm))
        self.graph.add_node(Node(id="birth_rate"))
        self.graph.add_node(Node(id="death_rate"))
        self.graph.add_node(Node(id="carrying_capacity"))
        self.graph.add_edge(Edge("birth_rate", self.names[0]))
        self.graph.add_edge(Edge("carrying_capacity", self.names[0]))
        for i in range(1, n):
            self.graph.add_edge(Edge(self.names[i-1], self.names[i]))
        self.graph.add_edge(Edge("death_rate", self.names[-1]))
        self.state = {nm: self.rng.uniform(5,30) for nm in self.names}
        self.state.update({"birth_rate":1.2,"death_rate":0.8,"carrying_capacity":100.0})

    def step(self, interventions=None):
        interventions = interventions or {}
        for k in ["birth_rate","death_rate","carrying_capacity"]:
            if k in interventions: self.state[k]=interventions[k]
            else: self.state[k] += self.rng.gauss(0,0.05)
        for i,nm in enumerate(self.names):
            if nm in interventions:
                self.state[nm]=interventions[nm]; continue
            growth = self.state["birth_rate"]*self.state[nm]*(1-self.state[nm]/max(1,self.state["carrying_capacity"]))
            if i>0: growth += 0.1*self.state[self.names[i-1]]
            self.state[nm] = max(0, self.state[nm]+growth-0.05*self.state["death_rate"]*self.state[nm]+self.rng.gauss(0,0.5))
        return dict(self.state)

    def run(self, T=50, interventions=None):
        interventions=interventions or {}
        return [self.step(interventions.get(t,{})) for t in range(T)]


class ResourceSystem:
    def __init__(self, seed=0):
        self.rng=random.Random(seed)
        self.graph=CausalGraph()
        for n in ["extraction","regeneration","stock","price","demand"]:
            self.graph.add_node(Node(id=n))
        self.graph.add_edge(Edge("extraction","stock",-1.0))
        self.graph.add_edge(Edge("regeneration","stock",1.0))
        self.graph.add_edge(Edge("stock","price",-0.5))
        self.graph.add_edge(Edge("price","demand",-0.3))
        self.graph.add_edge(Edge("demand","extraction",0.4))
        self.state={"extraction":5.0,"regeneration":3.0,"stock":50.0,"price":10.0,"demand":20.0}

    def step(self, interventions=None):
        interventions=interventions or {}
        for k in list(self.state.keys()):
            if k in interventions:
                self.state[k]=interventions[k]; continue
        self.state["stock"]=max(0,self.state["stock"]-self.state["extraction"]+self.state["regeneration"]+self.rng.gauss(0,1))
        self.state["price"]=max(0.1,100.0/max(1,self.state["stock"])+self.rng.gauss(0,0.5))
        self.state["demand"]=max(0,30-0.3*self.state["price"]+self.rng.gauss(0,1))
        self.state["extraction"]=max(0,0.4*self.state["demand"]+self.rng.gauss(0,0.5))
        self.state["regeneration"]=max(0,3+0.1*self.state["stock"]+self.rng.gauss(0,0.3))
        return dict(self.state)

    def run(self, T=50, interventions=None):
        interventions=interventions or {}
        return [self.step(interventions.get(t,{})) for t in range(T)]


class PhysicalModel:
    """Simple physical system: force, mass, acceleration, friction."""
    def __init__(self, seed=0):
        self.rng=random.Random(seed)
        self.graph=CausalGraph()
        for n in ["force","mass","friction","acceleration","velocity","position"]:
            self.graph.add_node(Node(id=n))
        self.graph.add_edge(Edge("force","acceleration",1.0))
        self.graph.add_edge(Edge("mass","acceleration",-1.0))
        self.graph.add_edge(Edge("friction","acceleration",-1.0))
        self.graph.add_edge(Edge("acceleration","velocity",1.0))
        self.graph.add_edge(Edge("velocity","position",1.0))
        self.state={"force":10.0,"mass":2.0,"friction":0.5,"acceleration":0.0,"velocity":0.0,"position":0.0}

    def step(self, interventions=None):
        interventions=interventions or {}
        for k in list(self.state.keys()):
            if k in interventions:
                self.state[k]=interventions[k]; continue
        self.state["acceleration"]=(self.state["force"]-self.state["friction"]*self.state["velocity"])/max(0.1,self.state["mass"])+self.rng.gauss(0,0.1)
        self.state["velocity"]+=self.state["acceleration"]+self.rng.gauss(0,0.05)
        self.state["position"]+=self.state["velocity"]+self.rng.gauss(0,0.05)
        return dict(self.state)

    def run(self, T=50, interventions=None):
        interventions=interventions or {}
        return [self.step(interventions.get(t,{})) for t in range(T)]


class ArtificialEconomy:
    def __init__(self, seed=0, n_agents=10):
        self.rng=random.Random(seed)
        self.n=n_agents
        self.graph=CausalGraph()
        for n in ["supply","demand","price","wage","employment","gdp"]:
            self.graph.add_node(Node(id=n))
        self.graph.add_edge(Edge("supply","price",-0.4))
        self.graph.add_edge(Edge("demand","price",0.4))
        self.graph.add_edge(Edge("price","demand",-0.2))
        self.graph.add_edge(Edge("wage","demand",0.3))
        self.graph.add_edge(Edge("employment","gdp",0.5))
        self.graph.add_edge(Edge("price","employment",-0.1))
        self.state={"supply":50.0,"demand":45.0,"price":10.0,"wage":5.0,"employment":0.8,"gdp":100.0}

    def step(self, interventions=None):
        interventions=interventions or {}
        for k in list(self.state.keys()):
            if k in interventions:
                self.state[k]=interventions[k]; continue
        self.state["price"]=max(0.1,10+0.4*(self.state["demand"]-self.state["supply"])+self.rng.gauss(0,0.5))
        self.state["demand"]=max(0,45-0.2*self.state["price"]+0.3*self.state["wage"]+self.rng.gauss(0,1))
        self.state["supply"]=max(0,50-0.1*self.state["price"]+self.rng.gauss(0,1))
        self.state["employment"]=max(0,min(1,0.8-0.05*self.state["price"]+self.rng.gauss(0,0.02)))
        self.state["gdp"]=max(0,100*self.state["employment"]+self.rng.gauss(0,2))
        self.state["wage"]=max(0.1,5+0.1*self.state["gdp"]/100+self.rng.gauss(0,0.2))
        return dict(self.state)

    def run(self, T=50, interventions=None):
        interventions=interventions or {}
        return [self.step(interventions.get(t,{})) for t in range(T)]
