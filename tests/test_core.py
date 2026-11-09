import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from causality_engine.core import CausalGraph, Node, Edge, SCM, Ecosystem

def test_dag():
    g=CausalGraph()
    g.add_node(Node(id="A")); g.add_node(Node(id="B"))
    g.add_edge(Edge("A","B"))
    assert g.is_dag()
    # cycle
    g2=CausalGraph()
    g2.add_node(Node(id="A")); g2.add_node(Node(id="B"))
    g2.add_edge(Edge("A","B")); g2.add_edge(Edge("B","A"))
    assert not g2.is_dag()

def test_scm_sample():
    g=CausalGraph()
    g.add_node(Node(id="X")); g.add_node(Node(id="Y"))
    g.add_edge(Edge("X","Y",1.0))
    scm=SCM(g, noise_std=0.1, seed=1)
    s=scm.sample(n_samples=5)
    assert len(s)==5
    assert "X" in s[0] and "Y" in s[0]

def test_intervention():
    g=CausalGraph()
    g.add_node(Node(id="X")); g.add_node(Node(id="Y"))
    g.add_edge(Edge("X","Y",1.0))
    scm=SCM(g, noise_std=0.0, seed=1)
    s=scm.do({"X":5.0}, n_samples=3)
    for row in s:
        assert row["X"]==5.0

def test_ecosystem():
    eco=Ecosystem(seed=1)
    d=eco.run(T=5)
    assert len(d)==5
    assert "resource" in d[0]

def test_shd():
    g1=CausalGraph()
    for i in range(3): g1.add_node(Node(id=f"X{i}"))
    g1.add_edge(Edge("X0","X1"))
    g2=CausalGraph()
    for i in range(3): g2.add_node(Node(id=f"X{i}"))
    g2.add_edge(Edge("X1","X0"))
    assert g1.structural_hamming_distance(g2)==2

def test_metrics():
    from causality_engine.core.metrics import false_discovery_rate, true_positive_rate
    from causality_engine.core.causal_graph import CausalGraph, Node, Edge
    t=CausalGraph(); p=CausalGraph()
    for i in range(3):
        t.add_node(Node(id=f"X{i}")); p.add_node(Node(id=f"X{i}"))
    t.add_edge(Edge("X0","X1"))
    p.add_edge(Edge("X0","X1")); p.add_edge(Edge("X0","X2"))
    assert false_discovery_rate(p,t)==0.5
    assert true_positive_rate(p,t)==1.0

def test_negative_control():
    from causality_engine.core.graph_recovery import score_based_hill_climbing
    import random
    rng=random.Random(1)
    keys=["R0","R1","R2","R3"]
    data=[{k:rng.gauss(0,1) for k in keys} for _ in range(200)]
    g=score_based_hill_climbing(data,keys)
    assert len(g.edges)==0

def test_benchmark_runs():
    from causality_engine.benchmark.runner import run_all
    r=run_all()
    assert "graph_recovery" in r
    assert "negative_controls" in r
