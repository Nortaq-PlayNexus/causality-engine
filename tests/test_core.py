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
