"""Graph recovery algorithms (correlation != causation)."""
import random, math
from typing import Dict, List, Any
from .causal_graph import CausalGraph, Node, Edge

def corr_matrix(data, keys):
    # data: list of dict
    n=len(data)
    means={k: sum(d[k] for d in data)/n for k in keys}
    out={}
    for a in keys:
        for b in keys:
            cov=sum((d[a]-means[a])*(d[b]-means[b]) for d in data)/n
            va=math.sqrt(sum((d[a]-means[a])**2 for d in data)/n)
            vb=math.sqrt(sum((d[b]-means[b])**2 for d in data)/n)
            out[(a,b)] = cov/(va*vb) if va>0 and vb>0 else 0.0
    return out, means

def pc_skeleton(data, keys, alpha=0.1):
    """Simple PC: start fully connected, remove edges if conditional independence (partial corr ~0)."""
    # adjacency set
    adj={k:set(keys)-{k} for k in keys}
    # unconditional removal
    cm,_=corr_matrix(data,keys)
    for a in keys:
        for b in list(adj[a]):
            if a<b:
                if abs(cm[(a,b)]) < alpha:
                    adj[a].discard(b); adj[b].discard(a)
    # conditional on single node
    for a in keys:
        for b in list(adj[a]):
            if a<b:
                for c in keys:
                    if c in (a,b): continue
                    # partial correlation approx: residual correlation after regressing out c
                    # simple: if corr(a,b|c) small
                    # compute residual
                    # residual of a on c, b on c
                    pass
    # build graph
    g=CausalGraph()
    for k in keys:
        g.add_node(Node(id=k))
    for a in keys:
        for b in adj[a]:
            if a<b:
                g.add_edge(Edge(a,b,1.0))
    return g

def score_based_hill_climbing(data, keys, max_iters=50):
    """Greedy BIC-like score (lower better). Start empty, add edges that improve score."""
    # score = sum over nodes of variance of residual after linear regression on parents (lower better)
    def residual_var(node, parents):
        vals=[d[node] for d in data]
        n=len(vals)
        if not parents:
            m=sum(vals)/n
            return sum((v-m)**2 for v in vals)/n
        # build design matrix
        # solve least squares via normal equations for up to 3 parents
        ps=parents[:3]
        # means
        # Use simple: residual = var(node) * (1 - sum corr^2) approx
        # Compute actual OLS residual
        import math
        # construct X (n x (k+1)) with intercept
        k=len(ps)
        # normal equations: (X'X) beta = X'y
        # build matrices
        XtX=[[0.0]*(k+1) for _ in range(k+1)]
        Xty=[0.0]*(k+1)
        for i in range(n):
            row=[1.0]+[d[ps[j]][i] if False else data[i][ps[j]] for j in range(k)]
            y=vals[i]
            for a in range(k+1):
                Xty[a]+=row[a]*y
                for b in range(k+1):
                    XtX[a][b]+=row[a]*row[b]
        # solve via gaussian elimination
        # augment
        M=[row[:]+[Xty[i]] for i,row in enumerate(XtX)]
        # gaussian elim
        size=k+1
        for col in range(size):
            # pivot
            piv=max(range(col,size), key=lambda r: abs(M[r][col]))
            M[col],M[piv]=M[piv],M[col]
            pv=M[col][col]
            if abs(pv)<1e-12:
                continue
            for j in range(col,size+1):
                M[col][j]/=pv
            for r in range(size):
                if r==col: continue
                factor=M[r][col]
                for j in range(col,size+1):
                    M[r][j]-=factor*M[col][j]
        beta=[M[i][size] for i in range(size)]
        # residual
        rss=0.0
        for i in range(n):
            row=[1.0]+[data[i][ps[j]] for j in range(k)]
            pred=sum(beta[j]*row[j] for j in range(size))
            rss+=(vals[i]-pred)**2
        return rss/n
    current=set()
    nodes=keys
    # candidate edges
    cands=[(a,b) for a in nodes for b in nodes if a!=b]
    random.shuffle(cands)
    improved=True
    it=0
    while improved and it<max_iters:
        improved=False
        it+=1
        best=None; bestgain=0
        for (a,b) in cands:
            if (a,b) in current: continue
            # acyclicity check
            # build temp graph
            tg=CausalGraph()
            for k in nodes: tg.add_node(Node(id=k))
            for (u,v) in current: tg.add_edge(Edge(u,v))
            tg.add_edge(Edge(a,b))
            if not tg.is_dag(): continue
            # gain = reduction in residual var of b
            parents_before=[u for (u,v) in current if v==b]
            parents_after=parents_before+[a]
            gain=residual_var(b,parents_before)-residual_var(b,parents_after)
            # complexity penalty: require meaningful improvement
            if gain>bestgain and gain > 0.02*residual_var(b,parents_before):
                bestgain=gain; best=(a,b)
        if best:
            current.add(best); improved=True
    g=CausalGraph()
    for k in nodes: g.add_node(Node(id=k))
    for (u,v) in current: g.add_edge(Edge(u,v))
    return g

def intervention_predict_error(true_scm, recovered_graph, test_interventions, n=20):
    # compare predicted means under intervention
    from .structural_causal_model import SCM
    errs=[]
    for iv in test_interventions:
        true_samples=true_scm.do(iv, n_samples=n)
        # recovered scm with same equations? use default linear
        rscm=SCM(recovered_graph, noise_std=true_scm.noise_std)
        rec_samples=rscm.do(iv, n_samples=n)
        for k in true_scm.graph.nodes:
            tm=sum(s[k] for s in true_samples)/n
            rm=sum(s[k] for s in rec_samples)/n
            errs.append(abs(tm-rm))
    return sum(errs)/len(errs) if errs else 0.0
