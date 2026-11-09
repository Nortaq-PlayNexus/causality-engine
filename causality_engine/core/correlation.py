"""Correlation utilities with explicit caveat: correlation != causation."""
import math
from typing import List, Dict

def pearson(x, y):
    n=min(len(x),len(y))
    if n<2: return 0.0
    x=x[:n]; y=y[:n]
    mx=sum(x)/n; my=sum(y)/n
    cov=sum((a-mx)*(b-my) for a,b in zip(x,y))
    vx=math.sqrt(sum((a-mx)**2 for a in x))
    vy=math.sqrt(sum((b-my)**2 for b in y))
    if vx==0 or vy==0: return 0.0
    return cov/(vx*vy)

def spurious_correlation_warning(r):
    if abs(r)>0.7:
        return "High correlation detected. WARNING: correlation does not imply causation. Verify with interventions."
    return "No strong correlation."

CAUSATION_REQUIRES = ["temporal precedence","covariation","elimination of confounders","interventional evidence","counterfactual support"]
