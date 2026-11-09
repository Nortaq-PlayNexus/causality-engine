"""Time-series analysis for causal discovery."""
import math, random
from typing import Dict, List, Any

def lag_correlation(series_x, series_y, lag=0):
    # Pearson corr of x[t] with y[t+lag]
    n = min(len(series_x), len(series_y))
    if lag>=0:
        xs = series_x[:n-lag] if lag>0 else series_x
        ys = series_y[lag:] if lag>0 else series_y
    else:
        xs = series_x[-lag:]
        ys = series_y[:n+lag]
    m = min(len(xs), len(ys))
    if m < 3: return 0.0
    xs=xs[:m]; ys=ys[:m]
    mx=sum(xs)/m; my=sum(ys)/m
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    vx=math.sqrt(sum((x-mx)**2 for x in xs))
    vy=math.sqrt(sum((y-my)**2 for y in ys))
    if vx==0 or vy==0: return 0.0
    return cov/(vx*vy)

def granger_like(x, y, max_lag=3):
    # simple: best lag correlation magnitude
    best=0.0; best_lag=0
    for l in range(1, max_lag+1):
        c=abs(lag_correlation(x,y,l))
        if c>best: best=c; best_lag=l
    return best, best_lag

def time_series_features(data: List[Dict[str,float]]):
    # data list of dicts per timestep
    keys=list(data[0].keys()) if data else []
    out={}
    for k in keys:
        s=[d[k] for d in data]
        out[k]={"mean":sum(s)/len(s),"std":(sum((x-sum(s)/len(s))**2 for x in s)/len(s))**0.5,
                "trend": (s[-1]-s[0]) if len(s)>1 else 0.0}
    return out
