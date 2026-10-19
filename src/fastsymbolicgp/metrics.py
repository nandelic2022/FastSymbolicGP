from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, log_loss, balanced_accuracy_score, accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

@dataclass(frozen=True)
class Metric:
    name: str
    function: Callable
    direction: str = "minimize"
    fused_compatible: bool = False

_REG: dict[str, Metric] = {}

def register_metric(name, function, *, direction="minimize", fused_compatible=False, overwrite=False):
    name=str(name).lower()
    if name in _REG and not overwrite: raise ValueError(f"Metric {name!r} already exists")
    if direction not in {"minimize","maximize"}: raise ValueError("direction must be minimize or maximize")
    m=Metric(name,function,direction,bool(fused_compatible)); _REG[name]=m; return m

def make_metric(function, name, **kwargs): return register_metric(name,function,**kwargs)
def get_metric(name):
    if callable(name): return Metric(getattr(name,"__name__","custom"),name,"minimize",False)
    key=str(name).lower()
    if key not in _REG: raise ValueError(f"Unknown metric {name!r}. Available: {sorted(_REG)}")
    return _REG[key]

def _avg(v,w=None): return float(np.mean(v) if w is None else np.average(v,weights=w))

def regression_loss(y,pred,metric="rmse",sample_weight=None, *, quantile=0.5, huber_delta=1.0):
    if callable(metric):
        try: return float(metric(y,pred,sample_weight=sample_weight))
        except TypeError: return float(metric(y,pred))
    m=str(metric).lower(); y=np.asarray(y,float); pred=np.asarray(pred,float); r=pred-y
    if m in {"mse","mean_squared_error"}: return float(mean_squared_error(y,pred,sample_weight=sample_weight))
    if m in {"rmse","root_mean_squared_error"}: return float(np.sqrt(mean_squared_error(y,pred,sample_weight=sample_weight)))
    if m in {"mae","mean_absolute_error"}: return float(mean_absolute_error(y,pred,sample_weight=sample_weight))
    if m in {"medae","median_absolute_error"}: return float(median_absolute_error(y,pred))
    if m in {"r2","r2_score"}: return 1.0-float(r2_score(y,pred,sample_weight=sample_weight))
    if m=="huber":
        a=np.abs(r); d=float(huber_delta); loss=np.where(a<=d,0.5*r*r,d*(a-0.5*d)); return _avg(loss,sample_weight)
    if m in {"logcosh","log_cosh"}: return _avg(np.logaddexp(r,-r)-np.log(2.0),sample_weight)
    if m in {"mape","mean_absolute_percentage_error"}: return _avg(np.abs(r)/np.maximum(np.abs(y),1e-12),sample_weight)
    if m=="smape": return _avg(2*np.abs(r)/np.maximum(np.abs(y)+np.abs(pred),1e-12),sample_weight)
    if m in {"quantile","pinball"}:
        q=float(quantile); e=y-pred; return _avg(np.maximum(q*e,(q-1)*e),sample_weight)
    if m=="rmsle":
        yy=np.maximum(y,0); pp=np.maximum(pred,0); return float(np.sqrt(_avg((np.log1p(pp)-np.log1p(yy))**2,sample_weight)))
    custom=get_metric(m); value=float(custom.function(y,pred,sample_weight)); return -value if custom.direction=="maximize" else value

def classification_loss(y, proba, metric="log_loss", sample_weight=None, threshold=0.5):
    if callable(metric):
        try: return float(metric(y,proba,sample_weight=sample_weight))
        except TypeError: return float(metric(y,proba))
    m=str(metric).lower(); y=np.asarray(y); proba=np.asarray(proba,float); pred=(proba>=threshold).astype(int)
    if m in {"log_loss","logloss"}:
        p=np.clip(proba,1e-12,1-1e-12); return float(log_loss(y,np.c_[1-p,p],sample_weight=sample_weight,labels=[0,1]))
    if m in {"balanced_accuracy","bacc"}: return 1-float(balanced_accuracy_score(y,pred,sample_weight=sample_weight))
    if m=="accuracy": return 1-float(accuracy_score(y,pred,sample_weight=sample_weight))
    if m in {"f1","f1_score"}: return 1-float(f1_score(y,pred,sample_weight=sample_weight,zero_division=0))
    if m in {"mcc","matthews"}: return 1-(float(matthews_corrcoef(y,pred,sample_weight=sample_weight))+1)/2
    if m in {"roc_auc","auc"}:
        try: return 1-float(roc_auc_score(y,proba,sample_weight=sample_weight))
        except ValueError: return 1.0
    if m in {"average_precision","ap"}:
        try: return 1-float(average_precision_score(y,proba,sample_weight=sample_weight))
        except ValueError: return 1.0
    custom=get_metric(m); value=float(custom.function(y,proba,sample_weight)); return -value if custom.direction=="maximize" else value

def fit_linear_scaler(raw,y,sample_weight=None):
    x=np.asarray(raw,float); y=np.asarray(y,float); w=np.ones_like(y) if sample_weight is None else np.asarray(sample_weight,float)
    mx=np.average(x,weights=w); my=np.average(y,weights=w); var=np.average((x-mx)**2,weights=w)
    if var<1e-16: return 0.0,float(my)
    cov=np.average((x-mx)*(y-my),weights=w); a=cov/var; return float(a),float(my-a*mx)


def fit_logistic_scaler_approx(raw, y, sample_weight=None, regularization=1e-4):
    x=np.asarray(raw,float);yy=np.asarray(y,float);w=np.ones_like(yy) if sample_weight is None else np.asarray(sample_weight,float)
    sw=float(np.sum(w))
    if sw<=0:return 1.0,0.0
    prior=float(np.clip(np.sum(w*yy)/sw,1e-8,1-1e-8));xm=float(np.sum(w*x)/sw);c=x-xm;var=float(np.sum(w*c*c))
    if var<=1e-18:return 0.0,float(np.log(prior/(1-prior)))
    cov=float(np.sum(w*c*(yy-prior)));curv=max(prior*(1-prior),1e-8)*var+float(regularization)
    a=float(np.clip(cov/curv,-1e6,1e6));b=float(np.clip(np.log(prior/(1-prior))-a*xm,-1e6,1e6));return a,b

def fit_logistic_scaler(raw,y,sample_weight=None,max_iter=25,reg=1e-4,regularization=None):
    if regularization is not None: reg=float(regularization)
    x=np.asarray(raw,float); yy=np.asarray(y,float); w=np.ones_like(yy) if sample_weight is None else np.asarray(sample_weight,float)
    xm=np.average(x,weights=w); xs=np.sqrt(np.average((x-xm)**2,weights=w))+1e-12; z=(x-xm)/xs
    prior=np.clip(np.average(yy,weights=w),1e-6,1-1e-6); a=0.0; b=float(np.log(prior/(1-prior)))
    for _ in range(max_iter):
        t=np.clip(a*z+b,-35,35); p=1/(1+np.exp(-t)); q=np.maximum(p*(1-p),1e-9)
        ga=np.sum(w*(p-yy)*z)+reg*a; gb=np.sum(w*(p-yy)); haa=np.sum(w*q*z*z)+reg; hab=np.sum(w*q*z); hbb=np.sum(w*q)+1e-12
        det=haa*hbb-hab*hab
        if abs(det)<1e-18: break
        da=(hbb*ga-hab*gb)/det; db=(-hab*ga+haa*gb)/det; a-=np.clip(da,-5,5); b-=np.clip(db,-5,5)
        if abs(da)+abs(db)<1e-8: break
    return float(a/xs),float(b-a*xm/xs)

def sigmoid(x): x=np.clip(np.asarray(x,float),-35,35); return 1/(1+np.exp(-x))

for _name in ["mse","rmse","mae","r2","huber","medae","logcosh","mape","smape","quantile","rmsle"]:
    if _name not in _REG: register_metric(_name,lambda y,p,w=None: 0.0,fused_compatible=_name in {"mse","rmse","mae","r2","huber"})
