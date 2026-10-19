from __future__ import annotations
import math
import numpy as np

# Compact postfix backend for built-in primitives. Custom primitives fall back to NumPy.
OPS={
    'feature':1,'const':2,'add':10,'sub':11,'mul':12,'div':13,'max':14,'min':15,
    'sin':20,'cos':21,'tanh':22,'abs':23,'neg':24,'log':25,'sqrt':26,'exp':27,
    'inv':28,'square':29,'cube':30,'sign':31,'relu':32,'softplus':33,'log1p':34,'clip':35,
}
SUPPORTED=set(OPS)-{'feature','const'}
OP_NAMES={v:k for k,v in OPS.items()}

try:
    from numba import njit, prange, set_num_threads, get_num_threads, config as _numba_config
    NUMBA_AVAILABLE=True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE=False
    njit=prange=set_num_threads=get_num_threads=None
    _numba_config=None


def configure_numba_threads(n_jobs=-1):
    if not NUMBA_AVAILABLE: return 1
    maximum=int(getattr(_numba_config,'NUMBA_NUM_THREADS',1))
    n=maximum if n_jobs is None or int(n_jobs)<0 else max(1,min(int(n_jobs),maximum))
    try: set_num_threads(n)
    except Exception: pass
    return int(get_num_threads())


def compile_program(program):
    # Cached postfix is authoritative for flat-only evolutionary individuals.
    meta=getattr(program,"metadata",None)
    cached=None if meta is None else meta.get("_compiled_postfix")
    key=id(program.root)
    if cached is not None and (cached[0]==key or program.root is None):
        meta["compile_cache_hits"]=int(meta.get("compile_cache_hits",0))+1
        return cached[1],cached[2]
    if program.root is None:
        raise ValueError("Flat-only program is missing its compiled postfix genome")
    ops=[]; args=[]
    def walk(n):
        for c in n.children: walk(c)
        if n.kind=='feature': ops.append(OPS['feature']); args.append(float(n.value)); return
        if n.kind=='const': ops.append(OPS['const']); args.append(float(n.value)); return
        name=str(n.value)
        if name not in SUPPORTED: raise ValueError(f"Primitive {name!r} is not supported by the Numba postfix backend")
        ops.append(OPS[name]); args.append(0.0)
    walk(program.root)
    opcodes=np.asarray(ops,np.int16); arguments=np.asarray(args,np.float64)
    if meta is not None:
        meta["_compiled_postfix"]=(key,opcodes,arguments)
        meta["compile_cache_misses"]=int(meta.get("compile_cache_misses",0))+1
    return opcodes,arguments

if NUMBA_AVAILABLE:
    @njit(cache=True,inline='always')
    def _one(X,row,ops,args):
        stack=np.empty(len(ops),dtype=np.float64); sp=0
        for k in range(len(ops)):
            op=ops[k]
            if op==1:
                stack[sp]=X[row,int(args[k])]; sp+=1; continue
            if op==2:
                stack[sp]=args[k]; sp+=1; continue
            if op<20:
                b=stack[sp-1]; a=stack[sp-2]; sp-=2
                if op==10: v=a+b
                elif op==11: v=a-b
                elif op==12: v=a*b
                elif op==13: v=1.0 if abs(b)<=1e-12 else a/b
                elif op==14: v=max(a,b)
                else: v=min(a,b)
                if not math.isfinite(v): v=0.0 if math.isnan(v) else (1e6 if v>0 else -1e6)
                if v>1e6: v=1e6
                elif v<-1e6: v=-1e6
                stack[sp]=v; sp+=1; continue
            a=stack[sp-1]; sp-=1
            if op==20: v=math.sin(a)
            elif op==21: v=math.cos(a)
            elif op==22: v=math.tanh(a)
            elif op==23: v=abs(a)
            elif op==24: v=-a
            elif op==25: v=math.log(abs(a)+1e-12)
            elif op==26: v=math.sqrt(abs(a))
            elif op==27: v=math.exp(max(-30.0,min(30.0,a)))
            elif op==28: v=0.0 if abs(a)<=1e-12 else 1.0/a
            elif op==29: v=a*a
            elif op==30: v=a*a*a
            elif op==31: v=1.0 if a>0 else (-1.0 if a<0 else 0.0)
            elif op==32: v=max(a,0.0)
            elif op==33: v=math.log1p(math.exp(-abs(a)))+max(a,0.0)
            elif op==34: v=math.log1p(abs(a))
            else: v=max(-1e3,min(1e3,a))
            if not math.isfinite(v): v=0.0 if math.isnan(v) else (1e6 if v>0 else -1e6)
            if v>1e6: v=1e6
            elif v<-1e6: v=-1e6
            stack[sp]=v; sp+=1
        return stack[0]

    @njit(cache=True,parallel=True)
    def _execute(X,ops,args):
        out=np.empty(X.shape[0],dtype=np.float64)
        for i in prange(X.shape[0]): out[i]=_one(X,i,ops,args)
        return out



if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _analyze_postfix_numba(ops):
        """Return subtree starts/depths and preorder->postfix mapping."""
        n=len(ops)
        starts=np.empty(n,dtype=np.int64)
        depths=np.empty(n,dtype=np.int16)
        left=np.full(n,-1,dtype=np.int64)
        right=np.full(n,-1,dtype=np.int64)
        stack=np.empty(n,dtype=np.int64); sp=0
        for i in range(n):
            op=int(ops[i])
            if op==1 or op==2:
                starts[i]=i; depths[i]=1; stack[sp]=i; sp+=1
            elif op<20:  # binary built-ins
                r=stack[sp-1]; l=stack[sp-2]; sp-=2
                left[i]=l; right[i]=r; starts[i]=starts[l]
                depths[i]=1+max(depths[l],depths[r]); stack[sp]=i; sp+=1
            else:  # unary built-ins
                c=stack[sp-1]; sp-=1
                left[i]=c; starts[i]=starts[c]; depths[i]=1+depths[c]
                stack[sp]=i; sp+=1
        preorder=np.empty(n,dtype=np.int64)
        todo=np.empty(n,dtype=np.int64); tsp=1; todo[0]=n-1; out=0
        while tsp>0:
            tsp-=1; node=todo[tsp]; preorder[out]=node; out+=1
            r=right[node]; l=left[node]
            if r>=0:
                todo[tsp]=r; tsp+=1
            if l>=0:
                todo[tsp]=l; tsp+=1
        return starts,depths,preorder
else:
    _analyze_postfix_numba=None



def analyze_postfix(opcodes):
    """Analyze a built-in postfix genome without materializing a Python tree."""
    ops=np.ascontiguousarray(np.asarray(opcodes,dtype=np.int16))
    if ops.size==0:
        raise ValueError("Empty postfix genome")
    if NUMBA_AVAILABLE:
        return _analyze_postfix_numba(ops)
    # deterministic Python fallback
    n=len(ops); starts=np.empty(n,np.int64); depths=np.empty(n,np.int16)
    left=np.full(n,-1,np.int64); right=np.full(n,-1,np.int64); stack=[]
    for i,op0 in enumerate(ops):
        op=int(op0)
        if op in (1,2): starts[i]=i; depths[i]=1; stack.append(i)
        elif op<20:
            r=stack.pop(); l=stack.pop(); left[i]=l; right[i]=r; starts[i]=starts[l]; depths[i]=1+max(depths[l],depths[r]); stack.append(i)
        else:
            c=stack.pop(); left[i]=c; starts[i]=starts[c]; depths[i]=1+depths[c]; stack.append(i)
    preorder=[]; todo=[n-1]
    while todo:
        node=todo.pop(); preorder.append(node)
        if right[node]>=0: todo.append(int(right[node]))
        if left[node]>=0: todo.append(int(left[node]))
    return starts,depths,np.asarray(preorder,dtype=np.int64)


def node_from_postfix(opcodes,arguments):
    """Materialize a Node tree from a postfix genome (cold path only)."""
    from .program import Node
    ops=np.asarray(opcodes,dtype=np.int16); args=np.asarray(arguments,dtype=np.float64)
    stack=[]
    for op0,arg in zip(ops,args):
        op=int(op0)
        if op==OPS['feature']:
            stack.append(Node('feature',int(arg)))
        elif op==OPS['const']:
            stack.append(Node('const',float(arg)))
        else:
            name=OP_NAMES.get(op)
            if name is None: raise ValueError(f"Unknown postfix opcode {op}")
            if op<20:
                b=stack.pop(); a=stack.pop(); stack.append(Node('func',name,(a,b)))
            else:
                a=stack.pop(); stack.append(Node('func',name,(a,)))
    if len(stack)!=1: raise ValueError("Invalid postfix genome")
    return stack[0]

def execute_numba(program,X,n_jobs=-1):
    if not NUMBA_AVAILABLE: raise RuntimeError('Numba is not installed')
    ops,args=compile_program(program); configure_numba_threads(n_jobs)
    return _execute(np.ascontiguousarray(np.asarray(X,np.float64)),ops,args)


def can_numba(program):
    if not NUMBA_AVAILABLE: return False
    if getattr(program,"root",None) is None:
        return bool(getattr(program,"metadata",{}).get("_compiled_postfix") is not None)
    return all(name in SUPPORTED for name in program.root.function_names())


# ---------------------------------------------------------------------------
# V1.1.1 fused regression population scoring
# ---------------------------------------------------------------------------
FUSED_REGRESSION_METRICS = {
    "mse", "mean_squared_error",
    "rmse", "root_mean_squared_error",
    "mae", "mean_absolute_error",
    "r2", "r2_score",
    "huber",
}


def fused_regression_supported(metric, scaling="linear"):
    """Return whether the V1.1.1 fused regression kernel supports a request."""
    name = str(metric).lower()
    mode = "none" if scaling is None else str(scaling).lower()
    return bool(NUMBA_AVAILABLE and name in FUSED_REGRESSION_METRICS and mode in {"linear", "auto", "none", "identity"})


def _regression_metric_code(metric):
    name = str(metric).lower()
    if name in {"mse", "mean_squared_error"}: return 0
    if name in {"rmse", "root_mean_squared_error"}: return 1
    if name in {"mae", "mean_absolute_error"}: return 2
    if name in {"r2", "r2_score"}: return 3
    if name == "huber": return 4
    raise ValueError(
        f"Fused regression backend does not support metric {metric!r}. "
        "Supported: mse, rmse, mae, r2, huber."
    )


def pack_programs(programs):
    """Pack programs using array concatenation, avoiding per-node Python lists."""
    programs=list(programs)
    n=len(programs)
    if n==0:
        return (np.empty(0,dtype=np.int16),np.empty(0,dtype=np.float64),np.zeros(1,dtype=np.int64))
    genomes=[]; lengths=np.empty(n,dtype=np.int64)
    total=0
    for i,program in enumerate(programs):
        cached=getattr(program,"metadata",{}).get("_compiled_postfix") if getattr(program,"root",None) is None else None
        if cached is not None: ops,args=cached[1],cached[2]
        else: ops,args=compile_program(program)
        genomes.append((ops,args)); lengths[i]=len(ops); total+=len(ops)
    offsets=np.empty(n+1,dtype=np.int64); offsets[0]=0; np.cumsum(lengths,out=offsets[1:])
    packed_ops=np.empty(total,dtype=np.int16); packed_args=np.empty(total,dtype=np.float64)
    for i,(ops,args) in enumerate(genomes):
        a=int(offsets[i]); b=int(offsets[i+1]); packed_ops[a:b]=ops; packed_args[a:b]=args
    return packed_ops,packed_args,offsets


def estimate_fused_population_memory_mb(programs, n_jobs=-1):
    """Conservative working-memory estimate for fused population scoring.

    The fused engine intentionally does not allocate population x sample prediction
    matrices.  The estimate covers packed genomes, output vectors and a per-thread
    evaluation stack sized to the largest program.
    """
    programs=list(programs)
    if not programs: return 0.0
    total_nodes=sum(max(1,int(p.size)) for p in programs)
    max_nodes=max(max(1,int(p.size)) for p in programs)
    threads=configure_numba_threads(n_jobs) if NUMBA_AVAILABLE else 1
    bytes_=total_nodes*(2+8) + (len(programs)*4*8) + threads*max_nodes*8
    # Safety factor for Numba/runtime temporaries and Python-side packed arrays.
    return float(bytes_*2.5/(1024.0**2))


def choose_population_batch_size(programs, memory_budget_mb=None, n_jobs=-1):
    """Choose a population chunk size that respects a requested memory budget."""
    n=len(programs)
    if n == 0 or memory_budget_mb is None: return max(1,n)
    budget=float(memory_budget_mb)
    if budget <= 0: raise ValueError("memory_budget_mb must be positive when provided")
    estimate=estimate_fused_population_memory_mb(programs,n_jobs=n_jobs)
    if estimate <= budget: return n
    # Scale linearly and keep at least one program per chunk.  Use 80% of the
    # requested budget to leave headroom for estimator state and NumPy arrays.
    chunk=max(1,int(n*(0.80*budget/max(estimate,1e-12))))
    return min(n,chunk)


if NUMBA_AVAILABLE:
    @njit(cache=True, inline='always')
    def _loss_from_accumulators(sum_sq, sum_abs, sum_huber, weight_sum, y_sum, y_sq_sum, metric_code):
        if weight_sum <= 0.0:
            return 1e12
        if metric_code == 0:  # MSE
            return max(0.0, sum_sq / weight_sum)
        if metric_code == 1:  # RMSE
            return math.sqrt(max(0.0, sum_sq / weight_sum))
        if metric_code == 2:  # MAE
            return max(0.0, sum_abs / weight_sum)
        if metric_code == 4:  # Huber
            return max(0.0, sum_huber / weight_sum)
        # 1 - R2 = SSE/SST, matching the loss convention used by metrics.py.
        sst = y_sq_sum - (y_sum * y_sum) / weight_sum
        if sst <= 1e-18:
            return 0.0 if sum_sq <= 1e-18 else 1.0
        return max(0.0, sum_sq) / sst

    @njit(cache=True, parallel=True)
    def _score_population_regression_fused(
        X_train, y_train, w_train, X_val, y_val, w_val,
        opcodes, arguments, offsets, scaling_code,
        metric_code, val_metric_code, huber_delta,
    ):
        n_programs=offsets.shape[0]-1
        scales=np.empty(n_programs,dtype=np.float64)
        intercepts=np.empty(n_programs,dtype=np.float64)
        train_losses=np.empty(n_programs,dtype=np.float64)
        val_losses=np.empty(n_programs,dtype=np.float64)
        has_val=X_val.shape[0] > 0

        for j in prange(n_programs):
            start=offsets[j]; end=offsets[j+1]
            local_ops=opcodes[start:end]; local_args=arguments[start:end]

            # First pass: sufficient statistics for optional affine scaling.
            sw=sx=sy=sxx=sxy=syy=0.0
            for i in range(X_train.shape[0]):
                raw=_one(X_train,i,local_ops,local_args)
                wt=w_train[i]; yy=y_train[i]
                sw += wt; sx += wt*raw; sy += wt*yy
                sxx += wt*raw*raw; sxy += wt*raw*yy; syy += wt*yy*yy

            if sw <= 0.0:
                scale=0.0; intercept=0.0
            elif scaling_code == 1:
                var_num=sxx - sx*sx/sw
                cov_num=sxy - sx*sy/sw
                if var_num <= 1e-16*max(sw,1.0):
                    scale=0.0; intercept=sy/sw
                else:
                    scale=cov_num/var_num; intercept=sy/sw-scale*sx/sw
            else:
                scale=1.0; intercept=0.0

            scales[j]=scale; intercepts[j]=intercept

            # V1.2.3 ultra-fast exact path.  For squared-error objectives the
            # first-pass sufficient statistics already contain everything needed
            # to compute SSE after affine scaling, so do not evaluate the symbolic
            # expression a second time.  MAE/Huber still require residual-wise
            # values and therefore keep the exact second pass.
            if metric_code==0 or metric_code==1 or metric_code==3:
                sq=(scale*scale*sxx + 2.0*scale*intercept*sx + intercept*intercept*sw
                    - 2.0*scale*sxy - 2.0*intercept*sy + syy)
                if sq<0.0 and sq>-1e-8*max(1.0,syy): sq=0.0
                train_losses[j]=_loss_from_accumulators(sq,0.0,0.0,sw,sy,syy,metric_code)
            else:
                sq=ab=hub=0.0
                for i in range(X_train.shape[0]):
                    raw=_one(X_train,i,local_ops,local_args)
                    r=scale*raw+intercept-y_train[i]; wt=w_train[i]; ar=abs(r)
                    sq += wt*r*r; ab += wt*ar
                    hub += wt*(0.5*r*r if ar <= huber_delta else huber_delta*(ar-0.5*huber_delta))
                train_losses[j]=_loss_from_accumulators(sq,ab,hub,sw,sy,syy,metric_code)

            if has_val:
                vsw=vsy=vsyy=vsq=vab=vhub=0.0
                for i in range(X_val.shape[0]):
                    raw=_one(X_val,i,local_ops,local_args)
                    yy=y_val[i]; wt=w_val[i]; r=scale*raw+intercept-yy; ar=abs(r)
                    vsw += wt; vsy += wt*yy; vsyy += wt*yy*yy
                    vsq += wt*r*r; vab += wt*ar
                    vhub += wt*(0.5*r*r if ar <= huber_delta else huber_delta*(ar-0.5*huber_delta))
                val_losses[j]=_loss_from_accumulators(vsq,vab,vhub,vsw,vsy,vsyy,val_metric_code)
            else:
                val_losses[j]=train_losses[j]
        return scales,intercepts,train_losses,val_losses


def _weights_or_ones(values,n):
    if values is None: return np.ones(int(n),dtype=np.float64)
    arr=np.ascontiguousarray(np.asarray(values,dtype=np.float64))
    if arr.ndim != 1 or arr.shape[0] != int(n):
        raise ValueError("sample weights must be one-dimensional with length n_samples")
    return arr


def score_population_regression_postfix(
    X_train,y_train,X_val,y_val,programs,*,metric="rmse",validation_metric=None,
    scaling="linear",sample_weight_train=None,sample_weight_val=None,n_jobs=-1,
    huber_delta=1.0,memory_budget_mb=None,
):
    """Fused population regression scoring with bounded working memory.

    Supported objectives are MSE, RMSE, MAE, R2-loss and Huber.  Linear/auto
    scaling is fit on the training split and then reused unchanged on validation.
    Population chunks are automatically reduced when ``memory_budget_mb`` is set.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("The fused regression backend requires Numba")
    metric_code=_regression_metric_code(metric)
    val_metric=metric if validation_metric is None else validation_metric
    val_metric_code=_regression_metric_code(val_metric)
    mode="none" if scaling is None else str(scaling).lower()
    if mode in {"linear","auto"}: scaling_code=1
    elif mode in {"none","identity"}: scaling_code=0
    else: raise ValueError("Fused regression scaling must be linear, auto, or none")
    if float(huber_delta) <= 0: raise ValueError("huber_delta must be positive")

    Xtr=np.ascontiguousarray(np.asarray(X_train,dtype=np.float64)); ytr=np.ascontiguousarray(np.asarray(y_train,dtype=np.float64))
    if Xtr.ndim != 2 or ytr.ndim != 1 or len(Xtr) != len(ytr): raise ValueError("Invalid training arrays")
    if X_val is None:
        Xv=np.empty((0,Xtr.shape[1]),dtype=np.float64); yv=np.empty(0,dtype=np.float64)
    else:
        Xv=np.ascontiguousarray(np.asarray(X_val,dtype=np.float64)); yv=np.ascontiguousarray(np.asarray(y_val,dtype=np.float64))
    wtr=_weights_or_ones(sample_weight_train,len(Xtr)); wv=_weights_or_ones(sample_weight_val,len(Xv))
    configure_numba_threads(n_jobs)

    programs=list(programs); n=len(programs)
    if not n:
        empty=np.empty(0,dtype=float); return empty,empty.copy(),empty.copy(),empty.copy()
    chunk=choose_population_batch_size(programs,memory_budget_mb,n_jobs=n_jobs)
    scales=[]; intercepts=[]; train=[]; val=[]
    for start in range(0,n,chunk):
        sub=programs[start:start+chunk]; ops,args,offsets=pack_programs(sub)
        result=_score_population_regression_fused(
            Xtr,ytr,wtr,Xv,yv,wv,ops,args,offsets,scaling_code,
            metric_code,val_metric_code,float(huber_delta),
        )
        for dst,src in zip((scales,intercepts,train,val),result): dst.extend(np.asarray(src,float).tolist())
    return tuple(np.asarray(x,dtype=float) for x in (scales,intercepts,train,val))


# ---------------------------------------------------------------------------
# V1.1.4 fused binary-classification population scoring
# ---------------------------------------------------------------------------
FUSED_CLASSIFICATION_METRICS = {
    "log_loss", "logloss", "accuracy", "balanced_accuracy", "bacc",
    "f1", "f1_score", "mcc", "matthews",
}


def fused_classification_supported(metric, scaling="approx_logistic"):
    name=str(metric).lower(); mode="none" if scaling is None else str(scaling).lower()
    return bool(NUMBA_AVAILABLE and name in FUSED_CLASSIFICATION_METRICS and mode in {
        "approx_logistic","elite_logistic","auto","none","identity","fixed_sigmoid","fixed"
    })


def _classification_metric_code(metric):
    name=str(metric).lower()
    if name in {"log_loss","logloss"}: return 0
    if name in {"accuracy","acc"}: return 1
    if name in {"balanced_accuracy","bacc"}: return 2
    if name in {"f1","f1_score"}: return 3
    if name in {"mcc","matthews"}: return 4
    raise ValueError(f"Fused classification backend does not support metric {metric!r}")


if NUMBA_AVAILABLE:
    @njit(cache=True,inline='always')
    def _classification_loss_counts(tp,tn,fp,fn,total_logloss,total_weight,metric_code):
        if total_weight<=0.0: return 1e12
        if metric_code==0: return total_logloss/total_weight
        if metric_code==1: return 1.0-(tp+tn)/total_weight
        if metric_code==2:
            pos=tp+fn; neg=tn+fp
            tpr=tp/pos if pos>0 else 0.0; tnr=tn/neg if neg>0 else 0.0
            score=0.5*(tpr+tnr) if pos>0 and neg>0 else (tpr if pos>0 else tnr)
            return 1.0-score
        if metric_code==3:
            den=2.0*tp+fp+fn; return 1.0-(2.0*tp/den if den>0 else 0.0)
        den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        mcc=(tp*tn-fp*fn)/math.sqrt(den) if den>0 else 0.0
        return 1.0-(mcc+1.0)/2.0

    @njit(cache=True,inline='always')
    def _classification_accumulate(raw,target,weight,scale,intercept,metric_code):
        z=scale*raw+intercept; predicted=1 if z>=0.0 else 0
        positive=target>=0.5
        tp=weight if predicted==1 and positive else 0.0
        tn=weight if predicted==0 and not positive else 0.0
        fp=weight if predicted==1 and not positive else 0.0
        fn=weight if predicted==0 and positive else 0.0
        ll=0.0
        if metric_code==0: ll=weight*(max(z,0.0)-target*z+math.log1p(math.exp(-abs(z))))
        return tp,tn,fp,fn,ll

    @njit(cache=True,parallel=True)
    def _score_population_classification_fused(Xtr,ytr,wtr,Xv,yv,wv,ops,args,offsets,scaling_code,reg,metric_code,val_metric_code):
        n=offsets.shape[0]-1
        scales=np.empty(n,np.float64); intercepts=np.empty(n,np.float64); train=np.empty(n,np.float64); val=np.empty(n,np.float64)
        has_val=Xv.shape[0]>0
        for j in prange(n):
            start=offsets[j]; end=offsets[j+1]; local_ops=ops[start:end]; local_args=args[start:end]
            sw=sx=sy=sxx=sxy=0.0
            raw_values=np.empty(Xtr.shape[0],np.float64)
            for i in range(Xtr.shape[0]):
                raw=_one(Xtr,i,local_ops,local_args); raw_values[i]=raw; wt=wtr[i]; yy=ytr[i]
                sw+=wt; sx+=wt*raw; sy+=wt*yy; sxx+=wt*raw*raw; sxy+=wt*raw*yy
            if scaling_code==1 and sw>0.0:
                prior=min(1.0-1e-8,max(1e-8,sy/sw)); xmean=sx/sw
                var=max(0.0,sxx-sx*sx/sw); cov=sxy-xmean*sy
                curvature=max(prior*(1.0-prior),1e-8)*var+reg
                scale=cov/curvature if var>1e-18 else 0.0; scale=min(1e6,max(-1e6,scale))
                intercept=math.log(prior/(1.0-prior))-scale*xmean; intercept=min(1e6,max(-1e6,intercept))
            else: scale=1.0; intercept=0.0
            scales[j]=scale; intercepts[j]=intercept
            tp=tn=fp=fn=ll=tw=0.0
            for i in range(Xtr.shape[0]):
                wt=wtr[i]; a,b,c,d,l=_classification_accumulate(raw_values[i],ytr[i],wt,scale,intercept,metric_code)
                tp+=a;tn+=b;fp+=c;fn+=d;ll+=l;tw+=wt
            train[j]=_classification_loss_counts(tp,tn,fp,fn,ll,tw,metric_code)
            if has_val:
                tp=tn=fp=fn=ll=tw=0.0
                for i in range(Xv.shape[0]):
                    raw=_one(Xv,i,local_ops,local_args); wt=wv[i]
                    a,b,c,d,l=_classification_accumulate(raw,yv[i],wt,scale,intercept,val_metric_code)
                    tp+=a;tn+=b;fp+=c;fn+=d;ll+=l;tw+=wt
                val[j]=_classification_loss_counts(tp,tn,fp,fn,ll,tw,val_metric_code)
            else: val[j]=train[j]
        return scales,intercepts,train,val


def score_population_classification_postfix(X_train,y_train,X_val,y_val,programs,*,metric="log_loss",validation_metric=None,
        scaling="approx_logistic",regularization=1e-4,sample_weight_train=None,sample_weight_val=None,n_jobs=-1,memory_budget_mb=None):
    if not NUMBA_AVAILABLE: raise RuntimeError("The fused classification backend requires Numba")
    metric_code=_classification_metric_code(metric); val_metric_code=_classification_metric_code(metric if validation_metric is None else validation_metric)
    mode="none" if scaling is None else str(scaling).lower()
    if mode in {"approx_logistic","elite_logistic","auto"}: scaling_code=1
    elif mode in {"none","identity","fixed_sigmoid","fixed"}: scaling_code=0
    else: raise ValueError("Fused classification scaling supports approximate logistic or fixed sigmoid")
    Xtr=np.ascontiguousarray(np.asarray(X_train,np.float64)); ytr=np.ascontiguousarray(np.asarray(y_train,np.float64))
    if X_val is None: Xv=np.empty((0,Xtr.shape[1]),np.float64); yv=np.empty(0,np.float64)
    else: Xv=np.ascontiguousarray(np.asarray(X_val,np.float64)); yv=np.ascontiguousarray(np.asarray(y_val,np.float64))
    wtr=_weights_or_ones(sample_weight_train,len(Xtr)); wv=_weights_or_ones(sample_weight_val,len(Xv)); configure_numba_threads(n_jobs)
    programs=list(programs); n=len(programs)
    if not n:
        e=np.empty(0,float); return e,e.copy(),e.copy(),e.copy()
    chunk=choose_population_batch_size(programs,memory_budget_mb,n_jobs=n_jobs)
    out=[[],[],[],[]]
    for start in range(0,n,chunk):
        sub=programs[start:start+chunk]; ops,args,offsets=pack_programs(sub)
        result=_score_population_classification_fused(Xtr,ytr,wtr,Xv,yv,wv,ops,args,offsets,scaling_code,float(regularization),metric_code,val_metric_code)
        for dst,src in zip(out,result): dst.extend(np.asarray(src,float).tolist())
    return tuple(np.asarray(x,float) for x in out)
