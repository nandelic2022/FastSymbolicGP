from __future__ import annotations
import hashlib
import numpy as np


def semantic_sample_indices(n_samples: int, sample_size: int = 128) -> np.ndarray:
    n=int(n_samples); k=max(1,min(int(sample_size),n))
    if k>=n: return np.arange(n,dtype=int)
    return np.unique(np.linspace(0,n-1,k,dtype=int))


def semantic_signature(program, X, sample_size: int = 128) -> np.ndarray:
    X=np.asarray(X,float)
    idx=semantic_sample_indices(len(X),sample_size)
    values=np.asarray(program.predict(X[idx]),float)
    values=np.nan_to_num(values,nan=0.0,posinf=1e6,neginf=-1e6)
    mean=float(np.mean(values)); std=float(np.std(values))
    if std<=1e-12:
        return np.zeros_like(values,dtype=float)
    return np.clip((values-mean)/std,-12.0,12.0)


def semantic_distance_vectors(a,b) -> float:
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.shape!=b.shape: raise ValueError("semantic vectors must have matching shapes")
    if a.size==0: return 0.0
    return float(np.sqrt(np.mean((a-b)**2)))


def semantic_distance(a,b,X,sample_size: int=128) -> float:
    return semantic_distance_vectors(semantic_signature(a,X,sample_size),semantic_signature(b,X,sample_size))


def semantic_hash(signature, decimals: int=5) -> str:
    arr=np.round(np.asarray(signature,float),int(decimals))
    return hashlib.blake2b(np.ascontiguousarray(arr).tobytes(),digest_size=12).hexdigest()


def assign_semantic_novelty(population, X, archive=(), sample_size: int=128, k_neighbors: int=5):
    pop=list(population); archive=list(archive or [])
    if not pop: return []
    signatures=[]
    for p in pop:
        sig=semantic_signature(p,X,sample_size)
        p.metadata["semantic_signature"]=sig.tolist()
        p.metadata["semantic_hash"]=semantic_hash(sig)
        signatures.append(sig)
    archive_sigs=[]
    for p in archive:
        raw=p.metadata.get("semantic_signature")
        if raw is None:
            try: raw=semantic_signature(p,X,sample_size)
            except Exception: continue
        raw=np.asarray(raw,float)
        if len(raw)==len(signatures[0]): archive_sigs.append(raw)
    k=max(1,int(k_neighbors))
    for i,p in enumerate(pop):
        d=[]
        for j,sig in enumerate(signatures):
            if i!=j: d.append(semantic_distance_vectors(signatures[i],sig))
        d.extend(semantic_distance_vectors(signatures[i],sig) for sig in archive_sigs)
        d=sorted(x for x in d if np.isfinite(x))
        novelty=float(np.mean(d[:min(k,len(d))])) if d else 0.0
        p.metadata["semantic_novelty"]=novelty
    return signatures


def update_semantic_archive(archive, candidates, max_size: int=64):
    max_size=max(0,int(max_size))
    if max_size==0: return []
    pool=[p.clone() for p in (archive or [])]+[p.clone() for p in candidates]
    unique={}
    for p in pool:
        key=p.metadata.get("semantic_hash") or p.structural_hash()
        old=unique.get(key)
        score=(float(p.metadata.get("semantic_novelty",0.0)),-float(getattr(p,"validation_loss",np.inf)))
        if old is None:
            unique[key]=p
        else:
            oldscore=(float(old.metadata.get("semantic_novelty",0.0)),-float(getattr(old,"validation_loss",np.inf)))
            if score>oldscore: unique[key]=p
    pool=list(unique.values())
    # Preserve a mixture of behaviorally novel and accurate programs.
    pool.sort(key=lambda p:(-float(p.metadata.get("semantic_novelty",0.0)),float(getattr(p,"validation_loss",np.inf)),p.size))
    return pool[:max_size]


def semantic_archive_report(archive):
    archive=list(archive or [])
    nov=[float(p.metadata.get("semantic_novelty",0.0)) for p in archive]
    return {
        "size":len(archive),
        "mean_novelty":float(np.mean(nov)) if nov else 0.0,
        "max_novelty":float(np.max(nov)) if nov else 0.0,
        "unique_semantics":len({p.metadata.get("semantic_hash",p.structural_hash()) for p in archive}),
    }
