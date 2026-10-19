"""Thread-safe bounded caches retained for V1 API compatibility."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
import hashlib, numpy as np

@dataclass
class CacheStatistics:
    hits:int=0; misses:int=0; evictions:int=0; bytes_used:int=0; entries:int=0
    @property
    def hit_rate(self):
        n=self.hits+self.misses; return 0.0 if n==0 else self.hits/n
    def as_dict(self): return {"hits":self.hits,"misses":self.misses,"evictions":self.evictions,"bytes_used":self.bytes_used,"entries":self.entries,"hit_rate":self.hit_rate}

class ArrayLRUCache:
    def __init__(self,max_bytes=512*1024*1024): self.max_bytes=max(0,int(max_bytes)); self._items=OrderedDict(); self._lock=RLock(); self._stats=CacheStatistics()
    def get(self,key):
        if self.max_bytes<=0:return None
        with self._lock:
            if key not in self._items: self._stats.misses+=1; return None
            v=self._items[key]; self._items.move_to_end(key); self._stats.hits+=1; return v
    def set(self,key,value):
        if self.max_bytes<=0:return
        a=np.ascontiguousarray(np.asarray(value,float));
        if a.nbytes>self.max_bytes:return
        a.setflags(write=False)
        with self._lock:
            old=self._items.pop(key,None)
            if old is not None:self._stats.bytes_used-=old.nbytes
            while self._items and self._stats.bytes_used+a.nbytes>self.max_bytes:
                _,e=self._items.popitem(last=False); self._stats.bytes_used-=e.nbytes; self._stats.evictions+=1
            self._items[key]=a; self._stats.bytes_used+=a.nbytes; self._stats.entries=len(self._items)
    def clear(self):
        with self._lock:self._items.clear();self._stats.bytes_used=0;self._stats.entries=0
    def statistics(self): self._stats.entries=len(self._items); return self._stats.as_dict()
    def __getstate__(self): return {"max_bytes":self.max_bytes,"items":list(self._items.items()),"stats":self._stats}
    def __setstate__(self,s): self.max_bytes=s["max_bytes"];self._items=OrderedDict(s.get("items",[]));self._stats=s.get("stats",CacheStatistics());self._lock=RLock()

class MetadataLRUCache:
    def __init__(self,max_entries=100000): self.max_entries=max(0,int(max_entries));self._items=OrderedDict();self._lock=RLock();self._stats=CacheStatistics()
    def get(self,key):
        if self.max_entries<=0:return None
        with self._lock:
            if key not in self._items:self._stats.misses+=1;return None
            v=self._items[key];self._items.move_to_end(key);self._stats.hits+=1;return v
    def set(self,key,value):
        if self.max_entries<=0:return
        with self._lock:
            self._items.pop(key,None)
            while self._items and len(self._items)>=self.max_entries:self._items.popitem(last=False);self._stats.evictions+=1
            self._items[key]=value;self._stats.entries=len(self._items)
    def clear(self): self._items.clear();self._stats.entries=0
    def statistics(self): self._stats.entries=len(self._items);return self._stats.as_dict()

def dataset_token(X):
    a=np.asarray(X);flat=a.ravel();sample=np.ascontiguousarray(flat if flat.size<=96 else flat[np.linspace(0,flat.size-1,96,dtype=int)])
    return (tuple(a.shape),tuple(a.strides),str(a.dtype),hashlib.blake2b(sample.view(np.uint8),digest_size=8).hexdigest())
