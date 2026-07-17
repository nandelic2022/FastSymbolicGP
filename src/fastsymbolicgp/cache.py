"""Thread-safe bounded caches used by the V0.7.0 DAG evaluator."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Hashable
import numpy as np
import hashlib


@dataclass
class CacheStatistics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_used: int = 0
    entries: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 0.0 if total == 0 else self.hits / total

    def as_dict(self) -> dict:
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "evictions": int(self.evictions),
            "bytes_used": int(self.bytes_used),
            "entries": int(self.entries),
            "hit_rate": float(self.hit_rate),
        }


class ArrayLRUCache:
    """An LRU cache for NumPy arrays with a hard byte budget.

    Arrays are stored read-only to prevent accidental corruption across
    generations or islands. The cache is safe for the threaded island runner.
    """

    def __init__(self, max_bytes: int = 512 * 1024 * 1024):
        self.max_bytes = max(0, int(max_bytes))
        self._items: OrderedDict[Hashable, np.ndarray] = OrderedDict()
        self._lock = RLock()
        self._stats = CacheStatistics()

    def get(self, key: Hashable):
        if self.max_bytes <= 0:
            return None
        with self._lock:
            value = self._items.get(key)
            if value is None:
                self._stats.misses += 1
                return None
            self._items.move_to_end(key)
            self._stats.hits += 1
            return value

    def set(self, key: Hashable, value: np.ndarray):
        if self.max_bytes <= 0:
            return
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        if array.nbytes > self.max_bytes:
            return
        array.setflags(write=False)
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._stats.bytes_used -= previous.nbytes
            while self._items and self._stats.bytes_used + array.nbytes > self.max_bytes:
                _, evicted = self._items.popitem(last=False)
                self._stats.bytes_used -= evicted.nbytes
                self._stats.evictions += 1
            self._items[key] = array
            self._stats.bytes_used += array.nbytes
            self._stats.entries = len(self._items)

    def clear(self):
        with self._lock:
            self._items.clear()
            self._stats.bytes_used = 0
            self._stats.entries = 0

    def statistics(self) -> dict:
        with self._lock:
            self._stats.entries = len(self._items)
            return self._stats.as_dict()

    def __getstate__(self):
        with self._lock:
            return {
                "max_bytes": self.max_bytes,
                "items": list(self._items.items()),
                "stats": self._stats,
            }

    def __setstate__(self, state):
        self.max_bytes = int(state.get("max_bytes", 0))
        self._items = OrderedDict(state.get("items", []))
        self._stats = state.get("stats", CacheStatistics())
        self._lock = RLock()


def dataset_token(X: np.ndarray) -> tuple:
    """Return a stable, collision-resistant token for a matrix view.

    A short content fingerprint prevents stale cache reuse when NumPy recycles
    the same memory address for a newly sampled or perturbed matrix.
    """
    arr = np.asarray(X)
    pointer = int(arr.__array_interface__["data"][0]) if arr.size else 0
    flat = arr.ravel()
    if flat.size <= 96:
        sample = np.ascontiguousarray(flat)
    else:
        idx = np.linspace(0, flat.size - 1, 96, dtype=np.int64)
        sample = np.ascontiguousarray(flat[idx])
    fingerprint = hashlib.blake2b(sample.view(np.uint8), digest_size=8).hexdigest()
    return (pointer, tuple(arr.shape), tuple(arr.strides), str(arr.dtype), fingerprint)
