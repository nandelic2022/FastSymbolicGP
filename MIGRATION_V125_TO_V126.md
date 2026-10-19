# Migration from V1.2.5 to V1.2.6

No user-code changes are required for the standard API.

The main observable change is the profiler field:

```python
model.profile_["variation_engine"]
# V1.2.6 fast path: "flat_genome_inheritance_v1"
```

The fast flat-genome path is automatically used only where it is safe. Typed/unit-enforced or hard-scientific-constraint workflows retain the generic safe variation engine.
