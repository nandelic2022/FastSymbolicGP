# Migration from V1.2.6 to V1.2.7

No user-code changes are required for the standard API.

For a normal fast regression run, `profile_["variation_engine"]` now reports:

```text
numba_rootless_genome_v2
```

instead of:

```text
flat_genome_inheritance_v1
```

Advanced scientific workflows automatically fall back to the tree-backed variation engine when a real tree is required.
