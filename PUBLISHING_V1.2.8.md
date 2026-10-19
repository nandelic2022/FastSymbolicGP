# FastSymbolicGP 1.2.8 — Publishing checklist (Windows)

**Release state:** source preparation only; this document does NOT indicate that GitHub or PyPI has been updated.

## 1. Check your environment and test before upload

Open Anaconda Prompt in this directory and run:

```bat
conda activate py311
python --version
python -m pip install --upgrade build twine pytest
python -m pytest -q
```

Stop and investigate if the test suite fails. For the Numba-enabled fast paths,
install the optional accelerator before the tests:

```bat
python -m pip install "numba>=0.58"
```

## 2. Build distributions and check metadata

```bat
if exist dist rmdir /s /q dist
python -m build
python -m twine check dist\*
```

Expected artifact names:

- `dist\fastsymbolicgp-1.2.8-py3-none-any.whl`
- `dist\fastsymbolicgp-1.2.8.tar.gz`

## 3. Update existing GitHub repository while retaining its history

Recommended for an existing local clone of https://github.com/nandelic2022/FastSymbolicGP:
copy this release source over the working tree, remove obsolete files that are absent
from V1.2.8, then run `git status`, `git add -A`, `git diff --cached --stat`,
`git commit -m "Release FastSymbolicGP 1.2.8"`, `git push origin main`.

If you do not have a clone, in this CLEAN EXTRACTED SOURCE DIRECTORY (no `.git` directory):

```bat
git init -b main
git remote add origin https://github.com/nandelic2022/FastSymbolicGP.git
git fetch origin main
git reset --mixed origin/main
git add -A
git diff --cached --stat
git commit -m "Release FastSymbolicGP 1.2.8"
git push -u origin main
```

`git reset --mixed` here connects your freshly extracted source files to the
existing history WITHOUT replacing your working files. Review the staged diff
before committing and abort if it contains unexpected changes. **Never use**
`git push --force` for this release.

## 4. Publish to PyPI once validated

Ensure you can manage the existing `fastsymbolicgp` PyPI project. From the release
source directory:

```bat
python -m twine upload dist\*
```

Twine will ask for credentials. Use PyPI username `__token__` and paste your
project-scoped PyPI API token in the hidden password prompt. Do not paste it
into chat, files, source control, or terminal commands.

Alternative: set up trusted publishing with PyPI and a protected GitHub Actions
release workflow. Do not upload the same version twice: PyPI release filenames
cannot be reused.

## 5. Tag and validate

```bat
git tag -a v1.2.8 -m "FastSymbolicGP 1.2.8"
git push origin v1.2.8
python -m pip install --upgrade "fastsymbolicgp[acceleration]==1.2.8"
python -c "import fastsymbolicgp; print(fastsymbolicgp.__version__)"
fastsymbolicgp-self-test
```

Publish the GitHub Release from tag `v1.2.8` with `RELEASE_NOTES_V1.2.8.md`.
