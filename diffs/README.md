# Complete per-file comparison

Direction: `single_agent/` → `two_agents/`.

Every corresponding source/documentation/test file has a patch. Shared root files are explicitly recorded as unchanged.

| File | Status | Patch |
| --- | --- | --- |
| `README.md` | changed | [README.md.patch](README.md.patch) |
| `__init__.py` | changed | [__init__.py.patch](__init__.py.patch) |
| `cartpole_bc.py` | changed | [cartpole_bc.py.patch](cartpole_bc.py.patch) |
| `test_cartpole.py` | changed | [test_cartpole.py.patch](test_cartpole.py.patch) |
| `shared/.gitignore` | unchanged | [shared__.gitignore.patch](shared__.gitignore.patch) |
| `shared/README.md` | unchanged | [shared__README.md.patch](shared__README.md.patch) |
| `shared/HIGHWAY_FORMAT.md` | unchanged | [shared__HIGHWAY_FORMAT.md.patch](shared__HIGHWAY_FORMAT.md.patch) |
| `shared/pyproject.toml` | unchanged | [shared__pyproject.toml.patch](shared__pyproject.toml.patch) |
| `shared/practice_io.py` | unchanged | [shared__practice_io.py.patch](shared__practice_io.py.patch) |
| `shared/plot_highway.py` | unchanged | [shared__plot_highway.py.patch](shared__plot_highway.py.patch) |
| `shared/test_highway_plot.py` | unchanged | [shared__test_highway_plot.py.patch](shared__test_highway_plot.py.patch) |
| `shared/example_highway.csv` | unchanged | [shared__example_highway.csv.patch](shared__example_highway.csv.patch) |
| `shared/make_diffs.py` | unchanged | [shared__make_diffs.py.patch](shared__make_diffs.py.patch) |

`all_changes.patch` contains all changed-file hunks. `manifest.json` records both SHA256 hashes for every file.
Unchanged-file patches contain an explanatory comment and hash, with no change hunk.

Regenerate from the repository root:

```bash
python make_diffs.py
```

This generator does not import or execute the learning or plotting programs. It only regenerates files in the diff output directory.
