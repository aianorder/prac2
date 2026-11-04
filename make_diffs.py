"""Generate every per-file single-agent/two-agent diff without running either program."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

SHARED_FILES = [".gitignore", "README.md", "HIGHWAY_FORMAT.md", "pyproject.toml", "practice_io.py",
                "plot_highway.py", "test_highway_plot.py", "example_highway.csv", "make_diffs.py"]


def digest(content):
    return hashlib.sha256(content).hexdigest()


def generate(root, output):
    root, output = Path(root), Path(output)
    if output.resolve() in {root.resolve(), (root / "single_agent").resolve(), (root / "two_agents").resolve()}:
        raise ValueError("Diff output must be separate from the source directories")
    output.mkdir(parents=True, exist_ok=True)
    left_root, right_root = root / "single_agent", root / "two_agents"
    names = {str(p.relative_to(folder)) for folder in (left_root, right_root)
             for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    pairs = [(name, left_root / name, right_root / name, "variant") for name in sorted(names)]
    pairs += [("shared/" + name, root / name, root / name, "shared") for name in SHARED_FILES]
    manifest, combined = [], []
    for name, left, right, scope in pairs:
        before = left.read_bytes() if left.exists() else b""
        after = right.read_bytes() if right.exists() else b""
        left_name = f"single_agent/{name}" if scope == "variant" else name
        right_name = f"two_agents/{name}" if scope == "variant" else name
        patch = "".join(difflib.unified_diff(before.decode().splitlines(keepends=True),
                                             after.decode().splitlines(keepends=True),
                                             fromfile=left_name, tofile=right_name))
        patch_name = name.replace("/", "__") + ".patch"
        # An unchanged file still has its own file, with explicit hashes instead of a misleading hunk.
        text = patch or f"# UNCHANGED: {name}\n# SHA256 (both sides): {digest(before)}\n"
        (output / patch_name).write_text(text, encoding="utf-8")
        manifest.append({"file": name, "scope": scope, "status": "changed" if patch else "unchanged",
                         "left": left_name, "right": right_name, "left_sha256": digest(before),
                         "right_sha256": digest(after), "patch": patch_name})
        if patch:
            combined.append(patch)
    (output / "all_changes.patch").write_text("".join(combined), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "direction": "single_agent -> two_agents", "files": manifest,
        "source_files": len(manifest), "changed": sum(r["status"] == "changed" for r in manifest),
        "note": "Diff generation reads source files only; no collection, training, evaluation or tests are executed.",
    }, indent=2) + "\n", encoding="utf-8")
    lines = ["# Complete per-file comparison", "", "Direction: `single_agent/` → `two_agents/`.", "",
             "Every corresponding source/documentation/test file has a patch. Shared root files are explicitly recorded as unchanged.",
             "", "| File | Status | Patch |", "| --- | --- | --- |"]
    lines += [f"| `{r['file']}` | {r['status']} | [{r['patch']}]({r['patch']}) |" for r in manifest]
    lines += ["", "`all_changes.patch` contains all changed-file hunks. `manifest.json` records both SHA256 hashes for every file.",
              "Unchanged-file patches contain an explanatory comment and hash, with no change hunk.",
              "", "Regenerate from the repository root:", "", "```bash", "python make_diffs.py", "```", "",
              "This generator does not import or execute the learning or plotting programs. It only regenerates files in the diff output directory."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"source_files": len(manifest), "changed": sum(r["status"] == "changed" for r in manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.root, args.output or args.root / "diffs"), indent=2))


if __name__ == "__main__":
    main()

