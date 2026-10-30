#!/usr/bin/env python3
"""Per-episode outcome grades. Sidecar at <dataset root>/meta/outcomes.json, not a dataset
column (the verdict is unknown when frames are written and is the one field revised later):
  {"outcomes": {"<episode>": "good"|"mid"|"bad"|"drop"}}
good = clean success; mid = achieved but sloppy (trained as a success); bad = not achieved;
drop = excluded from training. Unlisted = good; legacy 1/0 read as good/bad. The collector
writes this file; this CLI relabels afterwards:
  python -m chess.real.outcomes my_dataset --bad 12 15 27 | --mid 4 | --drop 9 | --list
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# worst first: the order the CLI prints and the UI lays its buttons out in
GRADES = ("bad", "mid", "good")
# not a grade: training skips the episode entirely; frames stay on disk
DROP = "drop"
LABELS = GRADES + (DROP,)
_LEGACY = {1: "good", 0: "bad"}      # the 1/0 success/fail encoding of older sidecars


def grade(v) -> str:
    """Stored value (label string or legacy 0/1) -> one of LABELS; ValueError on anything else."""
    if isinstance(v, str) and v in LABELS:
        return v
    try:
        return _LEGACY[int(v)]
    except (TypeError, ValueError, KeyError):
        raise ValueError(f"not an outcome label: {v!r} (one of {', '.join(LABELS)})")


DEFAULT_NAMESPACE = "local"     # repo id namespace of a bare dataset name


def repo_id(data: str) -> str:
    """Dataset name -> lerobot repo id: `<namespace>/<name>` as given, a bare name under `local/`."""
    return data if "/" in data else f"{DEFAULT_NAMESPACE}/{data}"


def dataset_root(data: str) -> Path:
    """Dataset name -> on-disk root, $HF_LEROBOT_HOME/<repo id> (lerobot's own layout)."""
    hf_home = Path(os.environ.get("HF_LEROBOT_HOME",
                                  str(Path.home() / ".cache" / "huggingface" / "lerobot")))
    return hf_home / repo_id(data)


def _total_episodes(root: Path) -> int:
    return int(json.loads((root / "meta" / "info.json").read_text())["total_episodes"])


def path(root: Path) -> Path:
    return root / "meta" / "outcomes.json"


def load(root: Path) -> dict:
    p = path(root)
    return json.loads(p.read_text()) if p.exists() else {"outcomes": {}}


def save(root: Path, data: dict) -> None:
    path(root).write_text(json.dumps(data, indent=1))


def grades(root: Path) -> dict[int, str]:
    """{episode index: label} with legacy values normalized."""
    return {int(k): grade(v) for k, v in (load(root).get("outcomes") or {}).items()}


def dropped(root: Path) -> set[int]:
    """Episodes labeled drop; empty when the sidecar is absent or unreadable."""
    try:
        return {e for e, g in grades(root).items() if g == DROP}
    except (OSError, ValueError):
        return set()


def mark(root: Path, episodes, g: str) -> dict:
    """Label episode indices (checked against total_episodes) and write the sidecar; returns the dict."""
    if g not in LABELS:
        raise SystemExit(f"unknown label {g!r} (one of {', '.join(LABELS)})")
    total = _total_episodes(root)
    data = load(root)
    oc = data.setdefault("outcomes", {})
    for e in episodes:
        e = int(e)
        if not (0 <= e < total):
            raise SystemExit(f"episode {e} out of range (0..{total - 1})")
        oc[str(e)] = g
    save(root, data)
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="dataset name (<namespace>/<name> or bare <name>)")
    ap.add_argument("--bad", type=int, nargs="+", default=[], metavar="EP",
                    help="episodes where the task was not achieved")
    ap.add_argument("--mid", type=int, nargs="+", default=[], metavar="EP",
                    help="episodes achieved but sloppily (trained as a success)")
    ap.add_argument("--good", type=int, nargs="+", default=[], metavar="EP",
                    help="clean successes (undoes a prior --bad/--mid; unlisted defaults to good)")
    ap.add_argument("--drop", type=int, nargs="+", default=[], metavar="EP",
                    help="episodes excluded from training entirely (frames stay on disk)")
    ap.add_argument("--list", action="store_true", help="print the current labels and exit")
    args = ap.parse_args()
    root = dataset_root(args.data)
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"{root} not found")

    for g in LABELS:
        if getattr(args, g):
            mark(root, getattr(args, g), g)

    g = grades(root)
    tally = "  ".join(f"{n} {name}" for name in LABELS
                      for n in [sum(1 for v in g.values() if v == name)] if n)
    print(f"{root.name}: {_total_episodes(root)} episodes, {len(g)} judged "
          f"({tally or 'none'})")
    if args.list:
        for e in sorted(g):
            print(f"  ep {e}: {g[e]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
