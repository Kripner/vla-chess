#!/usr/bin/env python3
"""Sub-task marks of end-to-end episodes, in lerobot's native representation: persistent
`language_persistent` rows (style="subtask", content = the fixed instruction string from
chess.task, timestamp = when the span begins), broadcast to every frame of the episode —
the layout lerobot's own annotation pipeline writes and `language_render.active_at`
resolves. The rows in the data parquets are the single source of truth (no sidecar).

The writer collects marks while recording; because the streaming writer holds data
shards open (no footer until the file rolls or the dataset is finalized), fresh episodes'
rows wait in a transient journal (meta/segments_pending.json) and `flush` moves them into
the parquets as soon as the shard is readable. Every reader here falls back to the journal
for not-yet-flushed episodes. After a clean finish the journal is gone; a crash leaves it
as the recovery record. This CLI audits marks:

  python -m chess.real.segments <dataset> --list [EP]      # marks + gripper command at each
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

E2E_SEQUENCE = ("approach", "pick", "transport", "place", "home")
STYLE = "subtask"
ROLE = "assistant"


def instruction(name: str, source, target, piece: str = "king") -> str:
    """The fixed instruction string for a sub-task of the move source->target (cells)."""
    from chess import task as T
    return {"approach": lambda: T.approach_task(source, piece),
            "pick": lambda: T.pick_task(source, piece),
            "transport": lambda: T.transport_task(target, piece),
            "place": lambda: T.place_task(target, piece),
            "home": T.home_task}[name]()


def row(name: str, source, target, t: float, piece: str = "king") -> dict:
    """One persistent language row for a mark."""
    return {"role": ROLE, "content": instruction(name, source, target, piece),
            "style": STYLE, "timestamp": float(t), "camera": None, "tool_calls": None}


def _fps(root: Path) -> float:
    return float(json.loads((root / "meta" / "info.json").read_text())["fps"])


def _episode_shard(root: Path, ep: int):
    """(parquet path, table, row mask of the episode) for the shard holding episode `ep`."""
    import pyarrow.parquet as pq
    for p in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
        t = pq.read_table(p)
        eps = t.column("episode_index").to_numpy()
        if ep in eps:
            return Path(p), t, eps == ep
    raise KeyError(f"episode {ep} not found under {root}/data")


def _journal_path(root: Path) -> Path:
    return root / "meta" / "segments_pending.json"


def journal_append(root: Path, ep: int, rows: list[dict]) -> None:
    """Queue an episode's mark rows until its data shard is readable (recorder save path)."""
    p = _journal_path(root)
    data = json.loads(p.read_text()) if p.exists() else {}
    data[str(ep)] = rows
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(p)


def flush(root: Path) -> tuple[int, int]:
    """Move journaled rows into their episodes' parquets where readable -> (flushed, left).
    Drains fully once the dataset is finalized; the journal file disappears when empty.
    One read-modify-write per shard, atomic (tmp + rename), so an interrupt never leaves a
    truncated shard."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
    from lerobot.datasets.language import LANGUAGE_PERSISTENT

    p = _journal_path(root)
    if not p.exists():
        return (0, 0)
    data = json.loads(p.read_text())
    pending = {int(k): sorted(rows, key=lambda r: r["timestamp"]) for k, rows in data.items()}
    done: list[int] = []
    for shard in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
        try:
            t = pq.read_table(shard)
        except Exception:
            continue                          # shard still open; its episodes stay queued
        eps_col = t.column("episode_index").to_numpy()
        here = set(np.unique(eps_col)) & set(pending)
        if not here:
            continue
        old = (t.column(LANGUAGE_PERSISTENT).to_pylist()
               if LANGUAGE_PERSISTENT in t.column_names else [None] * len(t))
        cells = [pending[int(e)] if e in here else old[i] for i, e in enumerate(eps_col)]
        col = pa.array(cells)
        if LANGUAGE_PERSISTENT in t.column_names:
            t = t.set_column(t.column_names.index(LANGUAGE_PERSISTENT), LANGUAGE_PERSISTENT, col)
        else:
            t = t.append_column(LANGUAGE_PERSISTENT, col)
        tmp = Path(shard).with_suffix(".parquet.tmp")
        write_table_one_row_group_per_episode(t, tmp)
        tmp.replace(shard)
        done += [int(e) for e in here]
    for k in done:
        del data[str(k)]
    if data:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(p)
    else:
        p.unlink(missing_ok=True)
    return (len(done), len(data))


def _rows_of(root: Path, ep: int) -> list[dict] | None:
    """The episode's persistent rows: from its parquet, else the pending journal."""
    try:
        _, t, mask = _episode_shard(root, ep)
        idx = int(np.flatnonzero(mask)[0])
        cell = (t.column("language_persistent")[idx].as_py()
                if "language_persistent" in t.column_names else None)
        if cell:
            return cell
    except Exception:
        pass
    p = _journal_path(root)
    if p.exists():
        return json.loads(p.read_text()).get(str(ep))
    return None


def marks(root: Path, ep: int) -> list[tuple[int, str]]:
    """The episode's subtask marks as [(frame, content)], ascending. Empty when unmarked."""
    cell = _rows_of(root, ep)
    if not cell:
        return []
    fps = _fps(root)
    rows = [r for r in cell if r.get("style") == STYLE]
    rows.sort(key=lambda r: r["timestamp"])
    return [(int(round(r["timestamp"] * fps)), r["content"]) for r in rows]


def unmarked(root: Path) -> set[int]:
    """Episodes with no persistent rows on any frame: the collector and the re-mark leave a
    failure that does not decompose blank. One pass over the shards, row group by row group,
    two columns (a few seconds on 1.5M frames). Rows still in the pending journal count."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from lerobot.datasets.language import LANGUAGE_PERSISTENT
    seen: set[int] = set()
    marked: set[int] = set()
    for shard in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
        pf = pq.ParquetFile(shard)
        has_col = LANGUAGE_PERSISTENT in pf.schema_arrow.names
        cols = ["episode_index"] + ([LANGUAGE_PERSISTENT] if has_col else [])
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=cols)
            eps = t.column("episode_index").to_numpy()
            seen.update(int(e) for e in np.unique(eps))
            if has_col:
                n = pc.fill_null(pc.list_value_length(t.column(LANGUAGE_PERSISTENT)), 0).to_numpy()
                marked.update(int(e) for e in np.unique(eps[n > 0]))
    p = _journal_path(root)
    if p.exists():
        marked.update(int(k) for k, rows in json.loads(p.read_text()).items() if rows)
    return seen - marked


def name_of(content: str) -> str | None:
    """Sub-task name for a mark row's instruction sentence (None when unrecognized).
    The rows keep the sentences as human-readable annotation; the NAME is what training
    and switching consume — it is the single-token conditioning label (pi05_subtask)."""
    c = content.lower()
    if c.startswith("reach above"):
        return "approach"
    if c.startswith("pick up"):
        return "pick"
    if c.startswith("carry"):
        return "transport"
    if c.startswith("put the"):
        return "place"
    if c.startswith("return to"):
        return "home"
    return None


def validate(root: Path, ep: int, source=None, target=None, piece: str = "king") -> list[str]:
    """Problems with the episode's marks: all five names once each, ascending, approach at
    t~0; with (source, target) also that each content is the exact chess.task string."""
    ms = marks(root, ep)
    problems = []
    names = [name_of(c) for _, c in ms]
    if names != list(E2E_SEQUENCE):
        problems.append(f"marks are {names}, expected {list(E2E_SEQUENCE)}")
        return problems
    if ms[0][0] != 0:
        problems.append(f"approach mark at frame {ms[0][0]}, expected 0")
    frames = [f for f, _ in ms]
    if frames != sorted(frames) or len(set(frames)) != len(frames):
        problems.append(f"mark frames not strictly ascending: {frames}")
    if source is not None and target is not None:
        for (f, content), name in zip(ms, names):
            want = instruction(name, source, target, piece)
            if content != want:
                problems.append(f"{name}@{f}: {content!r} != {want!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset name (<namespace>/<name> or bare <name>)")
    ap.add_argument("--list", nargs="*", type=int, default=None, metavar="EP",
                    help="print marks (+ the gripper command at each mark frame); no EP = all")
    ap.add_argument("--flush", action="store_true",
                    help="drain pending journal rows into the data parquets")
    a = ap.parse_args()
    from chess.real.outcomes import dataset_root
    root = dataset_root(a.dataset)
    n_eps = json.loads((root / "meta" / "info.json").read_text())["total_episodes"]

    if a.flush:
        done, left = flush(root)
        print(f"flushed {done}, pending {left}")
        return 0 if left == 0 else 1

    eps = a.list if a.list else range(n_eps)
    for ep in eps:
        try:
            path, t, mask = _episode_shard(root, int(ep))
        except KeyError:
            continue
        grip = t.column("action").to_numpy(zero_copy_only=False)[mask]
        ms = marks(root, int(ep))
        if not ms:
            print(f"ep {ep:4d}: (no marks)")
            continue
        parts = [f"{name_of(c)}@{f} grip={float(np.asarray(grip[min(f, len(grip) - 1)])[5]):.0f}"
                 for f, c in ms]
        probs = validate(root, int(ep))
        print(f"ep {ep:4d}: " + "  ".join(parts) + ("   PROBLEMS: " + "; ".join(probs) if probs else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
