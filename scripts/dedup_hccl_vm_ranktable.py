#!/usr/bin/env python3
"""Remove exact duplicate addresses from an HCCL-VM ranktable.

The HCCL-VM competition topology can emit the same local EID more than once
in a rank_addr_list.  This tool removes only byte-for-byte equivalent JSON
entries.  Reusing an address with different metadata is treated as an error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate exact EID entries in an HCCL-VM ranktable."
    )
    parser.add_argument("ranktable", type=Path, help="Path to ranktable.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the planned changes without writing the file.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not save a timestamped copy before replacing the input file.",
    )
    return parser.parse_args()


def load_ranktable(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {path}: {error}") from error

    if not isinstance(data, dict) or not isinstance(data.get("rank_list"), list):
        raise ValueError("ranktable must contain a rank_list array")
    return data


def deduplicate(data: dict[str, Any]) -> tuple[int, list[str]]:
    removed = 0
    changes: list[str] = []
    group_counts: dict[tuple[int, str], dict[int, int]] = defaultdict(dict)

    for rank in data["rank_list"]:
        if not isinstance(rank, dict) or not isinstance(rank.get("level_list"), list):
            raise ValueError("every rank must contain a level_list array")
        rank_id = rank.get("rank_id")
        if not isinstance(rank_id, int):
            raise ValueError("every rank must contain an integer rank_id")

        for level in rank["level_list"]:
            if not isinstance(level, dict):
                raise ValueError(f"rank {rank_id} contains an invalid level entry")
            net_layer = level.get("net_layer")
            instance = level.get("net_instance_id")
            addresses = level.get("rank_addr_list")
            if not isinstance(net_layer, int) or not isinstance(instance, str):
                raise ValueError(f"rank {rank_id} contains invalid network metadata")
            if not isinstance(addresses, list):
                raise ValueError(
                    f"rank {rank_id} layer {net_layer} instance {instance} "
                    "does not contain a rank_addr_list array"
                )

            unique: list[dict[str, Any]] = []
            seen: dict[tuple[str, str], dict[str, Any]] = {}
            for entry in addresses:
                if not isinstance(entry, dict):
                    raise ValueError(f"rank {rank_id} contains an invalid address entry")
                addr_type = entry.get("addr_type")
                address = entry.get("addr")
                if not isinstance(addr_type, str) or not isinstance(address, str):
                    raise ValueError(f"rank {rank_id} contains an address without type or value")
                key = (addr_type, address)
                previous = seen.get(key)
                if previous is None:
                    seen[key] = entry
                    unique.append(entry)
                    continue
                if previous != entry:
                    raise ValueError(
                        f"rank {rank_id} reuses {addr_type} {address} with conflicting metadata"
                    )
                removed += 1

            if len(unique) != len(addresses):
                changes.append(
                    f"rank={rank_id} layer={net_layer} instance={instance} "
                    f"addresses={len(addresses)}->{len(unique)}"
                )
                level["rank_addr_list"] = unique

            group_key = (net_layer, instance)
            if rank_id in group_counts[group_key]:
                raise ValueError(
                    f"rank {rank_id} has duplicate level entries for layer {net_layer} "
                    f"instance {instance}"
                )
            group_counts[group_key][rank_id] = len(unique)

    inconsistent: list[str] = []
    for (net_layer, instance), counts_by_rank in sorted(group_counts.items()):
        counts = sorted(set(counts_by_rank.values()))
        if len(counts) != 1:
            details = ", ".join(
                f"rank{rank_id}={count}"
                for rank_id, count in sorted(counts_by_rank.items())
            )
            inconsistent.append(
                f"layer={net_layer} instance={instance}: {details}"
            )

    if inconsistent:
        raise ValueError(
            "address counts remain inconsistent after deduplication:\n  "
            + "\n  ".join(inconsistent)
        )
    return removed, changes


def write_atomic(path: Path, data: dict[str, Any]) -> None:
    source_mode = stat.S_IMODE(path.stat().st_mode)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False
        ) as stream:
            temporary_name = stream.name
            json.dump(data, stream, indent=4, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, source_mode)
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> int:
    args = parse_args()
    path = args.ranktable.resolve()
    try:
        data = load_ranktable(path)
        removed, changes = deduplicate(data)
        for change in changes:
            print(change)
        print(f"removed={removed} ranks={len(data['rank_list'])}")

        if args.dry_run:
            print("dry-run: ranktable not modified")
            return 0
        if removed == 0:
            print("ranktable already contains unique addresses")
            return 0

        if not args.no_backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = path.with_name(f"{path.name}.pre-dedup.{timestamp}.bak")
            shutil.copy2(path, backup)
            print(f"backup={backup}")
        write_atomic(path, data)
        print(f"updated={path}")
        return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
