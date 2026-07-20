#!/usr/bin/env python3
import json
from pathlib import Path


ranktable = Path("/home/workspace/hvm/hcomm/test/hccl_vm/hccl_vm_install/data/ranktable.json")
data = json.loads(ranktable.read_text())

for rank in data["rank_list"]:
    for level in rank.get("level_list", []):
        unique = []
        seen = set()
        for address in level.get("rank_addr_list", []):
            key = (address.get("addr"), tuple(address.get("ports", [])), address.get("plane_id"))
            if key not in seen:
                seen.add(key)
                unique.append(address)
        level["rank_addr_list"] = unique

counts = {}
for rank in data["rank_list"]:
    for level in rank.get("level_list", []):
        counts.setdefault(level["net_instance_id"], set()).add(len(level.get("rank_addr_list", [])))
if any(len(values) != 1 for values in counts.values()):
    raise RuntimeError(f"ranktable still has inconsistent endpoint counts: {counts}")

ranktable.write_text(json.dumps(data, indent=4) + "\n")
