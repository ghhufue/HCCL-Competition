# HCCL Broadcast Flat-Architecture P0/P1 Optimization Prompt

You must directly modify the HCCL Broadcast finals implementation in the current repository and complete the P0 and P1 optimizations below. Do not stop at analysis or pseudocode. Read the actual code, implement the changes checkpoint by checkpoint, compile the project, and validate each checkpoint.

## 1. Project Background and Fixed Baseline

Keep the current large-message algorithm as a flat architecture with N-way partitioning and contiguous owner blocks:

1. Split the complete buffer into N contiguous owner blocks, where N is the number of ranks.
2. Each non-root owner uses receiver-pull to fetch its block from the root.
3. As soon as an owner's tile becomes ready, the owner writes it directly to every other rank.
4. The root owner's block is already local and requires no seed pull.
5. Do not replace this design with hierarchical Broadcast, Ring, Parallel, or OmniPipe.
6. Do not restore the Direct parallel-write path.
7. Keep `enablePushBatchMerge=false` for now. Never delay a ready batch merely to wait for future tiles to merge.
8. Continue to handle partial tail tiles correctly, including cases such as `400 MiB + 4 B`.

Main source directories and files:

```text
Hccl_Broadcast_Final/
|-- op_host/broadcast.cc
|-- op_host/exec_op.cc
|-- op_host/exec_op.h
|-- op_kernel_ccu/ccu_kernel.cc
|-- op_kernel_ccu/ccu_kernel.h
`-- include/common.h
```

Before making changes, read `AGENTS.md`, the README, the problem statement, the validation scripts, and the source files listed above. The current code is the source of truth. If an optimization is partially implemented, first determine whether it is complete and satisfies the acceptance criteria. Do not add a second overlapping mechanism.

## 2. Non-Negotiable Constraints

- Preserve the current data layout, owner mapping, and receiver-pull to owner-write data path.
- Do not introduce a global cross-die CCU barrier. Do not assume that `CcuLocalNotify`, CCU events, or event tags work across dies.
- Do not obtain correctness by synchronizing the stream for every tile or joining host threads for every batch.
- Do not use `pushWindowDepth=1` as the final fix.
- Do not enlarge the tile size to hide a Ready Ring reuse problem.
- Do not wait for additional ready tiles solely for batch merging.
- Do not modify the small-message algorithm unless shared code makes it necessary. If it is modified, run separate small-message regression tests.
- Preserve all unrelated user changes in the worktree.
- Do not use local HCCL-VM runtime as evidence of physical performance. Local testing is mainly for correctness, dependency validation, and deadlock detection.

## 3. Implementation Process

Work through independent checkpoints in this exact order:

```text
P0-1 -> P0-2 -> P0-3 -> P1-1 -> P1-2 -> P1-3
```

For every checkpoint:

1. Explain the current bottleneck and identify the functions that will change.
2. Implement only the current checkpoint and avoid unrelated refactoring.
3. Compile and run all available checkers and regression tests.
4. Report the actual changes, validation results, and remaining risks.
5. If the checkpoint fails, diagnose and fix it before continuing.
6. Keep the code in a state that can be compared with or reverted to the previous checkpoint.

If the environment cannot run the complete test suite, at minimum perform static inspection and compilation. Clearly identify missing validation and do not claim that unexecuted tests passed.

## 4. P0: Implement First

### P0-1: Push the Root Owner Block on Both Dies Immediately

Current problem: when `rankId == rootRank`, the owner block is already in the local user buffer. It needs no seed pull, and the second die should not wait for a complete-block pull before it can begin pushing.

Implementation requirements:

- Add an explicit fast path for the root owner.
- Skip the seed read and unnecessary Ready Ring production.
- Start push kernels on both active dies concurrently as early as possible.
- Allow both dies to execute `RunPush<false>()` directly on the local data.
- Preserve the required owner and global completion steps. Do not return early.
- Degrade naturally on a single-die topology. Do not hard-code two dies.

Acceptance criteria:

- The root-owner path issues no remote read.
- The active dies do not run in a serialized pattern where one die completes its entire push before the other starts.
- The implementation is correct and deadlock-free with 2, 4, 8, 12, and 16 ranks.
- Active dies are selected correctly when the root is located on different ranks or dies.

### P0-2: Start Non-Seed-Die Pushes in a Tile or Segment Pipeline

Current problem: the seed die can pull and push tiles incrementally, but the non-seed die waits until the complete owner block has been pulled. Mesh and Clos therefore cannot operate concurrently from the first ready data.

Target schedule:

```text
Seed finishes pulling tile or segment k
    |-- seed die pushes k
    `-- non-seed die pushes k

At the same time, Seed pulls k+1 under bounded credit/window control.
```

Implementation requirements:

- Prefer a tile-level pipeline, but first prove that the selected ready/credit primitive has valid semantics on both dies.
- Never make both dies wait on the same die-local event.
- If the runtime has no reliable cross-die tile-ready primitive, use Host Thread Notify to implement a bounded-depth segment pipeline instead of waiting for the entire owner block.
- A segment should contain several contiguous tiles. Begin with double buffering or 2 to 4 segments. Derive segment sizes from the existing tile and owner-block sizes, including partial tails.
- While the non-seed die pushes segment k, Seed must be able to produce segment k+1.
- Every reusable segment or Ready slot must have explicit producer, consumer, and credit states. Seed may reuse the slot or overwrite corresponding staging data only after both push dies finish consuming it.
- Do not reuse a Thread Notify identifier in a way that allows Record/Wait operations from different generations to overwrite each other. Multiple windows or segments must be distinguishable by generation or slot.
- Do not perform a host thread join for every tile.

Before coding, create a short state table that covers at least the following:

| State | Seed | Seed-die push | Non-seed-die push | Slot reusable? |
|---|---|---|---|---|
| PRODUCING | Pull current segment | Wait for ready | Wait for ready | No |
| READY | Publish completion | May push | May push | No |
| ONE_DONE | Do not overwrite | One side finished | Other side unfinished | No |
| BOTH_DONE | Receive credit | Finished | Finished | Yes |

Acceptance criteria:

- The non-seed die begins pushing before the complete owner block has been pulled.
- Seed pull, seed-die push, and non-seed-die push overlap across at least two adjacent segments.
- Ready/credit signals are neither lost nor consumed by the wrong generation, and slots are never reused early.
- At least 20 consecutive Broadcast calls complete without timeout or stale-signal contamination.
- `TileCount=1`, the tail tile, a single-die topology, and the root-owner path all degrade correctly.

### P0-3: Convert the Write Window into a True Rolling Window

Current problem: the current control flow is close to "submit Depth batches, wait for all Depth batches, then submit the next group." This periodically drains the queue.

Target control flow:

```text
Submit B0(slot0)
Submit B1(slot1)
Wait B0(slot0) -> immediately submit B2(slot0)
Wait B1(slot1) -> immediately submit B3(slot1)
...
Drain only the slots that remain in flight at the end.
```

Implementation requirements:

- Track whether each event slot is in flight and which batch/offset it represents.
- Before reusing a slot, wait only for the previous batch in that slot, then immediately submit the next batch.
- Remove fixed groups that drain every event slot together.
- Ensure correctness for `Depth=1`, `Depth=2`, and `Depth=4`. Keep the current default value for now.
- Clearly separate Ready Ring credit from write completion. Never return credit while either push side may still read the corresponding ready generation.
- Correctly drain a tail containing fewer than Depth batches.
- First implement and verify the merge-disabled path. Do not rewrite opportunistic merging yet.

Acceptance criteria:

- During steady state, completion of one slot immediately permits submission of the next batch.
- No control flow forces all window slots to drain after every Depth batches.
- `Depth=2` and `Depth=4` run repeatedly without cross-overwriting event slots.
- Draining occurs only at the final tail.

## 5. P1: Implement Only After P0 Is Stable

### P1-1: Select Channels by Link Capability and Die Load

Current problem: `QueryBestCcuLinkToPeer()` primarily selects by protocol and endpoint address. Address ordering provides deterministic symmetry but does not indicate the best bandwidth.

Implementation requirements:

- Enumerate available links and collect every property exposed by the actual API, including protocol, local die, network layer, port group, and `ENDPOINT_ATTR_BW_COEFF` when available.
- Confirm the real HComm attribute-query API and return semantics before implementation. Do not invent APIs.
- The selection priority must consider at least availability/protocol, bandwidth coefficient, expected die load, and a stable symmetric tie-break.
- Both endpoints must be able to independently select the same physical link. Do not use local accumulated state that differs between endpoints.
- If dynamic die-load selection breaks endpoint agreement, replace it with a static load assignment derived from the rank pair, topology properties, and a globally reproducible rule.
- Route an intra-server peer through Clos only if RankGraph actually exposes a valid Clos link for that peer. Never assume such a link exists.
- Preserve a stable fallback when `BW_COEFF` is unavailable, and log enough diagnostic information to explain the decision.
- For every peer, log the selected protocol, layer, local die, bandwidth coefficient, and tie-break reason.

Acceptance criteria:

- Both endpoints select the same link.
- Endpoint address is no longer the primary performance heuristic.
- Valid channels are created on 8+4, 2x8, and single-die topologies.
- The original compatible fallback remains reliable when no high-speed alternative exists.

### P1-2: Preserve Small Tiles with 4 Ranks and Fix Cyclic Ready Ring Reuse

Current problem: to avoid a Ready Ring dependency cycle with 4 ranks, `BuildExecutionPlan()` enlarges the tile size until the owner block fits within eight Ready slots. This delays the first push and reduces pipeline granularity.

Implementation requirements:

- Remove the temporary policy that enlarges tiles so the entire owner block fits in one Ready Ring.
- Preserve the configured baseline tile size, such as 4 MiB, and allow one owner block to span multiple Ready Ring rotations.
- Implement a correct credit/reuse protocol for every Ready Ring generation.
- Seed may reuse a slot only after every push path that consumes it has completed.
- Resolve checker dependency cycles without disabling pipelining or forcing depth to 1.
- Test owner blocks containing fewer than, exactly, and more than eight tiles.

Acceptance criteria:

- With 4 ranks and 512 MiB, the tile size is no longer forced to approximately 16 MiB.
- Ready slots can rotate through multiple generations without deadlock.
- Tail tiles and a final Ready Ring generation containing fewer than eight slots drain correctly.

### P1-3: Split Each Die into Two Independent Peer Groups

Current problem: every peer on a die shares one combined completion mask. The slowest peer blocks reuse of the entire batch slot, causing head-of-line blocking.

Implementation requirements:

- Initially split each die into exactly two peer groups. Do not implement fully independent per-peer scheduling yet.
- Use a deterministic and stable grouping rule that balances expected byte volume and link capability when possible.
- Maintain an independent completion mask, event window, batch offset, and progress state for each group.
- After one group completes, it may immediately refill its rolling window without waiting for the other group.
- Check hardware and VM limits for events, threads, and notifies before allocating more resources.
- The two groups write the same owner tile to disjoint target peers. Do not duplicate or omit a destination.
- Degrade naturally when a die contains only one peer or one non-empty group.

Acceptance criteria:

- A slow group does not block the fast group from reusing its own event slot.
- Every destination rank receives exactly one copy of the corresponding owner data.
- Odd peer counts, a die with no peers, and tail batches are handled correctly.

## 6. Unified Validation Matrix

Cover at least the following dimensions supported by the repository. If script names differ, inspect the repository and use the real commands.

- Rank counts: 2, 4, 8, 12, and 16.
- Roots: rank 0, a boundary rank, and a representative cross-die rank, especially root 7.
- Message sizes: 4 B, 512 KiB, just above the algorithm threshold, 4 MiB, 4 MiB + 4 B, 64 MiB + 4 B, 400 MiB + 4 B, and 512 MiB.
- Window depths: 1, 2, and 4, with emphasis on 2 and 4.
- Repeated calls: at least 20 consecutive runs for critical 12-rank and 16-rank cases.
- Checks: build, checker, data correctness, timeout, out-of-bounds access, duplicate or missing writes, and Notify/Event resource conflicts.

After every checkpoint, provide a concise result table:

| Case | Build | Checker | Data | Repeated runs | Conclusion |
|---|---|---|---|---|---|

## 7. Code Quality and Diagnostics

- Add useful comments for critical state machines, slot lifetimes, and cross-die fallback paths. Do not comment every obvious line.
- Every new configuration option must have a default, range validation, and logging.
- Give counters explicit units such as bytes, tiles, batches, and segments.
- Use integer types wide enough for every offset and length. Check addition and multiplication for overflow.
- Do not hide synchronization bugs with large fixed delays or excessive timeouts.
- Diagnostic logs must reconstruct rank, root, die, segment/tile, Ready slot, event slot, peer group, and phase.
- Performance changes must preserve the asynchronous semantics of Broadcast.

## 8. Final Deliverables

After completing P0 and P1, provide:

1. A change summary grouped by P0-1 through P1-3.
2. The modified files and core functions.
3. Before-and-after descriptions of the data path, signal path, and rolling-window progression.
4. Complete validation results and a list of untested cases.
5. Remaining risks and recommended parameters for the next online submission.
6. An explicit list of P2/P3 items that were not implemented.

Start now. First inspect the current code and validation workflow, then determine which P0/P1 items are missing and which are only partially implemented. Begin actual implementation with P0-1. Do not implement P2/P3, and do not change the flat algorithm architecture.
