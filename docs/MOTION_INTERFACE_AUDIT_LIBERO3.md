# Motion interface audit for LIBERO-10 tasks 0/3/8

Audit date: 2026-08-20 UTC  
Status: data artifact validated; experiment B implementation and training blocked pending explicit approval

## Audited artifacts

- Three-task policy manifest: `/workspace/vla/p3/runtime_metadata/libero10_tasks_0_3_8_policy_aux_manifest.parquet`
  - SHA-256: `dc4c4add349d1d1289cc4b81821a6c79ee1353c37037ed139f87289dda64bf58`
  - 114 episodes and 29,250 policy frames, restricted to LeRobot task indices 0, 3, and 8
  - 28,110 `motion_valid=true` rows and 1,140 invalid rows (the final 10 frames of each episode)
- Motion cache: `/workspace/vla/p3/workspace/data/libero_four_suite_annotation/policy_aux_v1/motion_libero10_tasks_0_3_8_v1`
  - 28,110 unique, finite targets; shape `[28110, 256]`; `float32`
  - cache index SHA-256: `887a65bc8cbccb8d3ea6792999163a3db2eb1eda8c68b549f7eb0436c399c623`
  - 147 pilot-overlap targets reproduce bit-for-bit (`max_abs=0`)
  - full generation used eight workers, completed all selected rows, and reported no missing targets
- Authoritative design handoff: `/workspace/vla/P3_8XA100_MASTER_HANDOFF.md`

## Interface questions and answers

| Question | Audited answer |
| --- | --- |
| Physical meaning | A Track4World internal 3D/temporal motion latent, not an action, optical-flow image, or explicit trajectory. Track4World and Depth Anything 3 jointly process 11 real agent-view frames `t:t+10`; the feature is read from `flow_aggregator3d.global_blocks.3` at source-time index 0 and pooled with current agent-view relevant-entity mask coverage. |
| Shape and dtype | One FP32 vector of shape `[256]` per valid policy-frame anchor; the validated valid-target population is `[28110, 256]`. |
| Temporal alignment | The target is anchored at current policy frame `t`, but is computed after observing real frames through `t+10`. The policy action horizon is 10. Thus the target supervises the same frame anchor as the action chunk while summarizing a future-inclusive 11-frame visual window. |
| Padding and validity | There is no padded or fabricated zero target. The final 10 frames of every one of the 114 episodes are invalid because a full `t:t+10` clip is unavailable. These 1,140 rows remain in the policy manifest with `motion_valid=false` and must contribute exactly zero Motion loss. Only the 28,110 valid targets are stored in the target shards/index. |
| Normalization | Per-dimension mean and population standard deviation were computed over exactly the 28,110 three-task, train-split, valid targets. Accumulation is float64; stored target dtype is FP32; the handoff specifies a `1e-6` standard-deviation floor. Statistics are in `target_statistics_train.json`. |
| Future information | The teacher target intrinsically contains future visual information because Track4World consumes `t+1:t+10`. This is permissible only as an auxiliary training target. It would be leakage if future frames, the target, mask, stage label, or teacher internals entered the policy/action input. No experiment-B DataLoader/model path has yet been implemented, so teacher isolation has not yet passed the required no-leakage gate. |
| Motion query/head in action prefix | Not approved. The handoff proposes, but explicitly does not freeze, eight Motion queries, a per-query LayerNorm followed by query mean and `Linear(2048,256)`, and an action-readable Motion span. Whether Motion enters the action prefix and its attention topology remain human decisions. |
| Loss and reduction | Not approved for experiment B. The probe used standardized `SmoothL1Loss(beta=1, reduction=mean)` as a possible starting point, but the handoff labels it a candidate rather than a frozen policy-training definition. Masked reduction behavior for mixed valid/invalid policy batches must also be approved and tested. |
| `lambda_motion` | No value or approval record exists. The standalone probe has no `lambda_motion`, so no coefficient can legitimately be copied from it. |
| Teacher-free inference | Required design intent: yes. Current A inference is teacher-free. B cannot claim this property until its approved interface is implemented and the serving/no-leakage gates pass. |

## Decision

Experiment B (`Semantic + Geometry + Motion`) must stop before implementation and training. The existing artifact is sufficient to define the Motion teacher target, identity join, validity, and normalization, but it does **not** approve:

1. the Motion query count, head, prefix position, or attention topology;
2. the Motion loss and masked reduction;
3. `lambda_motion`;
4. whether static/moving samples receive equal or stratified weighting.

Explicit approval of those items is required before adding a `semantic_geometry_motion` mode, running its gates, or launching its 8-GPU preflight. Experiment B must initialize independently from the same official pi05 base as experiment A; it must never initialize from A.
