# Motion interface audit for LIBERO-10 tasks 0/3/8

Audit date: 2026-08-20 UTC  
Status: interface approved and implemented; formal training remains blocked pending final researcher confirmation

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
| Future information | The teacher target intrinsically contains future visual information because Track4World consumes `t+1:t+10`. It enters only the loss target. Motion queries are learned parameters, and neither future frames nor Motion targets enter the policy forward input. Joint-mask tests verify that SemanticTeacher is hidden from Action, Geometry, and Motion. |
| Motion query/head in action prefix | Approved and implemented as eight Motion queries followed by per-query LayerNorm, query mean, and `Linear(2048,256)`. Motion reads Context plus itself; Geometry and Motion cannot read each other; Action reads both. The prefix order is `Context|Geometry|Motion|SemanticTeacher`, with Action as the separate expert suffix. |
| Loss and reduction | Approved and implemented as train-standardized `SmoothL1Loss(beta=1, reduction=mean)` over valid samples and all 256 dimensions. Invalid samples are excluded; an all-invalid batch returns a differentiable exact zero. Static and moving valid samples have equal weight. |
| `lambda_motion` | Researcher-confirmed at `0.05` on 2026-08-20 after reviewing a 32-sample real-data initial raw Motion loss of 2.614148. The corresponding initial weighted contribution is 0.130707. |
| Teacher-free inference | Yes. Inference appends only learned Geometry and Motion queries and never accepts Semantic or Motion teacher targets. |

## Decision

Experiment B is frozen as the current experiment A implementation—P2 joint-masked
Semantic and Geometry with Grounding completely removed—plus the isolated Motion
branch above. A and B use the same data population, optimizer schedule, strict
FP32-converted base, BF16 training precision, global batch 256, 3,209 updates,
and RAW/no-EMA checkpoints. B initializes independently from the same base as A;
it never initializes from A's trained checkpoint. Formal A/B training remains
blocked until the researcher gives the requested final confirmation.
