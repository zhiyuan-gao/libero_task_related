# Final A/B code audit

Date: 2026-08-20

Status: code and dry-run handoff gates pass; formal training not started

## Lineage

- P2 to A: remove the Grounding mode, eight Ground queries, Ground head/data/loss,
  and `lambda_ground=0.50`. The P2 joint-masked Semantic path and Geometry path
  are shared. The only non-trajectory difference is checkpoint retention
  (`keep_period=5000` in legacy P2 versus 1000 in A/B).
- A to B: add only Motion mode metadata, eight Motion queries, Motion cache paths,
  and `lambda_motion=0.05`. Model config, policy data, optimizer, LR schedule,
  strict FP32 base, BF16 training precision, batch size, update count, and no-EMA
  protocol are identical.

## Frozen B computation

The PaliGemma-side training order is
`Context|Geometry(8)|Motion(8)|SemanticTeacher`; Action is the separate expert
suffix. Geometry and Motion each read Context plus their own query group and
cannot read each other. Semantic reads Context plus its causal teacher prefix.
Action reads Context, Geometry, and Motion, but cannot read SemanticTeacher.
Motion targets never enter the forward input.

The total loss is:

`L = L_action + 0.15 L_geometry + 0.01 L_semantic + 0.05 L_motion`

Motion uses train-standardized Smooth-L1 (`beta=1`) over valid samples and 256
dimensions. The 1,140 invalid episode-tail rows contribute exact zero.

## Gates completed

- 28 focused model/data/config/handoff tests pass.
- Attention tests cover query isolation and SemanticTeacher non-leakage.
- The real strict-FP32 model accepts only the ten expected new Geometry/Motion
  state keys; there are no Grounding state keys.
- One real BF16 batch completed forward and backward without an optimizer step.
  Action, Geometry, Motion, and base VLM parameters all received finite nonzero
  gradients. Peak allocated CUDA memory was 21,974,238,720 bytes.
- The measured one-batch loss exactly matched the frozen weighted formula.
- Sequential handoff tests prove B starts after A returns zero, B does not start
  when A fails, and formal orchestration refuses to run without explicit approval.
- Both A and B independently resolve the same strict FP32-converted base; B never
  consumes an A checkpoint.

The 8-GPU formal A/B launch remains blocked by the explicit confirmation guard.
