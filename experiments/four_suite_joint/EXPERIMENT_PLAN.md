# Four-suite B-Geo0.05 experiment plan

Status: code-release preparation, 2026-08-25. This document defines future
four-suite runs and contains no prior experiment results.

## Scientific question

With identical task-relevant Semantic, Geometry, and Motion auxiliary
supervision, does allowing the Action Expert to read the learned Geometry and
Motion queries improve policy success compared with auxiliary regularization of
the shared backbone alone?

This comparison does not by itself establish improvement over unmodified
pi0.5; no pi0.5 baseline is scheduled in this plan.

## Required runs

| Run | Supervision | Action reads Geometry/Motion | Purpose |
|---|---|---|---|
| B-Geo0.05 Main | Semantic + Geometry + Motion | Yes / Yes | Main four-suite method |
| B-Geo0.05 Supervision-only | Identical | No / No | Isolate latent conditioning from shared-backbone regularization |

Both runs use:

- `lambda_sem=0.01`, `lambda_geo=0.05`, `lambda_motion=0.05`
- no Ground query, target, head, or loss
- eight independent Geometry queries and eight independent Motion queries
- strict FP32-converted pi0.5 initialization
- BF16, no EMA, AdamW, gradient clip 1.0
- global batch 256 on 8 GPUs
- the same seed, sampler, LR schedule, update budget, and evaluation protocol

The only experimental difference is whether Action-to-Geometry/Motion attention
edges are enabled during both training and deployment.

## Data contract

- Exact LeRobot revision: `a4336d589d589045d1c56423ffdf3b88a0e19b1f`
- 1,693 episodes / 273,465 frames / 40 tasks
- Geometry: 273,377 valid targets
- Motion: 256,401 valid targets
- Semantic: present for every frame

Training samples frames with the standard LeRobot sampler. The resulting suite
weights are frame-proportional; evaluation reports every suite separately and a
four-suite macro average.

## Budget candidate

At global batch 256 the population provides about 1,068 updates per epoch. A
30,000-update run is 28.09 epochs and a 10,000-update warmup is 9.36 epochs,
matching the epoch semantics of the reviewed LIBERO-10 recipe. These values are
the recommended candidate, not authorization to launch.

## Evaluation contract

- Freeze the primary checkpoint-selection rule before evaluation.
- Do not choose the paper checkpoint by maximizing the final test rollouts.
- Use paired rollout seeds, task counts, initial states, and environment settings
  across both training variants.
- Report LIBERO-Spatial, Object, Goal, LIBERO-10, and their macro average.
- A Supervision-only checkpoint must be served with both query groups blocked.
- For the Main checkpoint, optional inference interventions may block Geometry,
  Motion, or both, but must be explicitly recorded in the evaluation manifest.

## Launch requirements

Formal training remains blocked until all of the following pass:

1. all 1,693 LeRobot episode parquet files are present;
2. transferred auxiliary targets and FP32 base pass checksum verification;
3. portable metadata preparation and static preflight pass;
4. both variants pass a real batch forward/backward check;
5. both variants pass a short 8-GPU optimizer preflight;
6. checkpoint retention and available storage are frozen;
7. the researcher explicitly approves the formal run.
