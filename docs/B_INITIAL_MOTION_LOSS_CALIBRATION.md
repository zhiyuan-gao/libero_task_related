# B initial Motion loss calibration

Date: 2026-08-20

Status: PASS (engineering scale check; no formal training started)

Protocol:

- Config: `pi05_libero3_semantic_geometry_motion_aux`
- Base: strict FP32-converted `pi05_base` checkpoint
- Model precision: BF16 training architecture
- Data: 32 shuffled real samples from LIBERO tasks 0, 3, and 8
- Motion target: train-standardized float32[256]
- Loss: Smooth-L1, beta 1.0, mean over valid samples and dimensions
- Valid Motion targets: 31/32; the invalid episode-tail sample contributed exactly zero
- Motion coefficient: 0.05 (researcher-confirmed after raw-scale review)
- No optimizer step, checkpoint write, or formal training

Initial raw loss means:

| Loss | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Action | 0.223418 | 0.237201 | 0.140600 | 0.261464 |
| Geometry | 8.844799 | 8.869881 | 8.568176 | 9.161568 |
| Semantic | 1.677245 | 1.748460 | 1.017744 | 2.173655 |
| Motion | 2.614148 | 2.633092 | 2.362752 | 2.767323 |

At coefficient 0.05, the initial weighted Motion contribution is 0.130707.
For comparison, the mean weighted Geometry and Semantic contributions are
1.326720 and 0.016772 respectively. Motion therefore begins at about 59% of
the Action loss and 10% of the weighted Geometry contribution.
