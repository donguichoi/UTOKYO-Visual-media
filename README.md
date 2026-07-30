# UTOKYO-Visual-media
# HaMeR Failure Analysis and Candidate-level NMS Improvement

Visual Media, Report Assignment
Dongui CHOI (26-37265119)
Department of Electrical Engineering and Information Systems,
Graduate School of Engineering, The University of Tokyo — Someya Lab.

**Target paper:** Pavlakos et al., *Reconstructing Hands in 3D with Transformers* (HaMeR), CVPR 2024
Original repository: https://github.com/geopavlakos/hamer

## Summary

The official HaMeR demo was run on 7 self-captured hand images. Two failure
modes were observed: complete detection failure and duplicate detection of a
single hand. Stage-by-stage logging was added to the demo pipeline to identify
where each failure occurs. The duplicate detection problem was then addressed
with candidate-level IoU NMS, and the resulting hand-side misclassification was
corrected by changing the candidate ranking criterion.

**Key finding:** both failure modes originate in the input construction stage,
not in HaMeR's mesh regression. Person detection succeeded on all 7 images
(score >= 0.995). Every observed failure is traceable to hard-coded constants
in the demo pipeline.

All measurements were reproduced identically across two independent runs.

## Diagnosis

| image | persons | L valid kp | L mean conf | R valid kp | R mean conf | L-R IoU | to HaMeR |
|---|---|---|---|---|---|---|---|
| ambiguous_hand_1 | 1 (0.995) | 2/21 | 0.603 | 2/21 | 0.563 | – | 0 |
| fist_closeup_1 | 1 (0.999) | 3/21 | 0.529 | 0/21 | – | – | 0 |
| fist_closeup_2 | 1 (0.999) | 0/21 | – | 0/21 | – | – | 0 |
| normal_hand_1 | 1 (0.999) | 21/21 | 0.703 | 11/21 | 0.610 | 0.795 | 2 |
| normal_hand_2 | 1 (0.999) | 21/21 | 0.796 | 20/21 | 0.641 | 0.889 | 2 |
| palm_closeup_1 | 1 (0.999) | 7/21 | 0.556 | 5/21 | 0.561 | 0.696 | 2 |
| two_hands_overlap | 1 (0.999) | 2/21 | 0.563 | 0/21 | – | – | 0 |

A hand bounding box requires more than 3 keypoints with confidence > 0.5.
Note that the confidence of the keypoints that did pass was not particularly
low (0.53–0.60); the immediate cause of failure was the count condition, not
the confidence values.

## Results

| image | baseline | NMS (v1) | NMS (v2) | side v1 | side v2 |
|---|---|---|---|---|---|
| ambiguous_hand_1 | 0 | 0 | 0 | – | – |
| fist_closeup_1 | 0 | 0 | 0 | – | – |
| fist_closeup_2 | 0 | 0 | 0 | – | – |
| normal_hand_1 | 2 | 1 | 1 | Left | Left |
| normal_hand_2 | 2 | 1 | 1 | Left | Left |
| palm_closeup_1 | 2 | 1 | 1 | **Right (wrong)** | **Left (correct)** |
| two_hands_overlap | 0 | 0 | 0 | – | – |

On `palm_closeup_1` the two candidates differed by only 0.005 in mean keypoint
confidence, which caused v1 to select the wrong hand side. Ranking by the number
of valid keypoints instead corrects this case while leaving the other two
unchanged.

## Modification

Candidates are collected instead of being committed immediately, then filtered
by IoU. See `src/demo_nms.patch` for the full diff against the official `demo.py`,
and `src/v1_to_v2.patch` for the one-line ranking change.

```python
candidates = []
for side_flag, keyp in ((0, left_hand_keyp), (1, right_hand_keyp)):
    valid = keyp[:, 2] > 0.5
    if sum(valid) > 3:
        bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                keyp[valid, 0].max(), keyp[valid, 1].max()]
        candidates.append({'bbox': bbox, 'is_right': side_flag,
                           'n_valid': int(sum(valid)),
                           'conf': float(keyp[valid, 2].mean())})

filtered = []
for c in sorted(candidates, key=lambda c: (c['n_valid'], c['conf']), reverse=True):
    if all(_iou(c['bbox'], k['bbox']) < 0.5 for k in filtered):
        filtered.append(c)
```

The IoU threshold of 0.5 is based on the measured range above (0.696–0.889).

## Files

| path | description |
|---|---|
| `notebooks/` | Full Colab notebook with execution outputs |
| `src/demo_debug.py` | Stage-by-stage logging. Behavior identical to baseline. |
| `src/demo_nms.py` | Improvement 1: candidate-level IoU NMS |
| `src/demo_nms_v2.py` | Improvement 2: ranking by valid keypoint count |
| `src/*.patch` | Unified diffs against the official `demo.py` |
| `data/` | 7 self-captured input images |
| `results/official/` | Official example images, pipeline sanity check |
| `results/baseline/` | Baseline outputs (duplicate meshes visible) |
| `results/nms_v1/`, `results/nms_v2/` | Outputs after each improvement |
| `results/*.txt`, `results/*.csv` | Execution logs and measurement tables |

## Reproduce

Google Colab, GPU runtime. Clone the official repository, place the pretrained
checkpoint and MANO model files, then run each script with
`--side_view --save_mesh --full_frame`.

MANO model files and the pretrained checkpoint are **not included** in this
repository due to their license terms. Obtain them from the official HaMeR
`fetch_demo_data.sh` script and the MANO distribution site.
