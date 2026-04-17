# Report Asset Manifest

This file locks the report-first asset package so the same tables and figures can be reused in slides without changing the narrative.

## Tables

| id | title | content | slide_use |
| --- | --- | --- | --- |
| T1 | Baseline Summary | Subset size, tag distribution, baseline review counts, one-sentence interpretation | Slide 1 |
| T2 | Ablation Comparison Summary | Ablation 1 vs Ablation 2 improved/same/worse counts, review distributions, one-sentence interpretation | Slide 3 |

## Figures

| id | clip_id | prompt | label_or_delta | why_this_case | slide_use |
| --- | --- | --- | --- | --- | --- |
| F1_baseline_success_small_object | 10360251 | car | good_tracking | Shows that the baseline can still succeed on a small-object case when the target remains visually separable. | Slide 2 |
| F2_baseline_success_easy | 853810 | dog | good_tracking | Clean single-target success; easy to explain the baseline's strongest behavior. | Slide 2 |
| F3_baseline_failure_wrong_object | 12699538 | person | wrong_object | Clear crowd-driven identity switch in a dense overlapping scene. | Slide 2 |
| F4_baseline_failure_drift | 5630823 | dog | drift | Compact example of small-object tracking collapsing toward a distractor. | Slide 2 |
| F5_ablation2_improved | 12699538 | person | wrong_object -> good_tracking | Best example that stricter periodic re-grounding can fully recover a severe baseline failure in a crowded scene. | Slide 4 |
| F6_ablation2_worse | 9910242 | person | good_tracking -> no_detection | Clearest remaining failure showing that even stricter periodic re-grounding still carries risk on some occlusion-heavy clips. | Slide 4 |

## Slide Package

- Slide 1: T1 baseline summary
- Slide 2: F1-F4 baseline success/failure panel
- Slide 3: T2 ablation comparison summary
- Slide 4: F5-F6 Ablation 2 improved-vs-worse panel
- Slide 5: Final discussion takeaway bullets

## Follow-up Recommendation

Do not add a broad new experiment branch. If there is time after the write-up is stable, the only justified follow-up is a trigger-based or confidence-gated re-grounding policy that tries to keep Ablation 2's gains while reducing its remaining regressions.
