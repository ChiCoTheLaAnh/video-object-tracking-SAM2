# Report Outline

- Problem framing
- Baseline approach
- Smoke test
- Quantitative evaluation
- Failure analysis
- Ablation study
- Final discussion

## Validation Log

### D1 Runtime Validation

- D1 smoke test passed on Colab with the primary SAM2 propagation path.
- Image smoke test produced an annotated image and valid `run_summary.json`.
- Video smoke test produced an overlay MP4 and valid `run_summary.json`.
- Confirmed `artifacts.video_mode == "sam2_video_predictor"` on the happy path.

### Stage 1 Custom Videos

- `2026-04-01 | hieu_con_quang_ninh | prompt=person | video_mode=sam2_video_predictor | review=good tracking | pass`
- `2026-04-01 | khanh_sky_uong_nuoc | prompt=fridge | video_mode=sam2_video_predictor | review=wrong object | pass`

### Fixed Subset Benchmark Seed

- `2026-04-13 | subset_manifest.csv | selected=20 | completed=20 | primary_path=sam2_video_predictor | pass`
- Tag distribution for the fixed subset seed: `easy=6`, `occlusion=5`, `crowded=5`, `small_object=4`.
- All completed subset runs used the primary `sam2_video_predictor` video path and wrote overlay artifacts under `results/quantitative/subset_eval/`.
- Current prompt source in the subset manifest is the `notes` column, which is being used as the inference prompt for each selected clip.
- One duplicate `clip_id` (`853810`) exists in the current manifest/output summary; this was left as-is for now and should be treated as a reporting caveat rather than a corrected benchmark row.

### Official Baseline Run

- `2026-04-14 | grounding_dino+sam2_no_regrounding | selected=19 | completed=19 | primary_path=sam2_video_predictor | pass`
- Official baseline tag distribution: `easy=5`, `occlusion=5`, `crowded=5`, `small_object=4`.
- Local reviewed baseline counts: `good_tracking=7`, `partial_tracking=7`, `drift=2`, `wrong_object=1`, `no_detection=1`, `fallback=1`.
- Selected 5 success cases: `10360251`, `853810`, `16436839`, `5730870`, `6326811`.
- Selected 5 failure cases: `11998127`, `12699538`, `5220726`, `5630823`, `6664239`.
- Final export of sample artifacts on Drive was not cleanly finalized in Colab, but the reviewed baseline table and 5/5 success-failure split were completed locally.

### Ablation Runs

- `2026-04-14 | grounding_dino+sam2_periodic_regrounding_every_10 | selected=19 | completed=19 | pass`
- Ablation 1 delta vs baseline: `improved=4`, `same=8`, `worse=7`.
- Ablation 1 review counts: `partial_tracking=18`, `fallback=1`.
- `2026-04-17 | grounding_dino+sam2_periodic_regrounding_every_10_iou_0_3 | selected=19 | completed=19 | pass`
- Ablation 2 delta vs baseline: `improved=9`, `same=6`, `worse=4`.
- Ablation 2 review counts: `good_tracking=12`, `partial_tracking=4`, `drift=1`, `no_detection=2`.

### Notes for Write-up

- Primary runtime stack: `grounding_dino+sam2`
- Custom-video qualitative validation succeeded on two user videos.
- Both successful custom runs stayed on the primary SAM2 video path without fallback.
- Observed failure mode on `khanh_sky_uong_nuoc`: when the fridge is behind the person, detection is no longer accurate and tends toward the wrong object category.
- First subset benchmark seed completed end-to-end on 20 selected rows with balanced qualitative tags across easy, occlusion, crowded, and small-object clips.
- The current subset run is suitable as an experiment scaffold, but any formal quantitative table should note the existing duplicate `clip_id` in the manifest summary.
- Official qualitative baseline was completed on a locked 19-clip subset using `Grounding DINO + SAM2` without re-grounding.
- All 19 official baseline runs stayed on `sam2_video_predictor`, so the main qualitative failure modes are now object ambiguity, drift, no-detection, and occlusion-related degradation rather than pipeline fallback.

## Baseline Results

The official baseline uses `Grounding DINO + SAM2` without any re-grounding step on a fixed 19-clip qualitative subset. The subset is balanced across four coarse difficulty tags: `easy=5`, `occlusion=5`, `crowded=5`, and `small_object=4`. All 19 runs completed successfully and stayed on the primary `sam2_video_predictor` path, which means the baseline is now stable enough to discuss model behavior rather than debugging the runtime stack.

At the qualitative review level, the baseline breaks down into `7 good_tracking`, `7 partial_tracking`, `2 drift`, `1 wrong_object`, `1 no_detection`, and `1 fallback`. This distribution suggests that the main bottleneck is not catastrophic pipeline failure but degraded tracking quality under harder scenes. In other words, the baseline is strong enough on clean targets to serve as a reference point, but it is not yet robust to dense distractors, severe occlusion, and very small targets.

Representative success cases for the write-up:

- `10360251` (`small_object`, prompt=`car`): tracks small cars stably in a wide scene.
- `853810` (`easy`, prompt=`dog`): tracks a single target with low ambiguity.
- `16436839` (`easy`, prompt=`bicycle`): handles a dominant foreground target cleanly.
- `5730870` (`easy`, prompt=`dog`): near-perfect tracking on a clear single target.
- `6326811` (`easy`, prompt=`backpack`): stable tracking in portrait framing with a clean target.

Representative failure cases for the write-up:

- `11998127` (`crowded`, prompt=`person`): `drift` caused by many visually similar distractors.
- `12699538` (`crowded`, prompt=`person`): `wrong_object` when the tracker switches targets in a dense overlapping crowd.
- `5220726` (`occlusion`, prompt=`bicycle`): `fallback`-like failure under heavy mutual occlusion.
- `5630823` (`small_object`, prompt=`dog`): `drift` toward a moving distractor while tracking a small target.
- `6664239` (`crowded`, prompt=`person`): `no_detection` in a dense night scene with heavy motion and distractors.

## Ablation Results

The ablation study kept the same experimental frame throughout: the same locked 19-clip subset, the same prompts, the same checkpoints, and the same periodic schedule of re-grounding attempts at frames `10, 20, 30, 40`. The only thing that changed across ablation variants was the reseeding criterion. This makes the interpretation much cleaner than introducing a new model branch or a different schedule.

The first variant, Ablation 1, used periodic re-grounding every 10 frames with `min_match_iou = 0.1`. That version produced `4 improved`, `8 same`, and `7 worse` outcomes relative to baseline, with review labels collapsing to `18 partial_tracking` and `1 fallback`. This is the negative control result: periodic reseeding with a permissive matching threshold was too aggressive and destabilized many clips that the baseline already handled well.

The follow-up variant, Ablation 2, kept the exact same periodic schedule and max-IoU matching rule but raised the acceptance threshold to `min_match_iou = 0.3`. That single change materially improved the result: Ablation 2 produced `9 improved`, `6 same`, and `4 worse` outcomes relative to baseline. Its review distribution also became much healthier, with `12 good_tracking`, `4 partial_tracking`, `1 drift`, and `2 no_detection`. Compared with Ablation 1, the stricter threshold recovered a large amount of tracking quality instead of collapsing most clips into the middle tier.

The strongest improvement cases are exactly the difficult scenes that motivated re-grounding in the first place. For example, `11998127` moved from `drift -> good_tracking`, `12699538` from `wrong_object -> good_tracking`, `5630823` from `drift -> good_tracking`, and `5220726` from `fallback -> partial_tracking`. These cases suggest that periodic detector refresh can help crowded, occluded, and small-object clips, but only when the reseeding rule is selective enough to avoid latching onto weak or noisy detections.

The remaining regressions still matter. Ablation 2 is not strictly better than baseline on every clip: `11073730` degraded from `partial_tracking -> no_detection`, `16436839` from `good_tracking -> partial_tracking`, `7825225` from `good_tracking -> partial_tracking`, and `9910242` from `good_tracking -> no_detection`. So the main conclusion is not that periodic re-grounding is universally superior. The conclusion is narrower and more defensible: **the failure of Ablation 1 came largely from over-aggressive reseeding, and a stricter matching threshold makes periodic re-grounding a net positive on this subset**.

## Failure Analysis

The 10 reviewed example cases show that the dominant failure modes are semantic ambiguity and target persistence, not infrastructure instability. Because all official baseline runs stayed on `sam2_video_predictor`, the failure analysis can focus on scene difficulty instead of implementation bugs.

The first recurring pattern is **crowd-driven ambiguity**. In clips such as `11998127`, `12699538`, and `6664239`, the prompt `person` is semantically correct but underspecified relative to the scene. When many people are present with similar appearance and motion, the detector can either drift to a neighboring instance, switch identities entirely, or fail to localize a confident starting target. This is the clearest argument for later improvements such as re-grounding or stronger prompt conditioning.

The second pattern is **occlusion-driven instability**. Clips such as `4992551`, `4992557`, `5021553`, and especially `5220726` show that once the target is partially hidden by other actors or framing, the baseline often degrades from `good_tracking` to `partial_tracking`, and in the worst case loses the target entirely. Even when the system does not fully collapse, mask quality and identity consistency degrade noticeably during crossings and mutual overlap.

The third pattern is **small-object fragility**. The contrast between `10360251` and `11073730` for `car`, and between `853810`/`5730870` and `5630823`/`6413967` for `dog`, suggests that small targets are not uniformly hard; they become hard when small scale is combined with distractors or wide framing. The system can track a small object when it remains visually isolated, but performance deteriorates quickly when the same object competes with clutter or secondary motion cues.

The practical conclusion after the ablations is more specific than before. The baseline is already strong enough to anchor the result section, and Ablation 1 showed that permissive periodic reseeding is a bad default. However, Ablation 2 shows that the periodic schedule itself is not the main problem. The real issue is acceptance quality: once reseeding is gated with a stricter `IoU >= 0.3` match, re-grounding becomes useful on many of the hard clips without collapsing the whole subset into `partial_tracking`. That shifts the next experimental direction from "abandon periodic re-grounding" to "make reseeding even more selective or trigger-based."

## Report Tables

### Table 1. Baseline Summary

| subset_size | tag_distribution | baseline_review_counts | interpretation |
| --- | --- | --- | --- |
| 19 | easy=5, occlusion=5, crowded=5, small_object=4 | good_tracking=7, partial_tracking=7, drift=2, wrong_object=1, no_detection=1, fallback=1 | The no-re-grounding baseline is reliable on clean single-target clips but degrades under crowding, occlusion, and small-object settings. |

### Table 2. Ablation Comparison Summary

| variant | improved | same | worse | review_counts | interpretation |
| --- | --- | --- | --- | --- | --- |
| Ablation 1 (`IoU >= 0.1`) | 4 | 8 | 7 | partial_tracking=18, fallback=1 | Permissive periodic re-grounding is too aggressive and pulls most clips into middling quality. |
| Ablation 2 (`IoU >= 0.3`) | 9 | 6 | 4 | good_tracking=12, partial_tracking=4, drift=1, no_detection=2 | Stricter matching turns periodic re-grounding into a net positive on the locked subset. |

## Figure Package

Lock the report/slides figure set to exactly six items so the narrative stays fixed.

- `F1_baseline_success_small_object`: `10360251`, prompt=`car`, label=`good_tracking`. Chosen because it shows the baseline can still succeed on a small-object clip when the target remains visually separable.
- `F2_baseline_success_easy`: `853810`, prompt=`dog`, label=`good_tracking`. Chosen because it is the cleanest single-target success case and makes the baseline strength easy to explain.
- `F3_baseline_failure_wrong_object`: `12699538`, prompt=`person`, label=`wrong_object`. Chosen because it visualizes crowd-driven identity switching clearly.
- `F4_baseline_failure_drift`: `5630823`, prompt=`dog`, label=`drift`. Chosen because it is a compact example of small-object tracking collapsing toward a distractor.
- `F5_ablation_improved`: `12699538`, prompt=`person`, `wrong_object -> good_tracking` under Ablation 2. Chosen because it is the clearest case where stricter re-grounding resolves a severe crowd-driven failure.
- `F6_ablation_worse`: `9910242`, prompt=`person`, `good_tracking -> no_detection` under Ablation 2. Chosen because it is the clearest remaining failure showing that periodic re-grounding still carries risk even with a stricter threshold.

The slide mapping should stay derivative from this report package:

- Slide 1: Table 1 baseline summary
- Slide 2: F1-F4 baseline success/failure examples
- Slide 3: Table 2 ablation delta summary
- Slide 4: F5-F6 ablation improved vs worse examples
- Slide 5: Final discussion takeaway bullets

## Final Discussion

The project now has a coherent result story rather than just a runnable pipeline. The official no-re-grounding baseline establishes that `Grounding DINO + SAM2` is already a credible reference system on the locked subset: it completes all runs, stays on the primary `sam2_video_predictor` path, and produces several genuinely strong qualitative cases on clear single-target clips. That matters because it means later comparisons are not being made against a broken or unstable baseline.

At the same time, the baseline exposes a clear and defensible weakness profile. Performance degrades most often when prompts are underspecified relative to the scene, especially in crowds, partial occlusion, and small-object settings. The failure analysis shows that the core issue is not infrastructure reliability but target identity preservation under ambiguity. This gives the project a clean problem statement for the experimental section: the question is how to recover from drift and wrong-object switches without destabilizing already-good tracks.

The ablation sequence answers that question more precisely than a single experiment could. Ablation 1 showed that naive periodic re-grounding every 10 frames with a permissive `IoU >= 0.1` gate is too blunt. It helped a few severe failures, but the overall `4 improved / 8 same / 7 worse` result showed that over-aggressive reseeding can inject instability into clips that baseline propagation already handled well.

Ablation 2 changes that interpretation in an important way. When the same periodic schedule is kept fixed but the matching threshold is tightened to `IoU >= 0.3`, the result flips to `9 improved / 6 same / 4 worse`, with `12 good_tracking` clips overall. That means the earlier negative result should not be read as "periodic re-grounding is fundamentally bad." The better reading is that **periodic re-grounding is highly sensitive to reseeding quality**, and the original failure came mainly from letting weak detector matches reset the tracker too easily.

The final project-level conclusion is therefore more nuanced than before. The baseline remains a strong and credible reference system: it is simple, stable, and already performs well on many clean clips. But the best result in this project is now the stricter follow-up ablation, not the original no-re-grounding system. On this locked 19-clip subset, periodic re-grounding with a stricter `IoU >= 0.3` acceptance rule is the strongest tested configuration because it improves many hard cases while keeping regressions limited.

The most justified next step is no longer "try stricter matching" because that question has now been answered. The next step would be a trigger-based or confidence-gated re-grounding policy that preserves the benefits of Ablation 2 while avoiding its remaining regressions on clips like `9910242` and `11073730`.
