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

### Notes for Write-up

- Primary runtime stack: `grounding_dino+sam2`
- Custom-video qualitative validation succeeded on two user videos.
- Both successful custom runs stayed on the primary SAM2 video path without fallback.
- Observed failure mode on `khanh_sky_uong_nuoc`: when the fridge is behind the person, detection is no longer accurate and tends toward the wrong object category.
- First subset benchmark seed completed end-to-end on 20 selected rows with balanced qualitative tags across easy, occlusion, crowded, and small-object clips.
- The current subset run is suitable as an experiment scaffold, but any formal quantitative table should note the existing duplicate `clip_id` in the manifest summary.
- Official qualitative baseline was completed on a locked 19-clip subset using `Grounding DINO + SAM2` without re-grounding.
- All 19 official baseline runs stayed on `sam2_video_predictor`, so the main qualitative failure modes are now object ambiguity, drift, no-detection, and occlusion-related degradation rather than pipeline fallback.
