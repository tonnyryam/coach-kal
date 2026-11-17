# Coach Kal implementation guide

Build an offline water polo film-review pipeline. Start with the existing RF-DETR Base checkpoint, a padded field-of-play polygon, ByteTrack, and a separate ball association pass. Deliver a timestamped review with player tracks and candidate shots and goals. Add passes, possession, saves, exclusions, and spatial analytics as their input signals become reliable.

## Recover the project and reproduce the baseline

- Inspect the local Court Vision and CVAT copies read-only when available. Inventory local scripts, checkpoints, dataset exports, game provenance, and CVAT tasks or volume backups. The [pinned repo](https://github.com/tonnyryam/coach-kal/tree/03ec4f1737e17a51e965c089d9fa4f6af767e91f) verifies frame extraction and training. The user reports additional tracking, homography, event heuristics, and labeling work on the laptop. Reuse recovered implementations that pass the comparisons below.
- The existing extractor samples about 3 fps at 720p, filters sequential near-duplicates with pHash, and uses CLIP and KMeans to select up to 200 frames per video. Add a manifest with match, angle, source timestamp, frame index, and image path before new extraction. Preserve native pixels for small caps and balls. Combine representative frames with known failures and random controls.
- The historical trainer uses Base, 25 epochs, batch size 4, and learning rate 1e-4. JupyterHub had 210/20/10 COCO images. The checkpoint exposes ball, player, and referee. Verify whether older exports or checkpoints contain a separate goalie label. Start locally. For training use university GPU compute, smoke-test the environment, stage data to scratch, and write to a unique output directory. The Jupyter P100 previously worked with PyTorch 2.5.1/cu118. Confirm the installed RF-DETR export signature before using the Slurm launcher.
- The local inputs are Segment_00002.MP4 (1080p/100 fps, evening, low angle) and PP vs AF - Angle 1.mp4 (1080p/30 fps, daytime, elevated). Their split provenance is unknown. The broader archive contains hundreds of hours and existing event tags. Obtain a small matching video/tag export before labeling events by hand.

From your local pilot directory, reproduce the existing RF-DETR comparison in a fresh scratch copy so the original evidence stays available.

```sh
pilot_dir="$PWD"
trial_dir="$(mktemp -d)"
cp -R batch/rfdetr-local "$trial_dir/"
"$pilot_dir/.cv-venv/bin/python" "$trial_dir/rfdetr-local/run.py" "${CKPT:?Set CKPT to the local checkpoint path}"
"$pilot_dir/.cv-venv/bin/python" "$trial_dir/rfdetr-local/make_report.py"
printf '%s\n' "$trial_dir/rfdetr-local/index.html"
```

Review the generated RF-DETR report and the geometry and temporal reports in your local pilot directory.

## Evidence and choices

| Existing trial | What to carry forward |
| --- | --- |
| Generic pose on one hard frame | Found 3–4 of seven provisional heads. Person boxes covered six, but arms made the top of a box unreliable as a head anchor. |
| Pose geometry on 12 stills | Full-frame 12 matches and 67 extras. Polygon 12 and eight. Padded crop 16 and 10. Four tiles 43 and 13. Head-specific SWIM-360 transferred poorly. |
| Trained RF-DETR on 12 stills | Full and tiled boxes both covered 88/91 provisional cap points. Head proxy matched 48/91 and 51/91. Both located four provisional visible balls. Full emitted 149 player and eight ball marks versus 173 and 11 tiled. Median CPU inference 84 vs 327 ms. These are coverage and candidate counts with incomplete reference labels. |
| User review | At evening 4:52, two visible players are missing from the reference, and the person nearest goal is the goalie. Evening crops admitted deck people and officials. Inside-water detections usually matched when made, with some misses. |
| Temporal probes | Yellow catches lane hardware and performs poorly in daytime. Flow can drift on pans despite plausible polygon overlap. Prior tracking lacked verified IDs. |

## Decision-ordered tests

1. **Benchmark.** Use the existing stills and short pilot clips for development. Freeze a 20–30 second clip from a confirmed unseen match for comparison. Add a second clip only for uncovered cases such as a cut, a crowded crossing, edge goalie, or ball release. Group all angles by match. Cache detections by source timestamp, checkpoint, model input, and crop transform. Keep development and holdout labels separate.
2. **Geometry.** Compare full frame, full frame plus padded polygon, and one padded crop with the same post-filter in full-frame coordinates. Seed from visible ropes, goal lines, and goalpost bases. Test propagation on a pan and reset at a cut. Widen or bypass an unreliable polygon. Apply player gating independently of ball and referee outputs. Measure edge/goalie misses, deck false positives, and end-to-end time. Train landmarks only if manual calibration becomes a bottleneck.
3. **Players and caps.** Compare player coverage and cap-center error independently. Try recovered labels or compatible weights first. If the head proxy limits team or role decisions, compare a crop-level cap detector inside player boxes with direct detection over the padded field. The direct variant can recover missed players. Start with 30–50 corrected training frames only if needed. If existing player boxes support a keypoint experiment, use one cap-center click and visibility per instance instead of recreating boxes.
4. **Tracking and roles.** Run ByteTrack at 5 and 10 fps from the same cached source frames and score shared timestamps. Include low-score candidates for association. Verify a few identities before and after overlap or disappearance. Replay one failure with corrected detections to locate its source. If necessary try 25 fps and one BoT-SORT or selective SAM2 challenger. Reset at cuts. Vote on good cap crops over each track, using goal position as corroboration. Leave uncertain team and role unknown. Use offense–defense proximity as context without enforcing pair assignments.
5. **Ball.** Compare highest score with confidence, size, and motion gating. Test 10 and 25 fps on the same release windows. Score visible-ball recovery, false locks, longest gap, and releases recovered. Distinguish observed, estimated, hidden, offscreen, and uncertain. If needed, test a targeted native-resolution pass near goal or the predicted path with full-frame reacquisition. Try ball-only fine-tuning or temporal models if that still leaves event-critical misses.
6. **Events.** Compare video-only candidates with available log assistance on one tagged game. Use ball separation and motion for release candidates, then goal-mouth evidence, scoreboard changes, and restarts. Keep ambiguous saves and goals reviewable. Align logs by period and game-clock anchors with stoppages, replays, and repeated clock ticks. Measure visually verified event recall within ±3 seconds, false clips per game, and review time. Export timestamps, evidence, and short clips. Surface calibration can support zones and relative positions. Airborne ball depth requires additional evidence.

## Annotation and acceptance

- First reuse verified CVAT boxes, tracks, and event exports. Round-trip a single CVAT job before bulk work. Preserve video tracks and keyframes for evaluation and COCO boxes for training.
- Verify the two missing caps at evening 4:52 and the edge goalie. On the short holdout, label visible cap centers, a few identities across difficult spans, visible ball centers, and field landmarks at major view changes. Mark ambiguous observations unknown. Evaluate on human-reviewed labels independent of the predictions under test.
- Only label water-polo-specific examples after an ablation identifies the bottleneck. Correct local prelabels for training and actively scan for missed objects. Use tight cap boxes for a cap detector, sparse track keyframes for identities, and two to four short release sequences for ball timing. Compare correction minutes against a small from-scratch sample.
- Report counts by view, model and tracker time per video minute, and reviewer minutes. Pilot tie-breaker: prefer the simpler/faster variant within one extra ordinary miss per 20 visible players, with no new edge or goalie failure and no loss of reviewable ball releases or event clips. Verify a decisive ambiguous case before expanding labeling or compute.

## Prior art to reuse

- [Roboflow soccer](https://github.com/roboflow/sports/tree/main/examples/soccer) for detector/tracker separation, team-crop classification, and keypoint calibration. Test cap color and track voting before adding embeddings.
- [Roboflow basketball](https://roboflow.com/blog/identify-basketball-players) and [basketball analysis code](https://github.com/abdullahtarek/basketball_analysis) for team and ID ideas. Roboflow's SAM2 example assumed all players were visible in the first frame and reported roughly 1–2 fps on a T4. Use it selectively for difficult tracklets or annotation propagation. Read cap numbers only from sufficiently sharp crops.
- [Roboflow ball tracking](https://roboflow.com/blog/tracking-ball-sports-computer-vision) for candidate continuity and slicing, [ReelCut](https://roboflow.com/blog/make-automatic-highlight-reels-from-soccer-games) for cached observations and ranked, clickable moments, and [TrackNetV4](https://tracknetv4.github.io/) for a conditional temporal challenger.
- [Felsen et al.](https://openaccess.thecvf.com/content_ICCV_2017/papers/Felsen_What_Will_Happen_ICCV_2017_paper.pdf) for water polo heads and pool landmarks. [Berkeley currently offers one game](https://pannaf.github.io/posts/sports-forecasting/water-polo-data) under noncommercial research and education terms. [Roboflow Universe](https://universe.roboflow.com/water-polo-tracking) has cap-number, pool-landmark, and player/ball data. Confirm provenance, footage overlap, license, annotation semantics, and local weight availability before using external data.
