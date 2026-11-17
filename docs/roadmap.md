# Coach Kal implementation roadmap

Offline water polo film review. First reliable player, goalie, team, and ball tracking, then a timeline of candidate shots and goals you can click to review. Prepared 16 November 2026.

## Context

**Why this plan exists.** You have an RF-DETR checkpoint, a small pilot, hundreds of hours of Hudl-tagged film, and university GPUs, but no working tracking or event pipeline. The priorities, in order, are accuracy on unseen games, your annotation minutes, and time to a useful result.

**What it is built from.** The implementation guide and pilot folder (16 Nov), the pasted historical handoff, the public repo at commit `03ec4f1` (five files, and `main` has not moved since), read-only checks I ran today on this Mac, and web research. Where the handoff and the pilot differ, the pilot evidence is used.

**What changed the plan.**

1. The extra Court Vision and CVAT work is not on this Mac. You confirmed it is on another laptop. Step 0 recovers it and nothing else waits on it.
2. Your tags are in the Hudl online library and no export has been pulled. A one-game export test gates everything that depends on tags.
3. The checkpoint has no goalie class and its `player` boxes cover the whole swimmer, so team and role have to come from caps. A generic pose model run on player crops found only 56 to 67 of 91 caps today. A cap detector is therefore the first training labels this plan asks for.
4. With real RF-DETR detections, tracks still fragment: 21 to 52 IDs for about 11 players in 15 seconds. The pilot's short track buffer was not the cause. Trackers that compensate for camera motion fragment least, but choosing one needs identity ground truth.
5. Accuracy on unseen games cannot be judged from one or two clips. The benchmark is spread thinly across eight unseen games instead.
6. For candidate shots and goals, the evidence favors a model trained on your existing tags, with whistle, stoppage, and goal-area cues as support. Ball tracking supplies evidence and ranking, not the only source of candidates.

**Outputs per game.** `detections.parquet`, `tracks.parquet` (track ID, team, role), `ball.parquet` (position and state), `events.json` (ranked candidates with evidence), and `review.html` (click a candidate to seek the video).

## Start here

This document is a roadmap. It contains no pipeline code. Section 4 lists each file to write, what it does, and which pilot code it reuses. That section, together with sections 13 and 14, is the brief for whoever writes the code, whether that is you or a coding assistant opened in the `coach-kal` checkout.

**First weeks, in order.**

1. Hudl export test (Step 0c, about 45 minutes). Do this first because it decides the most.
2. Other-laptop inventory (Step 0a, one hour at most) and hub inventory (Step 0b, about 15 minutes). Both are copy-and-paste commands in section 13.
3. Choose and download the eight test games (Step 0d), then set up the project (Step 0e, about 10 minutes).
4. Build v0 and run it on the two local games (Step 1, three days at most).
5. Label the benchmark stills (1.7 hours) and the first 50 cap frames (40 minutes), then train the cap detector and settle team and goalie (Steps 3 and 4).
6. Label the benchmark clips (2 hours), then choose the tracker and the ball method (Steps 5 and 6).
7. Start the event model as soon as item 1 shows tags with video times (Step 7). It does not wait for items 4 to 6.
8. Run the acceptance check on two more unseen games before processing the archive (Step 8).

**What Step 0 tells you, and what to do with it.**

| Finding | What it changes |
|---|---|
| Tags carry video seconds | Step 7 starts now. Tags also locate benchmark clips and shot windows. |
| Tags carry period and game clock only | Step 7 still runs, after the alignment described in section 10 |
| No usable tag times | Step 7 uses cues and track evidence only, checked against the official score and your own review marks |
| Training images come from one or two games or pools | Expect the "frames from each new pool" rung in Step 3, and pick test games from other pools |
| A local game is in the training set | Nothing. Both local games are already development-only. |
| The other laptop has CVAT boxes or tracks | Spot-check 20 items. Use them for training or development, never as test data. |
| The other laptop has tracking, homography, or event code | Run it on a benchmark clip and keep it only if it matches or beats the current variant |
| The hub disk is still full | Work on this Mac and on Slurm scratch |

At each later step the decision tables say what to test, the number that keeps a method, and what to try next if it fails. If a gate fails and its ladder is exhausted, that step is the bottleneck. Spend labeling there before moving on.

## 1. Evidence ledger

All numbers on the 12 pilot stills use 91 provisional cap points that an assistant placed, not you. The stills are six pairs of frames about 20 seconds apart, so they cover six scenes. Treat them as development signals.

### Verified

| Topic | Result | Source |
|---|---|---|
| Checkpoint | RF-DETR Base (DINOv2 windowed small), 560×560 input, 25 epochs, batch 4, lr 1e-4. Classes `ball`, `player`, `referee` (IDs 1 to 3). No goalie, team, or cap class. Trained from `test.v1-test.coco` on the hub. | Checkpoint metadata, read today |
| Input scaling | Each 1920×1080 frame is squashed to 560×560, so widths shrink to 0.29× and heights to 0.52×. A 15 to 20 px daytime ball arrives at about 5×9 px. | rfdetr 1.5.2 source |
| Player coverage | At score ≥ 0.25, 88 of 91 cap points fall inside a player box (evening 59/60, daytime 29/31), with 149 boxes. | `batch/rfdetr-local/scores.json`, reproduced today |
| Score threshold | 0.35 keeps 88/91 with 134 boxes. 0.50 drops to 86/91 with 117. | Today |
| Inference settings, same weights | Two square 1080 px tiles: 90/91, 181 boxes, twice the time. Full frame at 784 px: 88/91. At 896 px: 89/91. The pilot's four field tiles: 88/91. None is a clear gain without retraining. | Today, `scores.json` |
| Field polygon | The 12 draft polygons remove only 7 to 11 of 149 RF-DETR player boxes and no covered cap. They are loose and foreground spectators sit inside them. With the pose model the same polygons cut extras from 67 to 8. | Today, `batch/geometry.json` |
| Cap position | Top-center-of-box proxy: 48/91 within 14 px (at 960×540). Pose model on each player crop: 56/91. Pose, else proxy: 67/91 (evening 49/60, daytime 18/31). | `scores.json`, today |
| Ball | In the 4 stills with a labeled ball (all evening), the top-scoring ball candidate is the ball (scores 0.53 to 0.78). The strongest distractor scores 0.65, once. No daytime still has a ball candidate above 0.36 under any setting, and those stills were never checked for a visible ball. | Today |
| Ball baselines | Yellow threshold: 4/4 with 20 extras, and it fails in daylight. COCO sports-ball: 2/4. | `followup/metrics.json`, `REPORT.md` |
| Generic detectors | Pose: 12/91 full frame, 43/91 on field tiles. Swimmer-head model: 4/91. | `batch/geometry.json`, `batch/swimmer.json` |
| Speed | 74 ms per frame on CPU, 45 to 53 ms on the Apple GPU with identical outputs. Trackers cost 0.3 to 3.3 ms per frame. A 72-minute game at 10 fps is about 40 minutes of detection on this Mac. | Today |
| Environment | Pilot venv: Python 3.12, rfdetr 1.5.2, supervision 0.31.7, ultralytics 8.4.214, torch 2.15.0, PyAV 19. `sv.ByteTrack` no longer exists in supervision. Ultralytics ships six trackers that run on external boxes, and its `track_buffer` counts frames, not seconds. This Mac has no ffmpeg and no Docker, and 747 GB free. | Today |
| Footage here | Evening: 1080p, 100 fps, 22:04, PCM audio. Daytime (Hudl export): 1080p, 30 fps, 72:14, AAC audio, about 5 Mb/s. The camera pans and zooms with play in both. | PyAV probe today |
| Repo | Frame extractor, trainer, Slurm launcher, README, `.gitignore`. The trainer calls `model.export("runs/rfdetr_waterpolo")`, but checkpoints are written to `output/`, which is why the launcher's copy step warns. | `gh` read today, rfdetr 1.5.2 source |
| Your review | At evening 4:52 two visible players are missing from the reference and the person nearest goal is the goalie. Evening crops let in deck people and officials. | Guide |

Tracker proxies, run today on cached RF-DETR player boxes (about 11 per frame, 15 seconds, 10 fps, 3-second buffer unless noted, default thresholds). Each cell is unique IDs / median track length in seconds.

| Tracker | Evening 12:22 | Daytime 59:05 |
|---|---|---|
| ByteTrack, 0.7 s buffer (pilot setting) | 52 / 1.8 | 41 / 3.2 |
| ByteTrack | 48 / 1.9 | 41 / 3.2 |
| ByteTrack at 5 fps | 49 / 2.0 | 39 / 3.4 |
| BoT-SORT with camera-motion compensation | 42 / 2.5 | 27 / 4.1 |
| OC-SORT | 42 / 2.5 | 33 / 4.0 |
| FastTracker | 43 / 2.3 | 40 / 3.4 |
| TrackTrack with camera-motion compensation | 27 / 4.3 | 21 / 6.2 |

Fewer IDs is not proof of better identities, because a tracker can merge two players into one ID. TrackTrack's default score thresholds are tuned to YOLO, and it emitted about 15% fewer boxes on the evening clip.

### Not verified

| Unknown | Why it matters | Resolved in |
|---|---|---|
| Training set size, splits, source games, and validation scores (the guide reports 210/20/10 images) | Whether the two local games count as unseen, and how varied the training venues are | Step 0b |
| Every accuracy number above | The reference points are provisional, skip swimmers whose caps are under water, and include no daytime ball | Step 2 |
| Precision for players and ball | 149 boxes against 91 labeled caps mixes real swimmers with spectators | Step 2 |
| Whether the daytime ball is detectable at all | The Hudl export is the common source type in your archive | Step 2, then Step 6 |
| Identity accuracy of any tracker | No verified identities exist | Steps 2 and 5 |
| Camera-motion estimation on pans | The pilot only tested optical flow across a 2-second gap, which failed on the pan. Frame-to-frame motion is untested. | Step 1 |
| Hudl export fields, time base, tag vocabulary, completeness, timing offset | Decides whether tags can train and score the event stage | Step 0c |
| Contents of the other laptop | Possible free labels and code | Step 0a |
| Cluster environment (V100 nodes behind Slurm; the hub disk was about 100% full) | Training and batch inference | Step 0b |
| Cut detection | The pilot heuristic never fired and was never tested on a real cut | Step 1 |
| Every event method | Nothing has been run | Step 7 |

### Where the sources differ

| Topic | Handoff, repo, or guide | Newer evidence used here |
|---|---|---|
| Training outcome | Handoff: no successful run, classes unknown | A working checkpoint exists with ball, player, referee |
| GPU | Trainer comment: V100 16 GB | The hub GPU is a P100 that needed PyTorch 2.5.1/cu118. V100s are behind Slurm. |
| Trainer output | `runs/rfdetr_waterpolo` | Checkpoints land in `output/` |
| Field gate | Guide: start with a padded polygon | Draft polygons barely filter RF-DETR output |
| Holdout | Guide: one 20 to 30 second clip from an unseen match | Too narrow to claim unseen-game accuracy. Use eight games. |
| Tracking | Pilot report: association improves once detections are reliable | Fragmentation persists with RF-DETR detections |
| Events | Earlier pilot plan: rules first, a classifier later | Tags first, rules as supporting cues |
| Model class | `RFDETRBase` | Deprecated since rfdetr 1.7 and removed in 2.0. Its checkpoint does not load into the newer sizes. |

## 2. Order of operations

| Step | What happens | Your labeling | Exit gate |
|---|---|---|---|
| 0 | Recover the other laptop's work, inventory the hub, test one Hudl export, download and freeze eight test games, set up the project | None | You know which games the detector has seen and whether tags carry video times. The test games are fixed before any model sees them. |
| 1 | Thinnest end-to-end run ("v0") on the two local games, three days at most | None | One game runs start to finish, the review page seeks correctly, and minutes per video-minute and a failure log are recorded |
| 2 | Benchmark across eight unseen games, in two sittings | About 1.7 hours of stills, then about 2 hours of clips | Exports parse and the scorer prints baseline numbers for v0 |
| 3 | Add cap and goal labels to training frames and retrain. Settle player detection and the field gate. | About 40 minutes per 50 frames, continued only while it pays | Cap and player gates |
| 4 | Team and goalie role from caps | None | Team and goalie gates |
| 5 | Choose the tracker | None | Identity gates |
| 6 | Choose the ball method | None unless a gate fails | Ball gates |
| 7 | Event candidates and timeline. Starts as soon as Step 0c passes and runs alongside Steps 2 to 6. | None | Event gates on held-out tagged games |
| 8 | Full-game acceptance, then the archive | About 10 minutes of spot checks per acceptance game | All gates on two more unseen full games |

Rules that apply throughout:

- Every numeric gate is a proposed starting value.
- The two local games are development data. No method is kept or dropped on v0 output or on the 12 stills alone.
- Test games are scored only at gates and never used for tuning or training.
- With roughly 500 labeled caps the benchmark resolves differences of about 5 points, not 1 or 2. Inside that margin, keep the simpler and faster variant, as the guide's tie-breaker says.

## 3. Step 0: recover, inventory, export, freeze the test games

**0a. Other laptop (read-only, one-hour limit).** Run section 13a. Bring back the uncommitted and untracked Court Vision files, CVAT task backups and dataset exports, and any model outputs. Reuse labels after a 20-item spot check against the definitions in section 12. Reuse code only if it runs on a benchmark clip and matches or beats the current variant on the Step 2 scorer. Archive the rest.

**0b. Hub.** Run section 13b. It prints image counts per class, per image size, and per source video, the saved validation results, the four scripts that are not in the public repo, the rfdetr and torch versions, and free disk. Mark each game as seen or unseen. If train and validation share a game, re-split by match before any retraining.

**0c. Hudl export test.**
- Pick one tagged game whose video you can also download. Export its tag data from the video's menu in the library. If the tags came from Hudl Assist or a Sportscode timeline, ask for the XML as well. Hudl's manual says Sportscode XML stores each instance's `<start>` and `<end>` in seconds.
- The export is usable if every tagged event has a type that separates shots from goals and a time relative to the video (seconds, or clip start and end). If times are period and game clock instead, they need alignment and are a weaker second case.
- Audit 20 tags (10 shots, 10 goals, about 15 minutes). Record tag time minus true time and what the tag marks (release, ball in goal, whistle, or a clip window). Subtract the median offset from all tags. The spread sets the scoring tolerance, and this plan assumes ±3 seconds.
- Check completeness on the same game: count shots in one period by eye and compare with the tags. Missing tags make good candidates look false.
- If there are no usable times, ask Hudl support for an XML or timeline export and use the fallbacks in Step 7.

**0d. Freeze the test games.** Choose eight games by date, opponent, and pool, never by looking at model output. Requirements: none in the training set, at least four pools, at least one pool absent from training, and a mix that matches the archive (mostly Hudl exports, plus one night game and one low-angle game if you have them). Keep all angles of a match on the same side of the split. Download them and write `games.csv` (match, pool, date, source type, fps, split).

**0e. Project setup on this Mac.** Run section 13c. It clones the repo, pins the pilot's exact package versions, and copies the pilot helpers into the project so nothing depends on the `.codex` folder.

## 4. Step 1: v0 end to end, no labels

The point of v0 is the scaffolding: caches every later comparison reuses, a scorer, a review page, and a log of real failures. It also gives you something to click on within days.

| New file | What it does | Reuses |
|---|---|---|
| `cv/detect.py` | Decodes at 10 fps, runs the checkpoint on the full frame, and stores every box with score ≥ 0.05 together with source timestamp, video hash, checkpoint hash, input size, and transform. Later steps filter the cache and never re-detect. | Model call in `batch/rfdetr-local/run.py`, decode schedule in `pilot.py` |
| `cv/motion.py` | Frame-to-frame camera motion from sparse optical flow, plus cut detection. Cached per frame. | Flow code in `batch/geometry.py` |
| `cv/track.py` | Runs an Ultralytics tracker on cached boxes. Default: BoT-SORT with camera compensation and a 3-second buffer (`track_buffer = fps × seconds`). Resets at cuts. | `Boxes` wrapper and `track()` in `pilot.py` |
| `cv/ball.py` | One best path through the per-frame ball candidates over the whole clip, with a hidden state and a speed limit after camera compensation. Isolated fast detections are kept as possible flight. Candidates that stay still for more than 5 seconds are dropped as fixtures. States: `observed`, `interpolated`, `hidden`, `offscreen`, `uncertain`. | New |
| `cv/audio.py` | Whistle times from 3.5 to 4.5 kHz band energy | PyAV, already installed |
| `cv/events.py` | Placeholder candidates: whistle followed by a stoppage, plus ball-speed spikes | New |
| `cv/review.py` | Static page with the video and a candidate list. Click or J/K seeks to 3 seconds before. Keys 1/2/3 mark shot, goal, or reject. A button exports your decisions as CSV. | Report pattern in `make_report.py` |
| `tools/make_frames.py` | Writes benchmark frames as JPEGs named by game and source timestamp, with a manifest | Frame reads in `batch/stage.py` |
| `cv/score.py` | Reads the CVAT XML exports and the pipeline outputs and prints every metric named in Steps 3 to 7, pooled and per game | One-to-one matching in `run.py` |
| `tools/tags.py` | Turns the Hudl export into `tags.csv` (`game, seconds, type, team`) with the audited offset applied | New |
| `tools/prelabel.py` | Writes model output as CVAT XML pre-labels, for training frames only | New |

In v0 the player gate is the pilot's vertical band and team and role are blank. The evening file has PCM audio, which browsers may not play. `brew install ffmpeg` and a 720p H.264/AAC proxy fix that for review.

v0 also logs measures that need no labels. They rank variants early but each has a blind spot, so none of them decides anything.

| Measure | Blind spot |
|---|---|
| Boxes that blink off for one or two frames; more than 14 in-water boxes | Consistent misses and stable false boxes |
| Track births and deaths away from the frame edge; team color flipping inside one track | Swaps between teammates |
| Ball coverage during live play; two strong candidates far apart | A stable distractor passes |
| Predicted goals per period against the official score | Errors that cancel out |

## 5. Step 2: benchmark across eight unseen games

This is the only labeling required up front. The exact schema is in section 12A. Cap centers are labeled as single points, because one click is fast, does not depend on any detector's box shape, and serves detection, cap position, team, role, and identity at once.

**Sitting 1, stills (about 1.7 hours). Unblocks Steps 3, 4, and the still-frame part of 6.**
- 48 test stills, six per game at fixed fractions of the video. Each comes with two later alternates (5 and 10 seconds on). Label the first that shows live play. The frames half a second before and after are included for context only.
- 160 ball-only stills, 20 per game at random times: click the ball or tag it not visible.
- The 12 pilot stills: correct the 91 points, add the two missing players at 4:52, and add team and goalie.

**Sitting 2, clips (about 2 hours). Unblocks Steps 5 and 6.**
- Eight 15-second clips, six from test games and two from the local games (evening from 12:22, daytime from 59:05), as frames at 5 per second. Choose them yourself by watching, not from model output: two pans, two crowded set positions, a counterattack, a player leaving and re-entering the frame. Hudl tags help find them.
- Twelve 1-second release windows at 25 to 30 fps around tagged shots, eight from test games.

**Keeping the labels independent of the models.**
- Label with no model overlays or pre-labels.
- After the first models have run, do one pass over boxes that no label explains, add any players you missed, and save that as benchmark version 2.
- Re-label five stills blind a day later to see how well you agree with yourself.

## 6. Step 3: detector upgrade, player detection, field gate

**Already decided by evidence: train a cap detector.** The proxy and the pose model both miss too many caps (48, 56, and 67 of 91). The checkpoint stays as it is for v0.

- **Labels.** Add `cap` and `goal` boxes to existing training images, which already carry ball, player, and referee boxes (section 12B). First batch is 50 images spread across source videos. If the hub set proves unusable, label the 50 staged 1080p pilot frames in `batch/drafts/` for all classes, starting from the checkpoint's boxes.
- **Training.** On the cluster with rfdetr 1.5.2, start from the existing checkpoint with `pretrain_weights` (section 13d). The first 50 images train a separate cap-and-goal detector. The existing checkpoint stays untouched and still supplies players, ball, and referees, because a combined model trained on only 50 images would lose what the other 190 teach. Until the models are combined, detection takes two passes per frame.
- **Growing the set.** The cap model pre-labels the remaining training images for you to correct. Once every training image carries caps, train one combined model and keep it only if it matches the two-model setup on every gate.

| Decision | Cheapest test | Keep if (test stills) | Fallback ladder |
|---|---|---|---|
| Cap detection | Train the cap-and-goal detector two ways from the same labels: full frame, and two square 1080 px tiles (twice the cost, no sideways squash). Score both. | Recall on visible caps ≥ 0.90 pooled and ≥ 0.80 on the worst game, with ≤ 0.5 false caps per frame in the water | 1. Add 50 images at a time until an increment adds less than 2 points. 2. A small cap locator run on native-resolution player crops. 3. The RF-DETR keypoint preview with one cap point per player box. 4. Pretrain on the Berkeley head boxes with the other classes filled in by the current model. |
| Player detection | Score the existing checkpoint on the test stills. No extra labels. | Recall ≥ 0.95 of in-water players pooled, ≥ 0.90 on the worst game and for goalies, with ≤ 1 false player per frame in the water | 1. Threshold 0.35, which was free on the stills. 2. Correct 30 to 50 hard frames from v0's failure log (track gaps, low scores, deck boxes). 3. 25 frames from each new pool. 4. A/B `RFDETRMedium` from COCO weights on the same labels. That migration is due before rfdetr 2.0 anyway. |
| Field gate needed at all | Count false players and caps per frame with no gate | Skip the gate if the limits above are met without it | Go to the next row |
| Field gate method | No polygons. Score each box by the share of water-colored pixels in a ring around and below it, using a per-frame water color model. | Keeps ≥ 99.5% of labeled players and removes ≥ 80% of boxes that are not players | 1. Drop tracks that stay still for 30 seconds after camera compensation. 2. Retrain with spectators and bench players left unlabeled on low-angle frames. 3. A hand polygon per continuous shot, propagated frame to frame. 4. A landmark model only if zones or pool coordinates are needed later. |

The gate applies to players and caps only, never to the ball or referees. In training, deck people, bench players, and spectators get no box, so the detector learns them as background.

## 7. Step 4: team and goalie role

No new labels. Team and role belong to a track and are voted over its cap crops.

| Decision | Cheapest test | Keep if (test stills and clips) | Fallback ladder |
|---|---|---|---|
| Do cap colors separate | On the corrected pilot stills, sample color at the labeled cap centers. If dark, white, and red do not separate there, no locator will fix it. | Three separable groups in both local games | Go straight to the classifier in the next row's ladder |
| Team | Per-game clustering of each track's median cap color, with red set aside for goalies. Never fixed color classes, because dark caps differ by team. | Accuracy ≥ 0.95 on track-time that gets a label, with ≤ 15% left unknown | 1. SigLIP embeddings with k-means, as in Roboflow's `TeamClassifier`. 2. A small supervised cap-color classifier trained from the `color` attribute in the training labels. |
| Goalie | Red group, corroborated by time spent near a `goal` box | Precision ≥ 0.95 and recall ≥ 0.90 of goalie track-time | Same ladder as team |

Leave team or role unknown when the vote margin is under 2 to 1 or there are fewer than three clear crops. That is the rule already in `pilot.py`. NCAA rules help: visiting field players wear solid white and the home team dark, and goalkeeper caps are red with team-colored panels. A team may play without a goalkeeper, so never force one goalie per side.

## 8. Step 5: tracker

**What reliable means here.** Within one continuous camera view, a track ID never moves to a different person. Breaking into a new ID is acceptable. A player who leaves and re-enters the frame gets a new ID unless team, position, and timing make the link safe within 3 seconds. Season-long identity by cap number is out of scope, so the unit is a track with a team and a role.

- **Test.** Every variant runs on cached detections and cached camera motion in seconds, with no GPU. Vary the tracker (ByteTrack, BoT-SORT, TrackTrack, OC-SORT, FastTracker), the rate (10 fps and 25 or 30 fps), the buffer (0.7, 2, and 4 seconds, converted to frames), and the tracked object (player boxes or cap boxes). Set score thresholds from RF-DETR's own score distribution, not the YOLO defaults. Rank by the label-free measures, then confirm the top three on the clips.
- **Metrics.** Identity swaps per video-minute, purity (the share of a player's visible time spent under their main ID), and fragments per 10 seconds of visible time. Matching uses the labeled cap point inside the predicted box.
- **Keep if (test clips).** At most 2 swaps per video-minute, purity ≥ 0.85, at most 1 fragment per 10 seconds, and no goalie gap over 2 seconds while visible. Stay at 10 fps unless 25 or 30 fps cuts swaps by more than a fifth.
- **Fallback ladder.** 1. Forbid matches across teams, since the nearest neighbor is usually an opponent. 2. Associate on cap boxes. 3. Stitch fragments offline by team, time, and motion across gaps up to 3 seconds. 4. BoT-SORT with appearance features or McByte from Roboflow's `trackers` package. 5. SAM-based re-identification only on flagged crossings.
- **Extra development set with no labeling.** The Berkeley game has head boxes with track IDs. Use it if its terms (non-commercial research and education) fit your use.

## 9. Step 6: ball

| Decision | Cheapest test | Keep if (test games) | Fallback ladder |
|---|---|---|---|
| Finding the ball in a frame | Top-scoring candidate against the 160 ball stills, per game | Recall on visible balls ≥ 0.85 pooled and ≥ 0.70 on the worst game | 1. The two-square-tile model from Step 3. 2. Native-resolution crops around the ball carrier, the predicted path, and the goal mouth. |
| Following it over time | The v0 path method against the clip ball tracks | Locked onto something that is not the ball in ≤ 10% of frames | Tune the speed limit and the still-object rule on the local clips |
| Sampling rate for shots | 10 against 25 or 30 fps on the twelve release windows, from the same cache | Release located within ±0.2 seconds for ≥ 80% of shots, and at least one in-flight or arrival detection for ≥ 60% | Run 25 to 30 fps only in windows opened by ball speed, goal proximity, or an event candidate |
| Ball-specific model | Only if the rows above fail after tuning | Not applicable | 1. Correct 400 to 600 ball pre-labels drawn from tagged shot windows (about 1.5 hours) and retrain on tiles. 2. A multi-frame heatmap tracker (WASB, MIT license) fine-tuned on those sequences. No published result covers held balls, so treat it as an experiment. |

A shot travels 15 to 25 m/s, which is 1.5 to 2.5 m per frame at 10 fps. Flight needs 25 to 30 fps even if possession does not.

## 10. Step 7: candidate shots and goals

**Three sources, combined.**
- **A model trained on your tags (main recall source).** Start with frozen per-frame features (DINOv2 or SigLIP at 4 fps) and a small temporal head, with labels shifted by the audited offset and widened to ±3 seconds. It trains in minutes once features are cached. At this input size the model cannot see the ball and will lean on body motion and camera behavior, so hold out by pool.
- **Whistle and stoppage cues.** Ordinary fouls make whistles far more frequent than goals, so a whistle is a feature, not a candidate. The goal signature is whistle, stoppage, then a restart from halfway. A small classifier fit on tagged goals against other whistles uses no new labels.
- **Track evidence.** Ball release speed toward a `goal` box and the ball entering or vanishing at the goal. These rank candidates and appear as evidence on the review page.

Candidates within 3 seconds merge. A small model fit on tagged games ranks them.

| Decision | Cheapest test | Keep if (held-out tagged games) | Fallback ladder |
|---|---|---|---|
| Is the tag-trained model good enough | Train on 20 to 30 games, hold out about 15% by pool, score cues, model, and their union on the same games | Goal recall ≥ 0.95 and shot recall ≥ 0.85 within ±3 seconds, with at most 2 candidates per true event | 1. The union of all three sources. 2. End-to-end E2E-Spot (BSD-3 license) at about 398×224 and 8 fps. Its authors trained on one 24 GB GPU in hours. A 16 GB card with smaller batches is untested. 3. A larger input. 4. More games. |
| Do the cues earn their place | Add them to the union and rescore | They recover events the model missed and stay inside the 2-per-event limit | Keep them only as evidence on the review page |
| Review cost | Time yourself on one game | 15 minutes or less per game | Raise the score floor, or review goals first |

Every decision you make on the review page is saved as a label, so reviewing games also grows the training set.

**If the Hudl export has no video times.**
- Times as period and game clock: align tagged goals to detected restarts, in order, anchored on each period's opening sprint.
- No times at all: compare predicted goals per period with the official score. Tag shots yourself with hot keys on the review page for four games (about 20 minutes each) to measure shot recall. Run cues and track evidence only, and build training labels from your review decisions.

## 11. Step 8: acceptance and rollout

- Run the whole pipeline on two more unseen tagged games. Event gates must hold. Then open ten random 10-second windows per game with overlays on and mark each pass or fail for boxes, team, goalie, and ball. A clip benchmark can hide errors that only show over a full match.
- Record model and tracker minutes per video-minute and your review minutes.
- Batch the archive on the cluster one game at a time. Stage video and caches on scratch, because the hub disk is full. Key every output by video hash and checkpoint hash.
- Lock each newly reviewed game as test data before any of its labels are used for training.

## 12. Annotation spec

Self-hosted CVAT. It already ran on the other laptop at `localhost:8080`. Round-trip one task (create, label two frames, export, parse) before bulk work.

Definitions: a player is any athlete in the water, including goalies. Bench players, coaches, officials, and spectators are never labeled as players or caps.

**A. Benchmark. Required once, about 3.7 hours in two sittings.**

| Label | Shape | Used in | Rule |
|---|---|---|---|
| `cap_dark`, `cap_white`, `cap_goalie`, `cap_unknown` | One point | Stills and clips | Click the cap center of every in-water player whose cap is visible. Pick the label with its number key. In clips this is a track: keyframe at least once a second, mark `occluded` while the head is under water or hidden, mark `outside` when the player is off frame. |
| `swimmer_nocap` | One point | Stills | A player who is visibly present but whose cap cannot be seen. Click the estimated head position. |
| `ball` | One point, attribute `state` (free, held) | Stills, clips, release windows | Click the ball center if any part is visible. In clips it is a track, `outside` when not visible. |
| `no_ball_visible`, `skip` | Image tag | Stills | No ball in view. Dead time or an alternate you did not use. |
| `goal` | Rectangle | Stills | Around the posts and crossbar when a goal is in view |
| `water` | Polygon | Stills | Along the pool edges and ropes that bound the field of play in view |
| `release` | Image tag, attribute `type` (shot, pass) | Release windows | On the frame the ball leaves the hand |

| Item | Count | Frames | Estimate |
|---|---|---|---|
| Test stills, fully labeled | 48 | Named by game and timestamp, with context frames | 70 min |
| Ball-only stills | 160 | Same | 16 min |
| Pilot stills, corrected | 12 | Existing | 15 min |
| Clips | 8 × 15 s | 5 per second | 95 min |
| Release windows | 12 × 1 s | 25 to 30 per second | 25 min |

Export stills as "CVAT for images 1.1" and clips as "CVAT for video 1.1". Both are XML and keep track IDs, the keyframe, outside, and occluded flags, and attributes.

**B. Cap and goal training labels. Triggered now by the cap-position result.**

| Item | Detail |
|---|---|
| Images | Existing training images from training games only, 50 at a time |
| `cap` | Tight rectangle around the visible cap, not the face. Attribute `color` (white, dark, goalie). In-water players only. |
| `goal` | Rectangle around the posts and crossbar |
| Pre-labels | Pose-on-crop cap guesses imported as pre-annotations. About two thirds are right, so you correct rather than draw. Never used on benchmark frames. |
| Export | COCO 1.0, arranged as `train/ valid/ test/`, each with `_annotations.coco.json`, split by match. Cap and goal boxes are merged with the existing ball, player, referee boxes once every image carries caps. |
| Estimate | About 40 minutes per 50 images |

**C. Conditional labels. Only if a gate fails.**

| Trigger | What to label | Format |
|---|---|---|
| Player gate fails | Correct checkpoint boxes on 30 to 50 hard frames, at most 25 per game | COCO 1.0 |
| Ball gate fails after tuning | Correct ball boxes on 400 to 600 frames from tagged shot windows | COCO 1.0 |
| Hudl export has no times | Hot-key shot and goal marks on the review page for four games | Review CSV |

**D. Never labeled by hand.** Shot and goal events when Hudl tags carry times, team models (clustered per game), and identities beyond the benchmark clips.

## 13. Commands

Shared variables for this Mac:

```sh
PILOT="/path/to/water-polo-pilot"
CKPT="/path/to/checkpoint_best_total.pth"
export CKPT
PY="$PILOT/.cv-venv/bin/python"
```

**Reproduce the 12-still baseline (runs now).** Expect 88/91 covered, 48 head-proxy matches, 149 player boxes, and 4/4 balls for the full-frame row.

```sh
trial="$(mktemp -d)" && cp -R "$PILOT/batch/rfdetr-local" "$trial/"
"$PY" "$trial/rfdetr-local/run.py" "$CKPT"
"$PY" "$trial/rfdetr-local/make_report.py"
open "$trial/rfdetr-local/index.html"
```

**Read the checkpoint's classes and settings (runs now).**

```sh
"$PY" -I - <<'EOF'
import argparse, os, torch
torch.serialization.add_safe_globals([argparse.Namespace])
ck = torch.load(os.environ["CKPT"], map_location="cpu", weights_only=True)
a = vars(ck["args"])
print({k: a[k] for k in ("class_names", "num_classes", "resolution", "encoder", "epochs", "batch_size", "lr", "grad_accum_steps", "dataset_dir")})
EOF
```

**13a. Other laptop (read-only).**

```sh
mdfind -name Court-Vision; mdfind -name cvat | grep -v site-packages | head
cd /path/to/Court-Vision
git remote -v && git status --short && git stash list && git log --oneline -15
git diff --stat 03ec4f1737e17a51e965c089d9fa4f6af767e91f
docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -i cvat; docker volume ls | grep -i cvat
```

If CVAT starts (`docker compose up -d` in its folder, then `localhost:8080`), back up every task. `cvat-cli` needs Python 3.11 or newer.

```sh
pip install cvat-cli
cvat-cli --server-host http://localhost:8080 --auth <user> task ls
cvat-cli --server-host http://localhost:8080 --auth <user> task backup <id> task_<id>.zip
cvat-cli --server-host http://localhost:8080 --auth <user> task export-dataset --format "COCO 1.0" <id> task_<id>_coco.zip
```

If CVAT will not start but its containers still exist, save the volumes. Restoring needs the same CVAT version.

```sh
mkdir backup
docker run --rm --name temp_backup --volumes-from cvat_db -v $(pwd)/backup:/backup ubuntu tar -czvf /backup/cvat_db.tar.gz /var/lib/postgresql/data
docker run --rm --name temp_backup --volumes-from cvat_server -v $(pwd)/backup:/backup ubuntu tar -czvf /backup/cvat_data.tar.gz /home/django/data
```

**13b. Hub inventory (JupyterHub terminal).** The paths come from the checkpoint and an earlier session's file listing. I have not re-checked them.

```sh
cd ~/Court-Vision && ls -la && (git status --short; git log --oneline -5) 2>/dev/null
du -sh test.v1-test.coco frames videos training output 2>/dev/null; df -h ~ | tail -1
cat output/results.json; tail -n 3 output/log.txt
python - <<'EOF'
import json, collections, pathlib, re
root = pathlib.Path("test.v1-test.coco")
for split in ("train", "valid", "test"):
    d = json.load(open(root / split / "_annotations.coco.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    per_class = collections.Counter(cats[a["category_id"]] for a in d["annotations"])
    stems = collections.Counter(re.sub(r"(_\d+)?_(jpg|png)\.rf\..*$", "", i["file_name"]) for i in d["images"])
    sizes = collections.Counter((i["width"], i["height"]) for i in d["images"])
    print(split, len(d["images"]), "images", dict(per_class), dict(sizes))
    print("   source videos:", stems.most_common(15))
EOF
head -40 extract_diverse_frames_gpu.py process_candidates.py process_frame_archives.py process_ucsb.py
pip show rfdetr torch | grep -E "^(Name|Version)"; nvidia-smi --query-gpu=name,memory.total --format=csv
```

**13c. Project setup on this Mac.**

```sh
git clone https://github.com/tonnyryam/coach-kal.git
cd coach-kal
git switch -c pipeline-v0
uv pip freeze --python "$PY" > requirements.lock
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.lock cvat-cli
mkdir -p cv tools pilot_ref
cp "$PILOT"/pilot.py "$PILOT"/batch/geometry.py "$PILOT"/batch/stage.py pilot_ref/
cp "$PILOT"/batch/rfdetr-local/{run.py,make_report.py,labels.json,polygons.json} pilot_ref/
```

**13d. Training on the cluster.** Save this as `train_rfdetr_v2.py` next to the existing trainer. It writes checkpoints where you tell it and drops the ONNX export, which needs extras that are not installed.

```python
import os
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=os.environ.get("INIT", "output/checkpoint_best_total.pth"))
model.train(
    dataset_dir=os.environ["DATASET_DIR"],
    output_dir=os.environ.get("RUN_DIR", "runs/exp"),
    epochs=int(os.environ.get("EPOCHS", "60")),
    batch_size=4, grad_accum_steps=4, lr=1e-4, early_stopping=True,
)
```

```sh
# JupyterHub P100, inside tmux. The env that worked used PyTorch 2.5.1/cu118.
pip install "rfdetr==1.5.2"
DATASET_DIR=/path/on/scratch/dataset_caps_b1 RUN_DIR=runs/coach-kal-caps-b1 python train_rfdetr_v2.py
```

For Slurm, reuse `train_rfdetr.sbatch` with three edits: set `DATA_SRC` to the new dataset, run `train_rfdetr_v2.py`, and set `OUT_SRC` to the run directory. Smoke-test with `EPOCHS=1` first.

**13e. Sampling frames from new pools (repo script, cluster env).** Pass `--target_h 1080` to keep native pixels. The script writes no timestamps yet. Add a manifest (match, angle, source time, frame index, path) before using it for new data.

```sh
python extract_diverse_frames.py --input_dir <videos> --output_dir <frames> --frames 200 --sample_fps 3 --target_h 1080
```

**13f. CVAT on this Mac, if you prefer it to the other laptop.** Docker Desktop or OrbStack is needed first. I have not tested CVAT on Apple silicon.

```sh
git clone https://github.com/cvat-ai/cvat && cd cvat && docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

**13g. Benchmark tasks.** Save the labels below as `labels.json`. Check option names with `cvat-cli task create --help`. If the CLI rejects the file, paste it into the Raw tab of the label editor. Confirm the export format names in CVAT's export dialog.

```json
[
  {"name": "cap_dark", "type": "points", "attributes": []},
  {"name": "cap_white", "type": "points", "attributes": []},
  {"name": "cap_goalie", "type": "points", "attributes": []},
  {"name": "cap_unknown", "type": "points", "attributes": []},
  {"name": "swimmer_nocap", "type": "points", "attributes": []},
  {"name": "ball", "type": "points", "attributes": [
    {"name": "state", "input_type": "select", "mutable": true, "values": ["free", "held"]}]},
  {"name": "goal", "type": "rectangle", "attributes": []},
  {"name": "water", "type": "polygon", "attributes": []},
  {"name": "no_ball_visible", "type": "tag", "attributes": []},
  {"name": "skip", "type": "tag", "attributes": []},
  {"name": "release", "type": "tag", "attributes": [
    {"name": "type", "input_type": "select", "mutable": false, "values": ["shot", "pass"]}]}
]
```

```sh
cvat-cli --server-host http://localhost:8080 --auth <user> task create "coach-kal-bench_stills_v1" --labels labels.json local bench/stills/*.jpg
cvat-cli --server-host http://localhost:8080 --auth <user> task create "coach-kal-bench_clip_<game>" --labels labels.json local bench/clips/<game>/*.jpg
cvat-cli --server-host http://localhost:8080 --auth <user> task export-dataset --format "CVAT for images 1.1" <id> bench_stills_v1.zip
cvat-cli --server-host http://localhost:8080 --auth <user> task export-dataset --format "CVAT for video 1.1" <id> bench_clip_<game>.zip
# training frames only: load pre-labels
cvat-cli --server-host http://localhost:8080 --auth <user> task import-dataset --format "CVAT 1.1" <id> prelabels.xml
```

## 14. Existing work to reuse

| Asset | Where | Use |
|---|---|---|
| Checkpoint | `$CKPT` and `output/` on the hub | Baseline detector and starting weights |
| Training set and logs | `~/Court-Vision/test.v1-test.coco`, `output/log.txt`, `output/results.json` on the hub | Ball, player, referee boxes. Cap and goal boxes are added to these images. |
| Model call, tiling, matching score | `batch/rfdetr-local/run.py` | `cv/detect.py` and the scorer |
| Tracker config, decode schedule, cap-color vote, overlays | `pilot.py` | `cv/track.py`, team voting, review overlays |
| Polygon test and optical flow | `batch/geometry.py` | `cv/motion.py`, field gate fallback |
| HTML report pattern | `batch/rfdetr-local/make_report.py` | `cv/review.py` |
| 91 cap points, 4 ball points, 12 polygons | `labels.json`, `polygons.json` | Development checks once you correct them |
| 50 staged 1080p frames | `batch/drafts/` | Backup source of training frames |
| Frame extractor and trainer | Public repo | New-pool sampling and training, with the fixes above |
| Four more scripts, `frames/`, `videos/` | Hub, not in the repo | Inspect in Step 0b |
| CVAT tasks, tracking, homography, event code | Other laptop | Step 0a |
| Six trackers, sliced inference, audio decode | Installed packages | Steps 1, 5, 6, 7 |
| Hudl tags and video | Hudl library | Event training and scoring, finding benchmark clips |
| Berkeley game (head boxes by cap color, track IDs, possession) | Public, non-commercial research and education terms | Tracker development set and optional cap pretraining |

## 15. Verification

1. The baseline command reproduces the ledger numbers.
2. `cv/detect.py` on a 30-second window gives the same boxes on CPU and Apple GPU. The 12 stills already do.
3. A CVAT round trip: create a two-frame task, label it, export, and parse track IDs, flags, and attributes.
4. The scorer run on the benchmark with v0 outputs prints every metric named in Steps 3 to 6, pooled and per game.
5. `review.html` for one game opens, seeks to within a second of each candidate, and exports your decisions.
6. With tags loaded, the event scorer prints recall within ±3 seconds and candidates per true event for a held-out game.
7. Step 8 on two more unseen full games.

## 16. Risks

| Risk | Mitigation |
|---|---|
| Training images come from a few pools, so unseen pools fail | Step 0b counts images per source video. Test games span four or more pools. New training frames are capped at 25 per game. Results on never-seen pools are reported separately. |
| The local games overlap the training set | Treat both as development only until Step 0b says otherwise |
| The existing split is random, with near-duplicate frames on both sides | Re-split by match |
| Hudl compression (about 5 Mb/s) hides small caps and the ball | The test set is weighted to Hudl exports, so the gates see it. Add compression and blur augmentation at the first retrain. |
| 100 fps and 30 fps sources behave differently | Everything is sampled by time, tracker buffers are set in seconds, and results are reported by source type |
| Tag times are late, mark clips, or are incomplete | The 20-tag audit and the completeness check, before any training |
| Benchmark metrics are noisy | 5-point rule for adopting a challenger, and full-game acceptance in Step 8 |
| The hub disk is full | Use scratch on Slurm nodes and keep video on Hudl or this Mac |
| rfdetr moves on from `RFDETRBase` | Pin 1.5.2 now and run the `RFDETRMedium` A/B at the first retrain |
| Other-laptop recovery stalls | One-hour limit. No step depends on it. |

## 17. Housekeeping

- Today's tracker check made Ultralytics write a default settings file at `$PILOT/Ultralytics/Ultralytics/settings.json`. It is harmless and can be deleted.
- The checks behind the "today" rows in section 1 were one-off scripts that wrote nothing to disk. Re-create them as files under `tools/` during Step 0e so the ledger numbers stay reproducible. Each one loads the checkpoint as `run.py` does and reads `labels.json` and `polygons.json`.
- Keep this roadmap in the project repo and update the evidence ledger whenever a gate is scored, so verified results and open experiments stay separate.

## Sources

The RF-DETR, CVAT, and Berkeley pages were fetched and quoted directly today. The other links come from a broader research pass and were not each re-opened.

- RF-DETR: [migration guide](https://rfdetr.roboflow.com/latest/getting-started/migration/), [training parameters](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/), [repo](https://github.com/roboflow/rf-detr)
- Trackers: [Ultralytics track mode](https://docs.ultralytics.com/modes/track), [Roboflow trackers benchmark](https://trackers.roboflow.com/latest/evaluations/results/), [supervision changelog](https://supervision.roboflow.com/latest/changelog/), [McByte](https://arxiv.org/html/2506.01373v1), [Selective Mask Propagation](https://arxiv.org/abs/2606.13033)
- Ball: [SAHI](https://arxiv.org/abs/2202.06934), [WASB-SBDT](https://github.com/nttcom/WASB-SBDT), [Roboflow ball tracking](https://roboflow.com/blog/tracking-ball-sports-computer-vision), [Roboflow sports](https://github.com/roboflow/sports)
- Events: [E2E-Spot](https://github.com/jhong93/spot), [T-DEED](https://arxiv.org/html/2404.05392v2), [whistle detection](https://arxiv.org/pdf/1411.6496), [audio-visual goal spotting](https://arxiv.org/html/2011.04258)
- Water polo: [Felsen et al. ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Felsen_What_Will_Happen_ICCV_2017_paper.pdf), [Berkeley data and terms](https://pannaf.github.io/posts/sports-forecasting/water-polo-data), [NCAA 2026-28 rules](https://collegiatewaterpolo.org/wp-content/uploads/2026/05/202627_202728NCAAWaterPoloRulesBook.pdf), [World Aquatics 2026 rules extract](https://www.bwpl.org/uploads/2/6/5/9/26598283/water_polo_rules_extract_from_2026-02-18_world-aquatics_cr-final.pdf)
- CVAT: [installation](https://docs.cvat.ai/docs/administration/community/basics/installation/), [CLI](https://docs.cvat.ai/docs/api_sdk/cli/), [CVAT XML format](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/), [backup guide](https://docs.cvat.ai/docs/administration/community/advanced/backup_guide/), [one-point interpolation](https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/annotation-with-points/liner-interpolation-with-one-point/)
- Hudl: [Sportscode XML edit list](https://www.hudl.com/support/manuals/sportscode/code/importing-files/xml-edit-list), [export breakdown data](https://support.hudl.com/s/article/import-export-stats-hudl-classic?language=en_US&topic=Breakdown_Stats_and_Reports_Classic)
