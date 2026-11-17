"""Make the offline review checklist and contact sheet from pilot outputs."""
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np

cases = [
    ("1-1", 1, "Which caps near the left goal did tiled pose recover, and which green marks are duplicates?"),
    ("1-1", 3, "How many players are visibly in the water? Flag spectator marks separately."),
    ("1-2", 1, "Circle the visible far-side caps. Include partly submerged heads if you can see a cap."),
    ("1-2", 4, "Which cap near splash or glare is easiest to miss? Mark unclear heads as unclear."),
    ("1-3", 1, "Mark the ball if visible. Is the raised arm holding it, or has it already been released?"),
    ("1-3", 3, "Which lower-right marks are spectators? Identify the nearest two valid caps."),
    ("2-1", 1, "How many visible heads are near the goal? Does this look like live play or warm-up?"),
    ("2-1", 3, "Is the ball visible? In the ball clip, flag any green box that follows a cap."),
    ("2-2", 1, "Mark capped heads close to the surface and say whether these players are warming up or playing."),
    ("2-2", 4, "Is there a visible head along the far rope on the right, or is it too obscured to label?"),
    ("2-3", 1, "Follow two nearby caps through the clip. Do the numbered IDs stay attached to them through any crossing?"),
    ("2-3", 3, "Check the five-second ball clip. If a release is visible, give the timestamp and say pass, shot, or unclear."),
]
metrics = json.loads(Path("metrics.json").read_text())
offsets = (1,4,8,12)
tiles = []
data = []
cards = []
for i,(clip,frame,question) in enumerate(cases,1):
    info = metrics["clips"][clip]
    time = info["start"] + offsets[frame-1]
    source = info["file"]
    image = f"frame-{clip}-{frame}.jpg"
    pose = "pose-tiles" if clip in ("1-1","2-3") else "pose-full"
    tracks = f"{pose}/tracks-{clip}.mp4"
    links = [f'<a href="{tracks}">Pose tracks</a>',f'<a href="tracks-{clip}.mp4">Swimmer-head tracks</a>']
    if i in (5,8,12):
        ball = "ball-action/ball-1.mp4" if i == 5 else ("ball-2.mp4" if i == 8 else "ball-action/ball-2.mp4")
        links.append(f'<a href="{ball}">Ball comparison</a>')
    e = html.escape
    cards.append(f"""<article><h2>{i}. {e(source)} · {time//60}:{time%60:02}</h2>
<p>{e(question)}</p><a href="{image}"><img src="{image}" loading="lazy" alt="Four detector views for prompt {i}"></a>
<p>{' · '.join(links)}</p></article>""")
    data.append([i,source,time,question,image,tracks,"",""])
    im = cv2.imread(image)
    if im is None:
        raise RuntimeError(f"Missing {image}")
    small = cv2.resize(im,(640,360))
    cv2.rectangle(small,(0,0),(640,25),(36,36,36),-1)
    cv2.putText(small,f"{i:02d} | {clip} | {time//60}:{time%60:02}",(9,18),0,.57,(255,255,255),1,cv2.LINE_AA)
    tiles.append(small)
cv2.imwrite("review-contact-sheet.jpg",np.vstack([np.hstack(tiles[k:k+2]) for k in range(0,12,2)]))
with Path("review-answers.csv").open("w",newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["prompt","video","time_seconds","question","detector_comparison","pose_tracking_clip","answer_or_unclear","notes"])
    writer.writerows(data)
Path("review.html").write_text("""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coach Kal</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#152630;background:#f7fafb}
article{background:white;padding:1rem 1.25rem;margin:1.2rem 0;border:1px solid #dce6eb;border-radius:.5rem}
h1,h2{line-height:1.2}h2{font-size:1.14rem}img{display:block;width:100%;height:auto}a{color:#005677}
</style><main><h1>Coach Kal</h1>
<p>These 12 prompts focus on cap misses, distracting objects, track IDs, and ball visibility. Record your observations in <a href="review-answers.csv">the answer sheet</a>. “Unclear” is a valid answer. Video times refer to the source file.</p>
<p><a href="review-contact-sheet.jpg">Contact sheet</a> · <a href="REPORT.md">Pilot report</a></p>
"""+"\n".join(cards)+"\n</main>",encoding="utf-8")
