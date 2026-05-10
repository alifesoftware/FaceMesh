# Face detection API integration
_Exported on 5/10/2026 at 01:39:41 PDT from Cursor (3.3.27)_

---

**User**

I have used Android's Face Detector API in a separate project, just to detect/count number of faces, and getting the bounding box for each face. I use a simple threshold ratio to reduce false positives.

The API is this: https://developer.android.com/reference/kotlin/android/media/FaceDetector 

Threshold ratio works like this: (1) Identify the biggest/largest bounding box by area, (2) Calculates areas of all other faces' bounding box, (3) Discard any faces with bounding box that are more than 35% smaller in area than the largest face. 

Do you think it can be leveraged alongside our full range or short range models?

---

**Cursor**

Short answer: I'd push back on adding `android.media.FaceDetector`, but the **size-ratio idea itself** is worth folding in — and you can do that today without bringing the legacy detector along. They're two independent things.

## Why `android.media.FaceDetector` is a poor companion to BlazeFace

The API has aged badly and overlaps almost exactly with the niche your current short-range model already covers:

- **API 1 era, effectively unmaintained.** It still exists, but Google itself has long moved everyone to ML Kit Face Detection.
- **Bitmap format friction.** It demands `Bitmap.Config.RGB_565` with even width/height. Your pipeline runs `BitmapDecoder.decode → ARGB_8888` and feeds that into BlazeFace ([`FaceProcessor.process`](app/src/main/kotlin/com/alifesoftware/facemesh/ml/FaceProcessor.kt#L66-L114)). You'd need a second 565 conversion per image just to feed it.
- **Frontal-only, single threaded, CPU only.** That's the same envelope `BlazeFaceShortRangeDetector` is trained for ("selfie-distance frontal faces filling 30–80% of the frame", per `FaceDetector.kt:7-12`). On full-range / group photos / off-axis faces — exactly where you currently get a 2× recall lift from full-range — `android.media.FaceDetector` gets dramatically worse than even short-range BlazeFace.
- **Sparse output.** It only gives you `eyesDistance` and an eye midpoint. You can't use it to enrich the 6-point BlazeFace landmarks that drive your `FaceAligner` similarity fit, and you can't really replace `FaceFilters.eyeWidthRatioMin/Max`'s geometry check with it because the box is implicit (derived from `eyesDistance × confidence`).
- **Per-call cost.** Whether you use it as the primary detector, as a second-pass consensus check ("BlazeFace and the legacy detector both agree"), or only as a falsification step, you're paying another full pass over the bitmap on the CPU for each image. Given your `Detector.scoreThreshold = 0.55` was tuned to recover faraway faces, the legacy detector — which is *worse* at distant/small faces — would mostly vote against the boxes you specifically care about keeping.

If you really want an ensemble, ML Kit's face detector (`com.google.mlkit:face-detection`) is the modern equivalent: works on ARGB_8888, has FAST / ACCURATE modes, returns contours and head Euler angles, and runs off-thread. But I'd argue even that is over-engineering on top of the filtering you already have.

## On the 35%-of-largest-area heuristic itself

This is a **good idea, and you already have a near-cousin shipping** in [`FaceFilters`](app/src/main/kotlin/com/alifesoftware/facemesh/ml/FaceFilters.kt#L23):

```236:259:app/src/main/kotlin/com/alifesoftware/facemesh/config/PipelineConfig.kt
EXTREME_OUTLIERS_ONLY,
```

`SizeBandMode.EXTREME_OUTLIERS_ONLY` drops `width < 0.25 × median` or `width > 4.0 × median`. It differs from your version on two axes:

| Dimension | Your old project | Current `FaceFilters` |
|---|---|---|
| Anchor | **largest** face's area | **median** face's width |
| Metric | area (w × h) | width (w only) |
| Cull rule | drop if ≥35% smaller in area than the largest | drop if outside `[0.25 × median, 4.0 × median]` width |

Tradeoffs worth being honest about:

- **Largest-anchored is more aggressive.** On a portrait + small-far-subject group shot the largest face dominates and the far subject gets killed. This is exactly the regression that `groupPhotoSizeBandSkipThreshold = 4` and the move from `SYMMETRIC_BAND` to `DISABLED` were put in to fix (see the KDoc on `SizeBandMode.SYMMETRIC_BAND` and `Detector.defaultVariant` in `PipelineConfig.kt`). A median-anchor degrades more gracefully because two big-and-two-small faces give you a sane median; max-anchor doesn't.
- **Area is more discriminating than width.** A 100×100 face vs 70×70 face is 2× area but only 1.4× width. Area tracks "this looks much smaller than the others" the way humans perceive it. So *the metric* in your old version is arguably better than what the current `EXTREME_OUTLIERS_ONLY` uses.
- **35% of largest area** ≈ keep faces with `√(0.65) ≈ 0.806` × the largest face's width or larger. That's stricter than `0.25 × median` in the common case, looser in pathological cases where the largest face is itself an outlier.

## What I'd actually do, if I were you

Don't add the legacy detector. Instead, treat your old project's heuristic as a recipe and add it as a third `SizeBandMode` in the existing dispatch, where it's free and well-understood:

```kotlin
enum class SizeBandMode {
    DISABLED,
    EXTREME_OUTLIERS_ONLY,    // existing, median-anchored, width-based
    SYMMETRIC_BAND,            // existing
    LARGEST_AREA_RATIO,        // new: drop faces whose area < (1 - X) × maxArea
}
```

Three things make this fit cleanly into the current architecture:

1. The dispatcher in [`FaceFilters.apply`](app/src/main/kotlin/com/alifesoftware/facemesh/ml/FaceFilters.kt#L91-L119) already has the `DISABLED` / `n ≤ 1` / `n ≥ groupPhotoSizeBandSkipThreshold` short-circuits — keeping `LARGEST_AREA_RATIO` under the same `groupPhotoSizeBandSkipThreshold = 4` guard prevents the very regression that pushed you to `DISABLED` in the first place.
2. Your old 35% threshold becomes a single new `Filters.largestAreaRatioMin = 0.65f` in `PipelineConfig` (i.e., "must be at least 65% of largest face's area"), with the same KDoc shape every other filter knob already follows.
3. `DetectedFace.boundingBox: RectF` already exposes `width × height`, so the implementation is one `maxOf` over the candidate list and one filter pass — no new model, no new format conversion, no extra JNI hop.

This gives you the "second opinion" feeling your original project had — *we can prune false positives that BlazeFace was overconfident about* — without inheriting `android.media.FaceDetector`'s 2010 baggage. And because it runs after BlazeFace's own NMS and after the confidence/eye-distance filters, by the time it sees the candidate list there's normally only 1–3 boxes left, so even on group photos `groupPhotoSizeBandSkipThreshold` will short-circuit it correctly.

If you'd like I can sketch the diff for `FaceFilters.kt` + `PipelineConfig.kt` + a unit test mirror of `FaceFiltersTest.kt` — but you'd need to flip me out of Ask mode for me to actually apply it.

---

**User**

Let's wait. Another thought; can I run the picked images both thru short and full range models? get an aggregate result, and then go for clustering? Or does that not make sense?

---

**Cursor**

It's a reasonable instinct, but in your specific setup the math leans against it — and there's a Pareto-better variant worth considering instead.

## What an ensemble actually costs

Running both detectors is more than just "2 inferences per image". You have to glue them together:

1. **Cross-detector NMS.** Both models will fire on the same face. You need a second NMS pass over the *union* of candidates with some IoU threshold to dedup. That's a new stage to maintain.
2. **Score calibration mismatch.** Your current `Detector.scoreThreshold = 0.55f` is documented as "*deliberately shared across variants for now; if the two detectors' logit calibrations diverge, split into `ShortRange.scoreThreshold` and `FullRange.scoreThreshold` later*" (`PipelineConfig.kt:128`). The moment you start comparing scores across the two detectors during dedup ("which copy of this face do I keep?"), that "deliberately shared" assumption stops being benign.
3. **Landmark conflict.** When both models see the same face, their 6-point predictions differ. `FaceAligner` does a 4-point similarity fit against the ArcFace canonical template (`PipelineConfig.kt:406-411`); the alignment quality — and therefore the embedding quality — depends on which set of landmarks you pick. There's no clean way to merge two landmark sets, so it's a "pick one" call, and your only signal for picking is the not-comparable score.
4. **Latency.** Short-range at 128² ≈ ~40-50% the FLOPs of full-range at 192² (and short-range has 896 anchors vs full-range's 2304). So the ensemble path is roughly **1.5× of full-range alone**, not 2× — but on top of decode + filter + align + embed + DB write, every photo gets that overhead unconditionally.
5. **Memory.** `MlPipelineProvider` (`MlPipelineProvider.kt:42-47`) currently caches *one* detector and rebuilds on variant change. Ensemble = both interpreters resident, both anchor tables, both decoder buffers. Not a deal-breaker, but it's a real ~few-MB-RSS bump on lower-end devices.

## What does your own evidence say about the upside?

This is the part I'd weight heaviest. From `Detector.defaultVariant` KDoc (`PipelineConfig.kt:96-110`):

> empirical real-world testing on a 25-photo set showed full-range detected **32 faces vs short-range's 15** (more than 2x recall), **recovered every photo short-range was returning 0 faces on**, and produced **tighter same-person cosine distances (best 0.17 vs 0.23)**.

Two things stand out:

- **"Recovered every photo short-range was returning 0 faces on."** That's the strongest counter-argument to ensembling. The headline failure mode of short-range — "this photo had no detectable faces" — is already solved by full-range. The ensemble doesn't add anything on those photos.
- **"Tighter same-person cosine distances (best 0.17 vs 0.23)."** When both detectors *do* detect the same face, full-range's bbox/landmarks produce better embeddings. So on the overlap, you'd be picking full-range's output anyway — short-range is contributing nothing on the dedup'd majority.

That leaves only the "photos where short-range finds a face full-range misses" bucket as the ensemble's actual win condition. On the 25-photo test set that bucket appears to be ~empty. It's not provably empty in general — short-range *might* have higher precision on close-up frontal selfies where its prior is tighter — but it's a thin lane.

## The Pareto-better alternative: cascade, not ensemble

If you genuinely want to extract value from owning both models, run them **sequentially with a fallback**, not in parallel:

```
short_range_results = shortRange.detect(bitmap)
if (short_range_results is empty OR all confidence < someBar):
    return fullRange.detect(bitmap)
else:
    return short_range_results
```

Or the inverse, since full-range is your current default:

```
full_range_results = fullRange.detect(bitmap)
if (looks_like_close_up_frontal && full_range_results.size <= 1):
    short_range_results = shortRange.detect(bitmap)
    if (short_range_results.size > full_range_results.size) return short_range_results
return full_range_results
```

The cascade gives you ensemble-style "if one misses, the other catches it" recall **without** paying for the second inference on the ~majority of photos where the first one already did its job. It also dodges the score-calibration problem because you never have to compare scores across detectors — you pick a winner *per image*, not per face.

The "looks like close-up frontal" gate is the trickier bit; you could approximate it as `full_range_results.size == 1 && bbox covers >40% of frame`, since that's exactly where short-range's tighter prior could plausibly outperform.

## Where I'd actually spend my next bit of effort

Before building either an ensemble or a cascade, I'd want a measurement: **on your real-world test set, how many photos does full-range alone miss that short-range alone catches?** The answer is the upper bound on what *any* combining strategy can buy you. If it's <5% of photos, neither approach is worth the complexity. If it's 15-20%, cascade is genuinely interesting. The 25-photo notes in `defaultVariant`'s KDoc strongly suggest the first scenario, but it's an n=25 sample and you'd want to confirm before architecting around it.

If you're set on trying *something* in this space, my ranking would be:

1. **Tune full-range's `scoreThreshold` / NMS** — cheap, no new code paths, directly trades precision/recall on the model you already trust most.
2. **Cascade with full-range primary, short-range fallback for empty-or-single-face frames** — modest code, asymmetric latency cost, hits the documented failure modes.
3. **Add the largest-area-ratio filter we discussed** — orthogonal to detector choice, addresses precision instead of recall.
4. **True parallel ensemble with cross-detector NMS** — most code, most latency, smallest expected win given your own evidence.

So: the question makes sense, but I'd push you toward the cascade variant, and I'd push *that* behind a measurement step before committing to it.

---

**User**

I did mean running them as cascade - full range, fall back short range - not running them simultaneously.

Does that add any value? Feel free to push back

---

**Cursor**

Honest pushback: **no, full-range → short-range cascade specifically adds essentially nothing in your setup.** The direction is upside-down for the evidence you have.

## Why this direction is the wrong one

Your own measurement in `PipelineConfig.kt:96-110` is the kill shot for it:

> full-range detected 32 faces vs short-range's 15 (more than 2x recall), **recovered every photo short-range was returning 0 faces on**

Read that as a Venn diagram:

```
short-range catches:   {A}     (15 faces)
full-range catches:    {A ∪ B} (32 faces, 17 of which short-range missed)
short-range-only set:  {} ← empty, on this test set
```

Your fallback `if fullRange returned 0 → try shortRange` only fires when full-range returns 0. The data says: when full-range returns 0, short-range was *already* returning 0 on those photos before. The fallback path will basically never produce a face, because there's no documented case where short-range can see something full-range can't.

There are theoretical reasons to expect this asymmetry to hold beyond the 25-photo sample:

- **Anchor density.** Short-range has 896 anchors over a 16×16 + 8×8 grid; full-range has 2304 anchors over a single 48×48 grid (`PipelineConfig.kt:165, 189`). Full-range's grid is denser at every scale short-range covers — anything short-range's anchors lock onto, full-range has anchors for too.
- **Training distribution is a subset.** Short-range was trained on close, frontal, well-lit faces. Full-range's training distribution includes that subset *plus* small, off-axis, group-photo cases. There's no face-size band that short-range covers and full-range doesn't.
- **Input resolution.** 192² preserves more detail than 128² before the network even runs. For a small face in a large source bitmap, full-range's letterboxing sees more pixels per face than short-range does. So even on a "tiny near-frontal face" — short-range's home turf — full-range starts with more to work with.

The only mechanism by which short-range could catch something full-range misses is **calibration**: short-range might score a given anchor 0.62 where full-range scores it 0.51 (below your `Detector.scoreThreshold = 0.55f`). That's possible in principle, but:

1. The fix is **lower full-range's threshold**, not run a second model. One config tweak vs a new code path.
2. The same calibration argument also says short-range will produce **more false positives** on full-range's harder territory (off-axis / occluded / group). So you'd buy a tiny bit of recall in one bucket and import a precision regression in another.

## And there's a quality penalty even when it does fire

If you do rescue a face on the short-range fallback, the same KDoc warns:

> produced tighter same-person cosine distances (**best 0.17 vs 0.23**)

Short-range's bbox + landmark predictions feed `FaceAligner`'s 4-point similarity fit (`PipelineConfig.kt:406-411`); the alignment quality drives `FaceEmbedder` which drives DBSCAN. Short-range's outputs sit in the looser end of that distribution. So a "rescued" face has a higher chance of being marked as DBSCAN noise (`Clustering.defaultEps = 0.35f`, `PipelineConfig.kt:503`) or clustering wrong, which means the rescue itself is partially wasted compute.

## The opposite cascade has actual merit

Short-range → full-range fallback (cheap-first) is the direction the data supports:

- Short-range succeeds on the easy half of your library (close, frontal, well-lit). On those photos you save ~50% of detection time.
- When short-range returns 0 (or 1 face on what was clearly a multi-person frame, where N is uncertain), you fall back to full-range and get the recall you have today.
- Worst case: a photo where full-range would have found 1 face, short-range also finds 1 face — you used the worse-aligned bbox. That's a real cost; whether it matters depends on how the embedding-distance gap of 0.06 (0.17 vs 0.23) translates over a larger sample.

But you said "no parallel running" and asked specifically about full-range → short-range. For *that* direction, my recommendation is to drop the idea.

## Cheap experiment if you want to be sure

Before writing any cascade code, the cheap way to falsify my "short-range never adds anything full-range missed" claim is:

1. On your real photo library (not the 25-photo set), run full-range and collect the photos where it returned 0 faces.
2. On *just that subset*, run short-range.
3. Count: how many of those photos does short-range find ≥1 plausible face on?

If the answer is ~0 (which I expect), the cascade is dead in the water — no architecture can buy you what isn't there. If it's a meaningful fraction, then short-range's marginal contribution is real and you can re-evaluate. This is purely a `tools/` script — no production code change.

So: the cascade direction you're proposing is, on the evidence you yourself wrote down, doing work to recover faces that aren't there to be recovered. I'd skip it.

