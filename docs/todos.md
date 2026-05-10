Two big topics — let me lay them out so we can pick what to tackle first.

# Part 1 — Smoke test plan

The "smoke test" is the first end-to-end run on real hardware. It tells us whether the integration we've built actually behaves correctly when fed real photos and a real Play Services TFLite runtime. There are roughly **eleven milestones** we want to confirm in order, each with its own way to fail.

## The eleven milestones

| # | Milestone | What "success" looks like | How it can fail |
|---|---|---|---|
| 1 | App installs | `adb install` exits 0; app icon appears on home screen | Signing mismatch, sdk too old |
| 2 | Cold-launch renders Empty state | Pure black screen, "FaceMesh" title, centered + button | XML theme flash → light mode briefly visible |
| 3 | Photo picker opens | System Photo Picker UI appears | Permissions, Photo Picker not available on API < 30 fallback |
| 4 | Selecting state appears | "N photos selected", thumbnail fan, Clusterify button | Coil failing to load `content://` thumbnails |
| 5 | Settings → dynamic color toggles work | Theme re-renders live | Preference plumbing |
| 6 | Tap Clusterify → model download visible | Spinner with "Downloading face models" + progress bar | Network errors, SHA mismatch, GitHub LFS pointer issue |
| 7 | TFLite runtime initializes | (Logcat shows `Using GPU` or `Using CPU` toast appears) | GPU delegate refusal, Play Services TFLite not on device |
| 8 | Detection + embedding actually runs | Per-image progress counts up; processing finishes | Wrong tensor shapes, NHWC/NCHW mismatch, OOM |
| 9 | Clusters appear with circular avatars | Bottom row populated, count matches the number of unique people in the test set | DBSCAN params wrong → too many/too few clusters; representative crop empty |
| 10 | Filter flow produces correct keepers | Tap cluster → tap Camera → pick filter photos → tap Filter → swipeable keepers gallery | Cosine threshold wrong → all-keep or zero-keep |
| 11 | Reset wipes everything | Returns to Empty state cleanly | Stale Room rows, leftover representative thumbnails on disk |

## What I want to add before the smoke test

The app is currently silent (no `Log.d` anywhere — I deliberately stripped them in `proguard-rules.pro`). For diagnosis we want **rich debug logging at the seven choke points** that fail most often:

- `TfLiteRuntime` — log delegate selection
- `ModelDownloadManager` — log each progress event + final SHA
- `BlazeFaceDetector` — log discovered output port indices, per-call detection count
- `FaceFilters` — log how many candidates dropped at each stage
- `FaceProcessor` — log per-image (decode_ms / detect_ms / embed_ms / face_count)
- `ClusterifyUseCase` — log DBSCAN result (input N → cluster count + noise count)
- `FilterAgainstClustersUseCase` — log per-image (max_similarity, threshold, kept?)

These are tagged `FaceMesh` and stripped from release builds by the existing R8 rule. So in release nothing is logged, in debug we get a clear story in `adb logcat | grep FaceMesh`.

## How the smoke test will run

Step-by-step (assuming you have a phone with USB debugging on, or a Pixel emulator with Google Play image API 33+):

```bash
# 1. Build + install the debug APK
cd /Users/gifty/Development/FaceMesh
./gradlew :app:installDebug

# 2. Open a logcat window in another terminal — we want to see things as they happen
adb logcat -c                                    # clear old log
adb logcat FaceMesh:D AndroidRuntime:E '*:S'     # only our tag + crash trace

# 3. Launch the app
adb shell am start -n com.alifesoftware.facemesh.debug/com.alifesoftware.facemesh.MainActivity
```

Then a 2-minute manual flow on the device:

1. Pick **6–10 photos** containing **3 distinct people** (some with multiple people in one shot is great — proves the multi-face path).
2. Tap Clusterify.
3. Watch logcat for the milestone events. Expected total time on a Pixel 6: 5–15 seconds for 10 photos.
4. Verify cluster count matches your eyeball count of unique people.
5. Tap one cluster → Camera → pick **5 fresh photos** (mix of "should match" and "should not match").
6. Tap Filter, verify the swipeable gallery only shows the matches.
7. Reset → confirm app is back to Empty.

The recommended test set for repeatability: pick a folder where you know exactly how many distinct people appear — your own family album works great.

## What might fail and what to do

The most likely first-run failure modes, ranked by probability:

1. **DBSCAN clusters too aggressively or too conservatively.** GhostFaceNet 512-d cosine distances behave differently from 128-d MobileFaceNet. We may need to tune `eps` (currently 0.35 in manifest config). The fix is one HTTP edit to `manifest.json` in the GitHub Release — no app rebuild required, the next launch picks up the new config.
2. **Cosine threshold too tight or too loose.** Same story — `match_threshold` (0.65) is editable in the manifest.
3. **GPU delegate refused on emulator.** Common on emulators without Vulkan. The XNNPACK CPU fallback should kick in automatically; the toast confirms.
4. **OOM on huge bitmaps.** Our `BitmapDecoder` caps to 1280px. If we still OOM on a 100MP Samsung shot, we lower the cap.

I'll write up the **calibration-via-manifest-edit** workflow so you can iterate on `eps` / `threshold` without me in the loop.

---

# Part 2 — Other open questions

Tiered by urgency. Things in **Tier A** I'd want to settle before any user installs the app; **Tier B** can wait until first-run feedback; **Tier C** is v1.1 territory.

## Tier A — settle before any user-facing build

| Question | Notes |
|---|---|
| **Cluster naming UI** | Currently deferred to v1.1, but seeing 4 unnamed circles is bad UX. Easiest cut: long-press a cluster → text-edit dialog → store in the existing `ClusterEntity.name` column. ~1 hour of work. |
| **Long Clusterify in background** | If the user backgrounds the app mid-pipeline (incoming call, swipe to home), our `viewModelScope`-based pipeline survives but the OS may kill the process. A `WorkManager` foreground service would let the work finish + notify when done. Worth doing for any photo set ≥ ~30 images. |
| **Empty-state reachability of Settings** | Right now the gear is in `TopActions`, which doesn't render in the Empty state (you have to add at least one photo to see it). Easy fix: always show the gear regardless of state. |
| **What happens if a model SHA changes?** | Currently we re-download from scratch. That's correct; just calling it out so we don't get surprised. |

## Tier B — informed by smoke-test feedback

| Question | Notes |
|---|---|
| **OQ-3 DBSCAN noise faces** | When DBSCAN labels a face as noise (-1), we silently drop it. After smoke test we'll know whether this is rare (drop is fine) or common (we should surface as an "Unclassified" pseudo-cluster). |
| **OQ-5 Threshold calibration** | Empirical. Iterate via manifest edits (Tier B because we need the smoke test to know the real distribution). |
| **DBSCAN minPts for tiny inputs** | If user picks 2 photos and `minPts=2`, both faces must already be similar to cluster — otherwise they're noise → "no faces found". Maybe scale minPts dynamically with input size. |
| **Re-clustering when adding photos to existing clusters** | Right now each Clusterify run is independent (creates fresh clusters from the batch). Adding "incremental clustering" (merge new faces into existing clusters when similar enough, else create new) is meaningful UX work but can wait. |
| **Cluster sort order** | Currently `createdAt ASC` (oldest first). Probably want `faceCount DESC` (largest cluster first) for visual clarity. |

## Tier C — v1.1+

| Question | Notes |
|---|---|
| **OQ-4 Keeper output** | Save matched photos to `Pictures/FaceMesh/Keepers/` MediaStore folder so they're visible in the system Photos app. |
| **Share keepers** | Add a Share action in `KeeperGalleryScreen` (placeholder already exists). |
| **App icon glyph** | The current `ic_launcher_foreground.xml` is my placeholder smiley. Brand identity question. |
| **Centroid auto-update during Filter** | Every confirmed match could enrich the cluster centroid for better future matches. Off by default in v1; consider an opt-in confirm mode. |
| **Live camera capture as input** | Big feature; nice-to-have. |
| **Cluster merging** | Two clusters that are actually the same person → user-driven merge UI. |
| **Privacy policy / About / version screen** | Required for Play Store. |
| **Release signing keystore** | Required for Play Store. Currently release builds are debug-signed. |
| **Periodic manifest re-check** | If we publish `facemesh-v2` models, current app won't notice unless reinstalled. Could check manifest weekly. |

---

# What I'd suggest doing next, in order

1. **Add debug logging** at the seven choke points (~15 min). Makes the smoke test 10× more diagnostic.
2. **Tier A: always-visible Settings gear + cluster naming UI** (~30 min total). Both are tiny but visible UX wins.
3. **Run the smoke test together** — you on the device, me parsing logs / proposing manifest edits as needed.
4. **Settle Tier B** based on what we see.

If you agree, I'll start with step 1. Or, if you'd rather, we can skip straight to a barebones smoke test now (no logging, no Tier A polish) just to see if the happy path even works — it's a quick "does it install and not crash" check, then we add stuff after.

Which path do you want?