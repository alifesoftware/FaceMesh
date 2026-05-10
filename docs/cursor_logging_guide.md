# Logging levels for image processing
_Exported on 5/10/2026 at 00:14:16 PDT from Cursor (3.3.27)_

---

**User**

I want lot of logs in the app, especially during image inference, model input, model output, face detection, image cropping, face clustering, face filtering etc CORE areas. I want them to be Log.i, not Log.d Of course, warnings and errors should continue to use .w and .e. But yeah more logs the better

---

**Cursor**

Let me check test files and any other places we'd need to modify.

Here's my plan: I'll add a single `FaceMesh.<ClassName>` tag pattern across the core ML/decode/cluster/filter pipeline so you can filter cleanly with `adb logcat -s 'FaceMesh.*'`. All happy-path traces use `Log.i`; warnings (e.g. tolerated per-image failures, EXIF read fails) become `Log.w`; hard failures become `Log.e`. First, I'll allow `android.util.Log` calls in the pure-JVM unit tests so the existing `DbscanTest`/`EmbeddingMathTest` keep passing once those classes log.

## 1. Make Log.* safe in plain-JVM unit tests

## 2. BitmapDecoder — image decode + crop

## 3. BlazeFaceDetector — model input/output + inference

## 4. BlazeFaceDecoder — anchor decode + NMS

## 5. FaceFilters — face filtering stages

## 6. FaceAligner — face cropping/alignment

## 7. FaceEmbedder — embedding inference + I/O

## 8. FaceProcessor — per-image pipeline

## 9. Dbscan — face clustering

## 10. ClusterifyUseCase — high-level cluster phase

## 11. FilterAgainstClustersUseCase — high-level filter phase

## 12. TfLiteRuntime + MlPipelineProvider — graph init + delegate

Now `MlPipelineProvider`:

## 13. ModelDownloadManager — downloads + retries

Now let me update my todos and run the unit tests + a compile check.

Let me check for lint errors and also check that I haven't missed an unused import in FaceProcessor (the `RectF` import was already there).

Now let me run the unit tests to make sure nothing's broken.

All tests pass and the project compiles cleanly with the new logging.

Done. Here's a summary of what changed across the core pipeline.

## Tag convention

Every class uses a `FaceMesh.<Component>` tag so you can filter logcat cleanly:

```bash
adb logcat -s 'FaceMesh.Decoder:I' 'FaceMesh.Detector:I' 'FaceMesh.Embedder:I' \
              'FaceMesh.Aligner:I' 'FaceMesh.Filters:I' 'FaceMesh.Processor:I' \
              'FaceMesh.Dbscan:I' 'FaceMesh.Clusterify:I' 'FaceMesh.Filter:I' \
              'FaceMesh.Pipeline:I' 'FaceMesh.TfLite:I' 'FaceMesh.ModelDl:I'
```

Or just everything: `adb logcat -s 'FaceMesh.*:V'` (which still includes `Log.w` / `Log.e`).

## What now logs at `Log.i`

| Component | Tag | What it logs |
|---|---|---|
| `BitmapDecoder` | `FaceMesh.Decoder` | start uri, raw bounds, sample size, decoded dims+config, EXIF orientation/transform, total decode time |
| `BlazeFaceDetector` | `FaceMesh.Detector` | init (model file/size, input dims, output tensor names + indices), per-call source dims, prep time, inference time, raw logits min/max, # candidates, top scores |
| `BlazeFaceDecoder` | `FaceMesh.Decoder.Anchor` | source dims, threshold, candidates above threshold, kept after NMS |
| `FaceFilters` | `FaceMesh.Filters` | input count, count after each of confidence/geometry/size-band stages, median width + band |
| `FaceAligner` | `FaceMesh.Aligner` | source dims → output dims, bbox, eye distance, score, produced crop dims |
| `FaceEmbedder` | `FaceMesh.Embedder` | init (model file/size, input/embedding dims), per-call prep+infer time, first 4 components of L2-normalized embedding |
| `FaceProcessor` | `FaceMesh.Processor` | per-image start/end with uri, decode time, candidate count, post-filter count, per-face align+embed time + score + bbox + display crop dims, total time; also logs the display-thumbnail cropping decision |
| `Dbscan` | `FaceMesh.Dbscan` | start (n, eps, minPts), each cluster close, final clusters/noise/sizes + time |
| `ClusterifyUseCase` | `FaceMesh.Clusterify` | full phase trace: source count, eps/minPts, per-image faces added, processing-phase time, DBSCAN time, grouping result, per-cluster persisted (id, members, thumb source), total time |
| `FilterAgainstClustersUseCase` | `FaceMesh.Filter` | input sizes, threshold, loaded centroids, per-image faces + best dot score + keep decision, final keepers + time |
| `TfLiteRuntime` | `FaceMesh.TfLite` | GPU-delegate probe result, init time, model mmap size, interpreter creation (delegate, threads, time) |
| `MlPipelineProvider` | `FaceMesh.Pipeline` | first-call graph build (manifest version, delegate, model names), graph teardown |
| `ModelDownloadManager` | `FaceMesh.ModelDl` | cache hit/miss, per-model download (url, time, size), SHA-256 verification result |

## Warnings/errors (`Log.w` / `Log.e`)

Previously-silent failures now surface:
- `BitmapDecoder.applyExifRotation`: EXIF read failure → `Log.w` with throwable.
- `BitmapDecoder.decode`: null bitmap → `Log.e` before throw.
- `ClusterifyUseCase` per-image catch: `Log.w` with the exception.
- `FilterAgainstClustersUseCase` per-image catch: `Log.w` with the exception.
- `MlPipelineProvider.close` / detector / embedder `close()`: any throw → `Log.w`.
- `ModelDownloadManager`: each retry attempt → `Log.w`; exhaustion → `Log.e`; SHA mismatch → `Log.e`; missing manifest → `Log.e`.
- `FaceProcessor.cropDisplayThumbnail`: degenerate input/bbox → `Log.w` (was returning null silently).

## Build/test plumbing

`unitTests.isReturnDefaultValues = true` in `app/build.gradle.kts` so the pure-JVM tests (`DbscanTest`, `EmbeddingMathTest`, `BitmapDecoderTest`) keep passing now that `Dbscan` and `BitmapDecoder` log via `android.util.Log`. Robolectric tests are unaffected.

Tests run green: `./gradlew :app:testDebugUnitTest --rerun-tasks` → BUILD SUCCESSFUL.

---

**User**

Are the logs done?

---

**Cursor**

Yes — logging is fully wired up across all the core areas you called out, the project compiles cleanly, and the unit-test suite passes (`./gradlew :app:testDebugUnitTest --rerun-tasks` → BUILD SUCCESSFUL).

Quick recap of what got `Log.i` coverage:

- **Image decode + crop** — `BitmapDecoder` (uri, bounds, sample size, decoded dims, EXIF, total time)
- **Model input/output + inference** — `BlazeFaceDetector`, `FaceEmbedder` (model file/size, input dims, prep time, infer time, output sample)
- **Face detection** — `BlazeFaceDecoder` (candidates above threshold, kept after NMS), `BlazeFaceDetector` (top scores, # faces)
- **Image cropping / face alignment** — `FaceAligner`, `FaceProcessor.cropDisplayThumbnail`
- **Face filtering** — `FaceFilters` (per-stage drop counts, median+band)
- **Per-image pipeline** — `FaceProcessor` (decode/detect/filter/align+embed timings, score, bbox per face)
- **Face clustering** — `Dbscan` (eps, minPts, clusters, noise, sizes, time), `ClusterifyUseCase` (full phase trace)
- **Filter phase** — `FilterAgainstClustersUseCase` (threshold, centroids, per-image best score + keep/skip)
- **Runtime + graph init** — `TfLiteRuntime` (GPU probe, delegate, init time), `MlPipelineProvider` (graph build/teardown)
- **Model downloads** — `ModelDownloadManager` (URLs, retries, SHA verification)

`Log.w` / `Log.e` now surface previously-silent paths: per-image exceptions in both use cases, EXIF failures, SHA-256 mismatches, retry exhaustion, interpreter close failures, etc.

Tag scheme is `FaceMesh.<Component>` so you can filter logcat with:

```bash
adb logcat -s 'FaceMesh.*:V'
```

If you want, I can also wire logs into the UI/ViewModel layer (e.g. `HomeViewModel` intent handling, model-download progress events) — that's outside the "core ML" scope you listed but it'd give you end-to-end traces from button-tap to keepers shown. Just say the word.

---

**User**

I want more logs, especially in any sort of calculations, where we discard things or include things etc, or we do early exists

---

**Cursor**

Let me re-read the current state of the files I'll modify so I work against accurate content.

## 1. BlazeFaceDecoder — anchor-level discard counts + NMS group merges

## 2. FaceFilters — per-face discard reasons

## 3. EmbeddingMath — log zero-norm + meanAndNormalize details

## 4. Dbscan — per-point classification

## 5. FaceAligner — log landmarks + canonical mapping

## 6. FaceProcessor.cropDisplayThumbnail — log clamp/shift decisions

## 7. FilterAgainstClustersUseCase — per-face per-centroid score detail

## 8. ClusterifyUseCase — pickRepresentative reasoning + member URIs

## 9. BitmapDecoder.computeInSampleSize — log halving iterations

Now let me check the tests and lints, then run the test suite.

Done. Build green, all unit tests pass. Here's what got deeper visibility this round, focused on calculations, include/discard decisions, and early-exit paths.

## Per-decision logs added

**`BlazeFaceDecoder.decode`** (`FaceMesh.Decoder.Anchor`)
- For every anchor: per-anchor `KEPT` line with score, bbox, anchor center, regressor offset/size; or `dropped degenerate rect` with raw l/t/r/b values.
- End-of-loop summary: `topLogit`, `topScore`, `droppedLowScore`, `droppedDegenerate`, `kept` totals.
- NMS `weightedNms`: per-group log on each absorbed overlap (anchor score, other score, IoU value), per-group merge summary, final group count. Early `empty` return is logged.

**`FaceFilters.apply`** (`FaceMesh.Filters`)
- Per-face `DROP confidence` (with score vs threshold), `DROP geometry width<=1`, `DROP geometry eyeDist/width=…` (with the actual ratio + bounds), or `PASS geometry`.
- Size-band stage: full sorted widths list, median, band, per-face `PASS size-band` / `DROP size-band` with deviation.
- Empty-input early return logged.

**`EmbeddingMath`** (`FaceMesh.EmbedMath`)
- `l2NormalizeInPlace`: norm + scale factor on the normal path; `Log.w` zero-fill warning when norm < 1e-8.
- `meanAndNormalize`: dim + count.

**`Dbscan.run`** (`FaceMesh.Dbscan`)
- Per-point classification: `MARK noise`, `CORE seeding cluster=…`, `CORE expanding cluster=…`, `BORDER cluster=…`, `PROMOTE noise->cluster=…`, `SKIP already labelled=…`.
- Per-cluster close: member count.
- `regionQuery` per call: neighbour count and distance range `[min..max]` so you can see how tight the cluster is vs `eps`.
- Done summary now includes `corePoints`, `borderPoints`, `expansions`, plus the existing sizes/noise/time.

**`FaceAligner.align`** (`FaceMesh.Aligner`)
- Logs the four source landmarks (right-eye, left-eye, nose, mouth) and the canonical destination quad before `setPolyToPoly`.
- `Log.w` if `setPolyToPoly` returns `false` (degenerate landmark set), with `polyMapOk` reported on the produced-crop log.

**`FaceProcessor.cropDisplayThumbnail`** (`FaceMesh.Processor`)
- Logs the `CLAMP cropSize` decision when the padded square exceeds the source's min side.
- Logs each `SHIFT left/top/right/bottom` with the underflow/overflow magnitude.
- Logs `FINAL CLAMP` with before/after rect when the pathological-input clamp actually changes anything.
- Existing early-exit `Log.w`s on tiny source / degenerate bbox stay.

**`FilterAgainstClustersUseCase.run`** (`FaceMesh.Filter`)
- Per-face × per-centroid dot-product line, so you see every comparison: `face[i] vs centroid[j] score=… (threshold=…)`.
- `KEEP` line names exactly which face × centroid pair short-circuited the match.
- `DROP` line shows the best face/centroid pair and its score vs threshold.
- Separate `DROP no faces detected` early-exit log.

**`ClusterifyUseCase`** (`FaceMesh.Clusterify`)
- Per-cluster pre-persist log with the full list of `sourceUris` belonging to that cluster.
- `pickRepresentative` now emits its decision: chosen URI, face index, crop area, and the candidate ratio (`X/Y candidates with crops`). Falls back path uses `Log.w`.

**`BitmapDecoder.computeInSampleSize`** (`FaceMesh.Decoder`)
- Each halving step (`#1`, `#2`, …) with the running estimate; final summary line with total halvings; and a "already fits" line for the no-op case.

## Tests / build

`./gradlew :app:testDebugUnitTest --rerun-tasks` → **BUILD SUCCESSFUL** (33 tasks). No new lint warnings.

One caveat worth flagging: the Dbscan `regionQuery` log fires once per scan (so `O(n)` per `run` call, not `O(n²)`) but the per-point classification logs in `Dbscan` and the per-anchor logs in `BlazeFaceDecoder` can produce a lot of lines for big batches. If you ever want to dial back specific tags without touching code, you can use `adb shell setprop log.tag.FaceMesh.Decoder.Anchor SILENT` (or `WARN`) at runtime. Let me know if you'd like a debug-only toggle baked in.

---

**User**

Also need detailed logging in Face Detection - BlazeFaceDetector.kt

---

**Cursor**

Build green, tests pass. `BlazeFaceDetector` (`FaceMesh.Detector` tag) now produces a much richer trace covering every interesting computation, decision, and early exit:

**At interpreter init**
- Builds-interpreter line with model name, size, absolute path.
- Per-input tensor: name, shape, dtype, byte count.
- Per-output tensor: name, shape, dtype, byte count.
- Resolution decision: which output index resolved to "regressors" vs "classifications".
- Final ready line: input shape, anchor count, regressor stride, active TFLite delegate.

**At `detect()`**
- Start line: source dims, bitmap config, byte count, target input shape.
- `Log.e` + early `emptyList()` return if the source bitmap is recycled.
- `prepareInput` timing line.
- Inference exception path: catches throwables from `interpreter.run`, logs elapsed time before re-throwing.
- Logit-distribution stats per call: `min/mean/max`, count of positive logits, count exceeding `sigmoid > 0.5`, count exceeding the typical `0.75` threshold, plus the top-anchor index and its raw logit.
- For the top anchor: its decoded `dx, dy, w, h` regressor values so you can see what the model "thinks" the strongest face looks like.
- Final line: face count, decode time, top-3 scores, top-3 bboxes.

**At `prepareInput()`**
- Letterbox math: scale factor, scaled content dims, padding (x, y).
- Per-stage timings: temp-bitmap resize, `getPixels` read, normalisation loop.
- Buffer capacity, pixel count.
- Per-channel post-normalisation stats `R/G/B [min, mean, max]` so you can spot black/white/clipped frames.
- `Log.w` if all three channels show < 0.05 dynamic range (probably blank source — detection won't find anything).

**At `close()`**: already covered (info on release, warn if `interpreter.close()` throws).

Filter just this component with:

```bash
adb logcat -s 'FaceMesh.Detector:V'
```

or pair it with the anchor decoder's tag for the full detection slice:

```bash
adb logcat -s 'FaceMesh.Detector:V' 'FaceMesh.Decoder.Anchor:V'
```

---

**User**

Create a commit message and commit the changes

---

**Cursor**

Committed as `387732c` on `main` (now 1 commit ahead of `origin/main`). All 10 modified files are in:

- `app/src/main/kotlin/com/alifesoftware/facemesh/ml/BlazeFaceDetector.kt`
- `app/src/main/kotlin/com/alifesoftware/facemesh/ml/BlazeFaceDecoder.kt`
- `app/src/main/kotlin/com/alifesoftware/facemesh/ml/FaceFilters.kt`
- `app/src/main/kotlin/com/alifesoftware/facemesh/ml/FaceAligner.kt`
- `app/src/main/kotlin/com/alifesoftware/facemesh/ml/FaceProcessor.kt`
- `app/src/main/kotlin/com/alifesoftware/facemesh/ml/cluster/Dbscan.kt`
- `app/src/main/kotlin/com/alifesoftware/facemesh/ml/cluster/EmbeddingMath.kt`
- `app/src/main/kotlin/com/alifesoftware/facemesh/media/BitmapDecoder.kt`
- `app/src/main/kotlin/com/alifesoftware/facemesh/domain/ClusterifyUseCase.kt`
- `app/src/main/kotlin/com/alifesoftware/facemesh/domain/FilterAgainstClustersUseCase.kt`

`+473 / −64` lines.

The leftover untracked `.kotlin/` is the Kotlin compiler daemon cache — it's already in your gitignore-style noise (and not worth committing); leaving it alone. Working tree is otherwise clean. Push whenever you're ready.

---

**User**

Our logs should be such a level of detail that we can completely analyze what happened n detail thru the entire run of the app across various operations

---

**Cursor**

I'll do a thorough sweep across the rest of the app — startup, DI, ViewModels (user intents + state transitions), navigation, repository, prefs, model store, screens — so a single logcat run can replay an entire session end-to-end. Let me first map what's actually there.

I have enough context. Time to add the logging — startup → DI → factories → ViewModel intents/transitions → flows → repo/prefs/store → nav routes → screens.

## 1. App startup + DI graph

## 2. AppDatabase build event

## 3. AppPreferences — every read/write

## 4. ClusterRepository — every operation

## 5. ModelStore — cache hits/misses

## 6. HomeViewModel — every intent + state transition + flow event

## 7. SettingsViewModel + Factories

## 8. MainActivity — lifecycle + picker outcomes

## 9. Navigation routes

## 10. HomeScreen — UiMessage + dialog visibility

## 11. KeeperGalleryScreen — entry/exit + count

## 12. SettingsScreen — toggle observation

Now lint + tests.

Done. Build green, all 33 unit-test tasks pass. The whole UI/data/control surface now traces every interesting event so a single logcat capture can be replayed end-to-end.

## What's now logged across the rest of the app

**Process lifecycle** — `FaceMesh.App`
- `onCreate` line with pid, app id, version, build type, debug flag, SDK, manufacturer/model, supported ABIs, AppContainer build time.
- `onLowMemory` / `onTrimMemory(level=…)` warnings when the system reclaims.

**DI graph** — `FaceMesh.DI`
- `init` line confirming wiring is complete and lazy.
- A line per first-access of each lazy holder (`AppDatabase`, `AppPreferences`, `ClusterRepository`, `ModelStore`, `ModelDownloadManager`, `MlPipelineProvider`) so you can see the order things actually get built.

**Database** — `FaceMesh.Db`
- Room build line with the absolute db path, file size before open, and total open time.

**Preferences** — `FaceMesh.Prefs`
- Every `set*` call logs the key + new value.
- Every observed flow (`dbscanEps`, `dbscanMinPts`, `matchThreshold`, `modelsVersion`, `lastModelsCheck`, `pendingFilterSession`, `gpuFallbackToastShown`, `dynamicColorEnabled`) emits an `observe key=value` line on each distinct change.
- `clearAll` logs `Log.w` so a Reset always stands out in the trace.

**Repository** — `FaceMesh.Repo`
- `loadClusters` reports row count, ids and time.
- `loadCentroidsForIds` reports requested vs matched count and warns about missing ids.
- `saveCluster`, `deleteCluster`, `deleteAll` log start + completion with timings.
- `observeClusters` flow emits a line per emission with row count and ids.

**ModelStore** — `FaceMesh.Store`
- `init` shows the on-disk paths and whether the manifest is already cached.
- `fileFor` logs each resolved path + on-disk presence/size.
- `readManifest` logs parse success (with full config summary) or warn on parse failure.
- `allFilesPresent` enumerates each file's exists/size with a single OK/not-OK summary.
- `clear` warns and confirms each file deletion (used by Reset).

**ViewModels** — `FaceMesh.HomeVM`, `FaceMesh.SettingsVM`
- `HomeViewModel.handle(intent=…, curState=…)` for every intent — both user taps and pipeline events.
- Centralised `transition[reason]` and `updateState[reason]` helpers log every state machine change with old → new and a short reason tag (`Clusterify/start`, `Filter/finished`, `Reset`, `CancelProcessing`, etc.). No-ops are also logged so you can see when an intent was ignored.
- `emitMessage[reason]` logs every snackbar / navigation `UiMessage`.
- All "ignored in unexpected state" branches log with `Log.w`.
- Pipeline launch lines include wall time from launch and all download progress milestones.
- `SettingsViewModel` logs creation + each toggle commit; `SettingsViewModelFactory` logs construction.

**Navigation** — `FaceMesh.Nav`
- `host:` line on `NavController` creation and a per-back-stack-entry trace as the user navigates.
- `route enter` / `route exit` for Home, Keepers (with sessionId + item count), Settings.
- Every intra-app navigation call logs the source → destination edge.

**Screens** — `FaceMesh.HomeScreen`, `FaceMesh.KeeperScreen`, `FaceMesh.SettingsScreen`
- `compose: entered/disposed` lifecycle pairs.
- `tap:` line for every FAB / button / swipe (Add, Clusterify, Filter, Pick, Clear, Reset, Settings, ToggleCluster, SwipeDelete, CancelProcessing).
- `messages: received <UiMessage>` for every snackbar/navigation message.
- `dialog: Reset confirmation visible/confirmed/dismissed`.
- `switch: dynamicColor toggled -> true/false` in Settings.

**MainActivity** — `FaceMesh.MainActivity`
- `onCreate / onStart / onResume / onPause / onStop / onDestroy` with `isFinishing` + `isChangingConfigurations`.
- Each picker launch + result (item count + cancel detection).
- `persistUriPermissions` summary line (total / persisted / skipped) plus per-uri info on rejections.

## Tag cheat-sheet

```
FaceMesh.App           process lifecycle, memory pressure
FaceMesh.DI            object graph wiring
FaceMesh.Db            Room
FaceMesh.Prefs         DataStore reads + writes
FaceMesh.Repo          ClusterRepository
FaceMesh.Store         model store filesystem
FaceMesh.ModelDl       model downloads + retries (existing)
FaceMesh.TfLite        TFLite runtime + delegate (existing)
FaceMesh.Pipeline      ML graph build/teardown (existing)
FaceMesh.Decoder       BitmapDecoder (existing)
FaceMesh.Detector      BlazeFaceDetector (existing)
FaceMesh.Decoder.Anchor BlazeFaceDecoder (existing)
FaceMesh.Filters       FaceFilters (existing)
FaceMesh.Aligner       FaceAligner (existing)
FaceMesh.Embedder      FaceEmbedder (existing)
FaceMesh.Processor     FaceProcessor (existing)
FaceMesh.EmbedMath     L2 / mean (existing)
FaceMesh.Dbscan        clustering (existing)
FaceMesh.Clusterify    Clusterify use case (existing)
FaceMesh.Filter        Filter use case (existing)
FaceMesh.MainActivity  Activity lifecycle + pickers
FaceMesh.HomeVM        Home state machine
FaceMesh.SettingsVM    Settings VM
FaceMesh.Nav           navigation host + routes
FaceMesh.HomeScreen    Home UI taps + messages + dialogs
FaceMesh.KeeperScreen  Keepers UI
FaceMesh.SettingsScreen Settings UI
```

Capture a full session with:

```bash
adb logcat -v threadtime -s 'FaceMesh.*:V'
```

That should give you a chronological replay from process start → DI lazy builds → screen entry → user taps → preference reads → model download → TFLite init → per-image pipeline (decode → detect anchors → filter decisions → align → embed → cluster decisions) → cluster persist → navigation → filter run → keepers → reset/wipe → process death.

Want me to commit this batch as well?

---

**User**

Yes commit

---

**Cursor**

Committed as `2f2f986` on `main`. You're now 2 commits ahead of `origin/main`:

- `2f2f986` Trace lifecycle, DI, ViewModel, nav, and UI for end-to-end replay (`+709 / −112` across 14 files)
- `387732c` Deepen ML pipeline logging around per-decision branches

Working tree is clean (the untracked `.kotlin/` daemon cache is leftover noise — leaving it alone). Push whenever ready.

---

**User**

Create a Cursor rule so that everything we do in future on this project, adheres to such a logging verbosity. We want to add logs wherever it matters, and our goal is to totally replay our pipeline from the logs. The rule should be run for any code we write.

---

**Cursor**

I'll create a single always-apply rule that codifies the conventions we've established. Per the skill: under 50 lines, focused, concrete examples drawn from your real code.

Let me verify my claim about R8 stripping `Log.i` — I want to make sure the rule isn't claiming something untrue.

Good — the claim is accurate. `Log.v`, `Log.d`, and `Log.i` are all stripped at R8 release builds via `assumenosideeffects`, which is exactly why we can be liberal with `Log.i` without paying for it in production. `Log.w` and `Log.e` are intentionally NOT in the strip list, so warnings/errors stay visible.

Now let me commit the rule.

I'll commit only the rule — leaving the other untracked docs files alone since they're outside the current task.

Done. Committed as `6ccf89d`.

The rule lives at `.cursor/rules/logging-verbosity.mdc` with `alwaysApply: true`, so it'll be loaded into every chat / agent run on this project automatically — no need to opt in per file.

What it codifies:

- **Goal stated up front**: a single `adb logcat -s 'FaceMesh.*:V'` capture must replay the entire run end-to-end.
- **Tag scheme**: `FaceMesh.<Component>` in a companion `private const val TAG`, one tag per class.
- **Severity matrix**: `Log.i` for happy paths/decisions/calculations (R8 strips these in release), `Log.w` for tolerated failures and ignored intents, `Log.e` for hard failures. No `Log.d` / `Log.v` / `println` / custom logger.
- **Always-log list**: lifecycle, every intent + every state transition, every flow event, every prefs/DB write, navigation edges, screen taps + dialog visibility.
- **ML / pipeline specifics**: log inputs (dims, dtype, ranges) and outputs (counts, top scores, distribution stats), per-decision `PASS`/`DROP` with the actual reason, explicit early-exit lines, per-stage timings via `SystemClock.elapsedRealtime()`, tensor shapes/dtype/byteCount at init.
- **Helper patterns**: centralised `transition(reason, next)` so state mutations are never bare `_state.value = X`.
- **Anti-patterns** (with examples): silent catches, untraced state mutation, aggregate-only logs that hide per-item discards, per-iteration logs in O(n²) loops.
- **Build hygiene**: keep `unitTests.isReturnDefaultValues = true`; reminder that R8's `assumenosideeffects` already strips the verbose levels in release.

The rule pulls its concrete examples directly from patterns we just landed (`FaceMesh.Filters` PASS/DROP lines, the `transition[reason]` helper from `HomeViewModel`, the recycled-bitmap guard from `BlazeFaceDetector`), so anyone reading it can grep the codebase for live precedent.

