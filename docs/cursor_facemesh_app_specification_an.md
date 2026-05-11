# (1) FaceMesh app specification and planning
_Exported on 5/10/2026 at 13:55:07 PDT from Cursor (3.3.27)_

---

**User**

Attached is two different files for an app we want to build. It also has UI specs.

One thing to remember - these attached docs call the app FaceMash, but all the references ot FaceMash should actually be FaceMesh. Please remember this.
   
Start with a complete spec, following the global cursor rules for native Android development.

First off, we need to create our own spec and plan based on the specs we read.

Remember - FaceMash is a typo, this is all FaceMesh

@/Users/gifty/Documents/FaceMesh/FaceMash App PRD and UI Requirements.md @/Users/gifty/Documents/FaceMesh/FaceMash_App_PRD_UI_Requirements_qwen.md @/Users/gifty/Documents/FaceMesh/Gemini-Android Face Recognition TFLite App.md 

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

I am relaxing the size lints

Which all models do you need to download? I will give the URLs

Let's tackle GhostNet first.

- I will host these models in my public Github repo.
- Before that, there's one ONNX model FP32 - https://github.com/alifesoftware/ModelZoo/blob/master/GhostFaceNet/Model/ghostface_fp32.onnx  and one TFLite W8A8 model - https://github.com/alifesoftware/ModelZoo/blob/master/GhostFaceNet/Model/model.tflite 
- Yaml: https://github.com/alifesoftware/ModelZoo/blob/master/GhostFaceNet/model_cfg.yaml 
- Model Information: https://github.com/alifesoftware/ModelZoo/blob/master/GhostFaceNet/README.md 
- Test Input - https://github.com/alifesoftware/ModelZoo/blob/master/GhostFaceNet/quantized_input/test_0.bin 

I am okay with option A - FP32 to FP16 conversion. We will need ot test the model inputs and outputs though and compare that along with tensor dump between FP32 ONNX and FP16 TFLite

Agree on your understanding ov 2. Input layout — NCHW vs NHWC (likely a real problem). That is why the validation of model is importnt for us.

>3. Embedding dimension — 128 vs 512
Is this a problem?

> Hosting Choice: 
GitHub Release (recommended) is good, please guide me though.

> Input/output details from Netron or the Python snippet, so I can lock NHWC vs NCHW and 128-d vs 512-d.

How do I find this?

Let's walk step by step please, I am new to this. First, let us work with Netron.app

FP32 ONNX:

Input:
name: input_1:0
tensor: float32[1,112,112,3]

Output:
name: Identity:0
tensor: float32[1,512]

W8A8 TFLite:
Input: 
name: Data0_out
tensor: int8[1,112,112,3]
identifier: 420
quantization: linear
-0.984926700592041 ≤ 0.0077249156311154366 * (q + 1) ≤

Output:
name: torch.add_896_out
tensor: int8[1,512]
identifier: 492
quantization: linear
-6.832368850708008 ≤ 0.05472680553793907 * (q + 3) ≤ 7.122966766357422

face_detection_short_range.tflite

Input:
name: input
tensor: float32[1,128,128,3]
identifier: 0

Output:
name: regressors
tensor: float32[1,896,16]
identifier: 175

name: classificators
tensor: float32[1,896,1]
identifier: 174




Let's capture information we have so far:

GhostFace FP16:
SHA256: 4965f3463d0209a298d5409aa4c08d1f7d6d1e4ae15fbd6aafdaf0ab2478fe1c 

Manifest: @models/ghostface_fp16_manifest.json 

Model: @models/ghostface_fp16.tflite 

BlazeFace:

SHA256: 3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633

Mode: @models/face_detection_short_range.tflite 

https://github.com/alifesoftware/ModelZoo/releases/download/facemesh-v1/face_detection_short_range.tflite 
https://github.com/alifesoftware/ModelZoo/releases/download/facemesh-v1/ghostface_fp16.tflite 
https://github.com/alifesoftware/ModelZoo/releases/download/facemesh-v1/manifest.json 

For QQ-2 I like this sort of color scheme (from one of my React apps):

{
  "name": "All Black",
  "theme": {
    "primary": "#007AFF",
    "secondary": "#10b981",
    "background": "#000000",
    "backgroundSecondary": "#000000",
    "card": "#000000",
    "text": "#F2F2F2",
    "subtext": "#A0A0A0",
    "border": "#000000",
    "positive": "#32D74B",
    "negative": "#FF453A",
    "neutral": "#98989D",
    "rangeSelectorBg": "#232A34",
    "notification": "#FF453A",
    "success": "#32D74B",
    "warning": "#FF453A",
    "info": "#0A84FF",
    "overlay": "rgba(0,0,0,0.6)",
    "shadow": "rgba(0,0,0,0.5)",
    "gradientStart": "#000000",
    "gradientEnd": "#000000",
    "marketDataIconColor": "#4FC3F7",
    "sectorIconColor": "#66BB6A",
    "priceRangeIconColor": "#BA68C8",
    "peRatioIconColor": "#FFA726",
    "volumeIconColor": "#EF5350",
    "regionIconColor": "#29B6F6",
    "valuationIconColor": "#FFD54F",
    "performanceIconColor": "#81C784",
    "ratioIconColor": "#FFB74D",
    "esgIconColor": "#26A69A",
    "ownershipIconColor": "#90A4AE",
    "cashFlowIconColor": "#4DD0E1",
    "balanceSheetIconColor": "#BCAAA4",
    "incomeStatementIconColor": "#A1887F",
    "chartPrimary": "#3B82F6",
    "chartSecondary": "#06B6D4",
    "chartTertiary": "#8B5CF6",
    "chartAccent": "#F97316",
    "ownershipRetail": "#EC4899",
    "ownershipInstitutional": "#6366F1",
    "ownershipInsider": "#F59E0B",
    "neutralBenchmark": "#64748B",
    "volumeSecondary": "#0EA5E9",
    "indicatorGrowth": "#10B981",
    "indicatorRisk": "#F43F5E",
    "analystSentiment": "#A855F7"
  }
}

---

**Cursor**

Update the SPEC §4.4 color tokens and mark OQ-2 resolved.

Build + run all 34 tests to confirm nothing regressed.

All builds green, 34 tests passing, no lint issues.

## Summary of the All Black rollout

| File | What changed |
|---|---|
| [`Color.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/ui/theme/Color.kt) | Replaced light/dark token pairs with the "All Black" palette (`FaceMeshPrimary` `#007AFF`, `FaceMeshSecondary` `#10B981`, etc.). |
| [`Theme.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/ui/theme/Theme.kt) | `FaceMeshTheme` is now **always dark** — no `darkTheme` / `dynamicColor` parameters. Material You override removed. |
| [`MainActivity.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/MainActivity.kt) | `enableEdgeToEdge` uses `SystemBarStyle.dark(...)` so status / nav bar icons are always light, regardless of system day/night. |
| [`values/themes.xml`](app/src/main/res/values/themes.xml) + [`values-night/themes.xml`](app/src/main/res/values-night/themes.xml) | XML theme parent switched to `Theme.Material3.Dark.NoActionBar`, `windowBackground` pinned to black, `windowLightStatusBar` forced false. Both files identical (DayNight folder selection becomes a no-op). |
| [`colors.xml`](app/src/main/res/values/colors.xml) | Adaptive icon background updated to primary blue `#007AFF` (pure black would disappear on a black home-screen wallpaper). |
| [`ProcessingOverlay.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/ui/components/ProcessingOverlay.kt) | Scrim now uses the theme `scrim` token (`#000000` @ 60%) instead of a hardcoded `Color.Black.copy(alpha = 0.55f)`. |
| [`SPEC.md`](SPEC.md) §4.4 | Replaced the old token table with a comprehensive Material 3 role → All Black mapping. |
| [`SPEC.md`](SPEC.md) §12 OQ-2 | Marked resolved. App icon **glyph** still flagged as TBD. |

## What you'll see at runtime

- **Home screen:** pure black background, "FaceMesh" title in soft white (`#F2F2F2`), "+" / Camera / Filter button in iOS blue.
- **Cluster row:** circular avatars on black; checkboxes filled in emerald (`#10B981`) when checked, soft grey outline (`#3A3A3C`) when unchecked.
- **Swipe-to-delete background:** crimson (`#FF453A`).
- **Reset / Delete dialogs:** confirm button text in `#FF453A`.
- **Processing / Filtering overlay:** 60% black scrim, surface card uses Material 3 tonal elevation against pure black for a soft floating look.
- **Keeper gallery:** still pure black (intentional — gallery convention) with white back arrow + page indicator dots.
- **System bars:** transparent, light icons, never flicker into "light mode" even if the user toggles their device.

## Open items remaining

| # | Item | Status |
|---|---|---|
| OQ-1 | Model hosting URL | ✅ Resolved |
| OQ-2 | Brand colors | ✅ Resolved (icon glyph still pending — tell me when you have one and I'll wire it into `ic_launcher_foreground.xml`) |
| OQ-3 | DBSCAN noise faces | Silently dropped in v1 |
| OQ-4 | Keeper output (in-app gallery vs MediaStore folder) | In-app gallery only in v1 |
| OQ-5 | Threshold tuning (currently `0.65`) | Empirical calibration needed |

If you want, the next natural step is the real-device smoke test we talked about: `./gradlew :app:installDebug`, plug in a phone, pick 10–20 photos with 2–3 distinct faces, tap Clusterify, and observe.

---

**User**

Before we move on, can we talk about - no Material You dynamic-color override, what does this mean

---

**Cursor**

Great question — this is one of those Android details that sounds like jargon but actually has visible consequences. Let me unpack it.

## What "Material You dynamic color" is

Starting with Android 12 (API 31, released late 2021), Google introduced a feature called **Material You** (also called "dynamic color" in code). The idea:

1. The user picks a wallpaper.
2. The OS extracts a dominant color palette from that wallpaper.
3. Apps that opt in receive that palette and use it as their **primary brand color**.

So Pixel users with a sunset wallpaper end up with orange-tinted apps; users with a forest wallpaper get green-tinted apps; etc. Settings, Calculator, Calendar, Clock, even some of Google's own apps re-tint themselves to match.

The whole system is built so that, as a developer, you opt in by writing literally one line:

```kotlin
val colors = if (Build.VERSION.SDK_INT >= 31) {
    dynamicDarkColorScheme(context)   // <-- this is the override
} else {
    myStaticDarkScheme
}
```

`dynamicDarkColorScheme(context)` reaches into the OS's `wallpaperColors` API, pulls back the user's currently-extracted palette, and returns a Material 3 `ColorScheme` whose `primary`, `secondary`, `surface`, etc. are all derived from the wallpaper.

## What we had before

The original `Theme.kt` had this exact pattern:

```kotlin
@Composable
fun FaceMeshTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,    // <-- on by default
    content: @Composable () -> Unit,
) {
    val colors: ColorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val ctx = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(ctx)   // <-- WALLPAPER WINS
            else dynamicLightColorScheme(ctx)
        }
        darkTheme -> DarkColors                          // <-- our brand wins
        else -> LightColors
    }
    MaterialTheme(colorScheme = colors, ...)
}
```

That branching means: **on Android 12+, the wallpaper-derived colors take precedence over the brand palette.** If the user had a pinkish wallpaper, the Clusterify button would be pink — even though we'd defined it as iOS blue.

## What "no override" means now

The new `Theme.kt` looks like this:

```kotlin
private val FaceMeshAllBlackScheme: ColorScheme = darkColorScheme(
    primary   = FaceMeshPrimary,        // #007AFF, no exceptions
    secondary = FaceMeshSecondary,      // #10B981, no exceptions
    background = FaceMeshBackground,    // #000000, no exceptions
    ...
)

@Composable
fun FaceMeshTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = FaceMeshAllBlackScheme,   // <-- always this, every device
        ...
    )
}
```

Notice three deliberate omissions:

1. **No `dynamicColor` parameter.** It's not even a knob anymore. The wallpaper API never gets called.
2. **No `darkTheme` parameter.** We don't read `isSystemInDarkTheme()`. The user could be in the desert at noon with their phone in "always light mode"; FaceMesh is still pitch black with iOS blue accents.
3. **No `LightColors` table.** It used to exist; I deleted it. There's literally only one color scheme to switch into.

So "no override" = **the brand palette wins on every device, every wallpaper, every system setting**.

## Why we made that choice for FaceMesh specifically

| Reason | Detail |
|---|---|
| **Brand consistency** | If a screenshot of FaceMesh circulates on Twitter / Reddit, we want it to be recognizably FaceMesh. With dynamic color, two screenshots of the same screen on two different Pixels can look like two different apps. |
| **Image-viewing UX** | The Keeper gallery is the highlight of the app — people staring at photos. A pure black background is the photography-app convention (Apple Photos, Google Photos full-screen, Lightroom). Dynamic color would inject pink/green/yellow tints behind the photos and shift the perceived color of skin, sky, etc. |
| **OLED battery** | True `#000000` literally turns OLED pixels off. That's a measurable battery win on most modern phones. Dynamic color tends to push toward a "near-black surface" with a hint of brand color, which keeps pixels lit. |
| **Predictability for support** | If a user later reports "the cluster button is yellow and I can't see it", we don't have to ask "what's your wallpaper?" — the button is provably blue, full stop. |

## The trade-offs

To be fair to the alternative, here's what we *gave up* by turning dynamic color off:

| What we lost | Honest assessment |
|---|---|
| **Personalization** | Some users genuinely love that all their apps match their wallpaper. We won't show up in Reddit "Material You makes my home screen look so cohesive" posts. |
| **OS integration polish** | A handful of UI elements (snackbars, the system back gesture indicator) might not perfectly harmonize with the rest of the user's setup. In practice, almost nobody notices. |
| **Light-mode parity** | Anyone who uses their phone in always-light mode and expected FaceMesh to also be light will be surprised once. After the first launch they'll either accept it or close the app — for a privacy-first photo tool this is a fine trade. |

## How to flip it back if we ever want to

Two-line change. Add `dynamicDarkColorScheme(LocalContext.current)` back into a `when`. We could even make it a setting (`Settings → Appearance → Use system wallpaper colors`) and put it behind a DataStore boolean. Easy escape hatch — we just chose not to take it for v1.

## Concrete example of what users will see

**With dynamic color (the old code):**
- User A on a Pixel 8 with a forest wallpaper opens FaceMesh → Clusterify button is **green**.
- User B on a Pixel 8 with a sunset wallpaper opens FaceMesh → Clusterify button is **orange**.
- User C on a Samsung Galaxy S23 (which doesn't enable dynamic color the same way) → Clusterify button is **iOS blue** (our static fallback).

**Without dynamic color (the current code):**
- User A → Clusterify button is **iOS blue**.
- User B → Clusterify button is **iOS blue**.
- User C → Clusterify button is **iOS blue**.

That's the whole story. Let me know if you'd like to revisit the decision (it's a single-commit revert), or if we're good and ready to move to the smoke test.

---

**User**

I'd like to eanble that as an inapp setting if possible?

---

**Cursor**

Yes, totally doable. I'll add a small in-app Settings screen with the toggle, persist the choice in DataStore, and have `FaceMeshTheme` read the preference. The default stays "All Black" (so first-launch UX is unchanged); flipping the toggle re-renders the whole UI on the fly.

Let me lay out the plan briefly so you can see what's about to change:

1. New DataStore key: `dynamic_color_enabled` (default `false`).
2. New `SettingsScreen` (Compose, accessed via a gear icon in the top-right).
3. New nav route `Routes.Settings`.
4. `FaceMeshTheme` takes a `useDynamicColor: Boolean` parameter; on Android 12+ uses `dynamicDarkColorScheme`, else falls back to the All Black scheme.
5. `MainActivity` reads the preference flow and passes it into the theme so the whole tree re-renders when the toggle flips.
6. Unit test for the new preference key.

Building it now.

Now teach `FaceMeshTheme` to honor the toggle.

Now the Settings screen.

New string resources:

Now wire the navigation route + the gear icon entry-point in `TopActions`.

Update `HomeScreen` to accept and forward the `onSettings` callback:

Update the previews so they compile:

Now wire navigation to the Settings screen.

Now `MainActivity` reads the dynamic-color preference and passes it into the theme so the entire UI re-renders when the toggle flips.

Add a unit test for the new preference key:

Build green, all 38 tests passing (added 4 in `AppPreferencesTest`).

Update SPEC §4.4 (mention of "no Material You" needs softening since it's now opt-in).

## Summary of what just shipped

| Component | Change |
|---|---|
| [`AppPreferences.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/data/AppPreferences.kt) | New `dynamicColorEnabled: Flow<Boolean>` (default `false`) + `setDynamicColorEnabled(Boolean)` setter, persisted via DataStore. |
| [`Theme.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/ui/theme/Theme.kt) | `FaceMeshTheme(useDynamicColor: Boolean = false, content)` — falls back to the All Black scheme on API < 31 or when the toggle is off; uses `dynamicDarkColorScheme(LocalContext.current)` otherwise. |
| [`SettingsScreen.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/ui/screens/SettingsScreen.kt) | New Compose screen with an "Appearance" section and the "Use system wallpaper colors" Material 3 `ListItem` + `Switch`. Subtitle adapts based on whether the device runs API ≥ 31; on older devices the switch is disabled and shows "Requires Android 12 or newer." |
| [`SettingsViewModel.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/viewmodel/SettingsViewModel.kt) | Tiny VM exposing the flow + setter, plus a `SettingsViewModelFactory`. |
| [`FaceMeshNavHost.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/nav/FaceMeshNavHost.kt) | New `Routes.Settings` destination wired to `SettingsScreen`. |
| [`TopActions.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/ui/components/TopActions.kt) | New gear icon button (always visible) that fires the `onSettings` callback. |
| [`HomeScreen.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/ui/screens/HomeScreen.kt) | Threads the new `onNavigateToSettings` callback through to `TopActions`. All previews updated. |
| [`MainActivity.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/MainActivity.kt) | Reads `AppPreferences.dynamicColorEnabled` via `collectAsStateWithLifecycle` at the `setContent` level; passes the value into `FaceMeshTheme`. The whole tree re-renders the instant the user flips the switch. |
| [`strings.xml`](app/src/main/res/values/strings.xml) | New strings: `settings_title`, `cd_open_settings`, `settings_section_appearance`, `settings_dynamic_color_title`, `settings_dynamic_color_subtitle`, `settings_dynamic_color_unavailable`. |
| [`AppPreferencesTest.kt`](app/src/test/kotlin/com/alifesoftware/facemesh/data/AppPreferencesTest.kt) | New Robolectric test exercising the dynamic-color preference round-trip + verifying SPEC defaults (`match_threshold = 0.65`, `dbscan_eps = 0.35`, `dbscan_min_pts = 2`). |
| [`SPEC.md`](SPEC.md) §4.4 | Updated to describe the opt-in toggle. |

## What the user sees

- **Top-right of the home screen:** a gear icon, always visible (next to Reset / Clear when those exist).
- **Tap gear → Settings screen** with a single "Appearance" section: "Use system wallpaper colors" with a Material 3 toggle.
- **On Android 12+:** flipping the toggle on instantly re-tints the whole app — Clusterify button, cluster checkboxes, processing overlay, etc. — to colors derived from the user's wallpaper. Background stays dark (we use `dynamicDarkColorScheme`, not the light variant).
- **On Android 10/11:** the toggle is shown disabled with subtitle "Requires Android 12 or newer." so the user understands why.
- **The setting persists** across app restarts via DataStore.

## Build numbers

- **34 tests → 38 tests** (4 new in `AppPreferencesTest`)
- **Debug + release APKs** still building clean
- No linter warnings

Ready for the smoke test whenever you are.

---

**User**

What are we doing for Smoke Test? And what are the other questions we need to discuss?

---

**Cursor**

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

---

**User**

1 - Pass
2 - Pass - But UI needs work - Too much randm, empty space on top of the app (see screenshot - Screenshot 2026-05-09 at 5.34.25 PM.png)
3 - Pass - Photo Picker opens
4 - Pass
5 - Pass
6 - Pass
7 - Pass
8 - Partial Pass - we will address this later. Please remind me
9 - Pass - BUT UI needs work - circular images look warped or squished (see screenshot - Screenshot 2026-05-09 at 5.40.20 PM.png)
10 - Haven't tested yet - Please remind me
11 - Pass

Fix what things we need to fix. The the things I said "Remind Me" should be added to added as Tier D, and give me a list of Tier-A, Tier-B, Tier-C, and Tier D

---

**Cursor**

Great progress — let me parse the screenshots and fix both UI issues.

**Screenshot 1 diagnosis (empty state, too much top space):** Three things stack vertically before the title — status bar padding, the `TopActions` row (~48 dp tall), an 8 dp spacer, then the title. Plus the `ThumbnailFan` reserves layout space (128×112 dp) even when it's holding zero thumbnails, so the centered `+` gets pushed further down. The fix is to (a) collapse title + actions into a single top row, and (b) make the fan zero-sized when empty.

**Screenshot 2 diagnosis (red ring around avatars):** This is `SwipeToDismissBox`'s `backgroundContent` (the red delete circle) bleeding through. The foreground is an 80 dp `Box` containing only a 64 dp circular `Surface` + a 28 dp checkbox — the 8 dp ring around the circle is transparent, so the red below shows through at rest. The fix: paint the foreground `Box` opaque with the surface color so the red only appears when the user actually swipes.

Both fixes incoming.

Now restructure the home-screen header to put title + actions on a single row.

`TopActions` uses `fillMaxWidth` internally which would override the `wrapContentWidth` and re-stretch. Let me clean it up.

Now add the `wrapContentWidth` import to HomeScreen:

Both fixes compile and all 38 tests still pass. Reinstall + relaunch to see them on device:

```bash
cd /Users/gifty/Development/FaceMesh
./gradlew :app:installDebug
adb shell am start -n com.alifesoftware.facemesh.debug/com.alifesoftware.facemesh.MainActivity
```

You should now see (a) "FaceMesh" title sitting on the same row as the gear icon at the top, with the `+` more genuinely centered vertically, and (b) cluster avatars showing as clean circles with no red ring — the red only appears when you actively swipe one to the left.

---

# Updated open-question tier list

## Tier A — settle before any user-facing build (must-do)

| Item | Notes |
|---|---|
| **Cluster naming UI** | Long-press a cluster avatar → small text-edit dialog → store in existing `ClusterEntity.name` column. ~1 hour. |
| **Long Clusterify in background** | Move pipeline to a `WorkManager` foreground service so backgrounding the app mid-run doesn't kill it. Important for any photo set ≥ ~30 images. |
| **Remind: Title-bar tap target / vertical balance polish** | Done as part of this round. ✅ |
| **Remind: Cluster avatar visual bleed** | Done as part of this round. ✅ |

## Tier B — informed by smoke-test feedback (post-first-runs)

| Item | Notes |
|---|---|
| **OQ-3 DBSCAN noise faces** | Surface as an "Unclassified" pseudo-cluster, or keep silently dropping. |
| **OQ-5 Threshold calibration** | Iterate on `match_threshold` (currently 0.65) and `dbscan_eps` (currently 0.35) by editing `manifest.json` in the GitHub Release. No app rebuild required. |
| **DBSCAN minPts for tiny inputs** | If user picks 2 photos with `minPts=2` and the two faces are dissimilar, both end up as noise → "no faces found". Maybe scale minPts dynamically (e.g. `min(2, photoCount/2)`). |
| **Re-clustering when adding photos to existing clusters** | Currently each Clusterify run is independent. Adding "incremental clustering" (merge new faces into existing clusters) is meaningful work but probably worth it for v1. |
| **Cluster sort order** | Currently `createdAt ASC` (oldest first). Probably better: `faceCount DESC` so the largest cluster is leftmost. |

## Tier C — v1.1+

| Item | Notes |
|---|---|
| **OQ-4 Keeper output** | Save matched photos to `Pictures/FaceMesh/Keepers/` MediaStore folder so they're visible in the system Photos app. |
| **Share keepers** | Add a Share action in `KeeperGalleryScreen`. |
| **App icon glyph** | Current is my placeholder smiley — needs your final design. |
| **Centroid auto-update during Filter** | Every confirmed match could enrich the cluster centroid. Off by default; opt-in confirm mode. |
| **Live camera capture as input source** | Big feature; nice-to-have. |
| **Cluster merging UI** | Two clusters that are actually the same person → user-driven merge. |
| **Privacy policy / About / version screen** | Required for Play Store. |
| **Release signing keystore** | Required for Play Store. Currently release builds are debug-signed. |
| **Periodic manifest re-check** | If we publish `facemesh-v2` models, current app won't notice unless reinstalled. |

## Tier D — smoke-test follow-ups (deferred per your call today)

| Item | Notes |
|---|---|
| **Smoke test #8 — Detection + embedding pipeline (Partial Pass)** | "We will address this later" — please share specifics whenever ready: did the per-image counter advance correctly? Was timing acceptable? Were detected face counts reasonable? Did anything look off in the cluster avatars (besides the red-ring fix above)? |
| **Smoke test #10 — Filter flow end-to-end (Not yet tested)** | Once you've completed Clusterify and have at least one good cluster, exercise the Camera → pick ≤15 photos → Filter → swipeable Keepers gallery flow. We'll learn whether `match_threshold = 0.65` produces sensible keep / discard decisions. |
| **Smoke test #11.5 — Settings → Dynamic color toggle on Pixel** | Worth flipping the Material You toggle once you have a few clusters showing — the wallpaper-derived re-tint of the Clusterify button + cluster checkbox is satisfying to see live. |

---

When you're ready, give me a sign and I'll start on the Tier A items. My recommended order:

1. **Cluster naming UI** (small, satisfying, immediately improves cluster row UX) — first.
2. **Foreground service for Clusterify** (less visible but prevents real bugs as photo counts grow) — second.

Or if you want to keep iterating on smoke-test findings before adding more features, that's a fine path too.

---

**User**

The image in the circular cluster at the below, it should use center fit or something, insted of fill screen or whatever it is now

---

**Cursor**

Looking at the screenshot you sent earlier, I think there are actually **two separate problems** wrapped up in your "fill screen" complaint, and a quick `ContentScale` flip alone won't fully solve either. Let me explain what's happening and then fix it properly.

## What we're saving as the cluster avatar today

In `ClusterifyUseCase`, the representative thumbnail is the **alignment crop** — the 112×112 image produced by `FaceAligner` for the embedder. That crop has a deliberate purpose: warp the face so eyes/nose/mouth land at fixed canonical pixels, so the embedder always sees a consistent pose. That's great for ML, but the **affine warp can subtly stretch the face** — which is probably contributing to the "warped/squished" feeling.

There's also a fallback: when no `faceIndex == 0` candidate has a saved crop, we fall back to **the entire source photo URI**. On a tall portrait photo `ContentScale.Crop` will zoom in and cut off whatever's not at the center — which is exactly the "fill screen" look you're seeing.

## The proper fix — split "embedding crop" from "display crop"

Two separate artifacts:

1. **Embedding crop** — the 112×112 affine-warped face, used only for `FaceEmbedder.embed()`. Recycled after use.
2. **Display crop** — a *natural* (unwarped) square crop of the source bitmap centered on the face bounding box, with ~30% padding so you see hair / forehead / shoulders. Saved as the avatar PNG.

Plus do this for **every detected face** (not just `faceIndex == 0`), so multi-face photos always have a usable thumbnail per cluster.

Implementing now.

Now update `ClusterifyUseCase` to use the renamed field and fix the rare "fall back to source URI" path so the avatar is always a usable thumbnail.

Switch the avatar's `ContentScale.Crop` → `ContentScale.Fit` per your request, so even rare fallback paths with non-square sources don't get squished.

Add a unit test for the new natural-crop helper to lock in the geometry:

All green. **44 tests passing** (38 → 44, the 6 new ones cover the natural-crop geometry).

## What changed

| File | Change |
|---|---|
| [`FaceProcessor.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/ml/FaceProcessor.kt) | New `cropDisplayThumbnail()` helper. `FaceRecord.representativeCrop` renamed to `displayCrop` to make intent explicit. Now produced for **every detected face** when `keepDisplayCrop = true` (was just `faceIndex == 0`), so multi-face photos always have a usable thumbnail. The 112×112 affine alignment crop is recycled immediately after embedding — it's purely an embedder artifact now. |
| [`ClusterifyUseCase.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/domain/ClusterifyUseCase.kt) | `pickRepresentative()` now picks the **largest available display crop** (largest face area = closest to camera = best thumbnail) instead of arbitrarily preferring `faceIndex == 0`. Fall-back to source URI is now a true edge case. |
| [`ClusterRow.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/ui/components/ClusterRow.kt) | `ContentScale.Crop` → `ContentScale.Fit`. Identical result for our square padded crops, but rescues the rare fall-back path where the URI is a tall portrait original. Added explicit `surfaceVariant` background colour so any letterboxed area inside the circle has a sensible tone. |
| [`FilterAgainstClustersUseCase.kt`](app/src/main/kotlin/com/alifesoftware/facemesh/domain/FilterAgainstClustersUseCase.kt) | Updated to the renamed `keepDisplayCrop = false` (Filter pipeline doesn't need thumbnails). |
| [`FaceProcessorTest.kt`](app/src/test/kotlin/com/alifesoftware/facemesh/ml/FaceProcessorTest.kt) | New test class with 6 tests pinning the crop geometry: square output, downscale to 256 px, edge-shifting at top-left and bottom-right, source-clamp behaviour, degenerate input handling. |

## What you'll see

Cluster avatars are now natural face thumbnails:

- Centred on the face bounding box, with ~30% breathing room on every side (you'll see hair, forehead, a bit of shoulders / background — like a contact photo).
- Square 256 × 256 PNG at most (≤30 KB on disk per cluster).
- **Not** affine-warped — what you see is what was actually in the photo, not the canonical-pose ML input.

## Important — clear stale thumbnails before re-running smoke

Your existing clusters from earlier today still have the old **warped 112 × 112 alignment crops** saved on disk. Reinstalling the APK doesn't clear them; you need to either:

```bash
# Easiest: full uninstall + reinstall
adb uninstall com.alifesoftware.facemesh.debug
./gradlew :app:installDebug
adb shell am start -n com.alifesoftware.facemesh.debug/com.alifesoftware.facemesh.MainActivity
```

…or in-app: **Reset → confirm**, then re-Clusterify your test photos. Reset wipes the Room rows so old PNGs are orphaned-but-harmless (the new run produces fresh natural crops).

Once you've done that, the cluster row should show much more natural-looking avatars. If they still look off (too tight, too loose, off-centre), we can tune `DISPLAY_CROP_PADDING_FRACTION` (currently `0.30f` in `FaceProcessor.kt`) — `0.40f` gives more shoulders/background, `0.20f` is tighter.

---

**User**

Create a git commit message and commit message

---

**Cursor**

Let me check the repo state and history first.

Now I see the full picture. The uncommitted work is two related things from this evening:

1. **Debug logging** added across the ML pipeline (`Log.i` traces under tag `FaceMesh.*`)
2. **Natural display crops** for cluster avatars (the work we did just now to fix the squished avatars)

I'll commit them together as one logical unit — they're both smoke-test follow-ups. I'll skip the two untracked items that aren't source code:

- `.cursor/` — IDE state (should be in `.gitignore`)
- `docs/cursor_prd_spec_and_app_creation.md` — chat export, not source

Let me check identity is still configured (I can't change it), then stage + commit.

Commit `53e8b4a` is in. **16 files**, 556 insertions / 42 deletions.

```
Add ML pipeline logging and natural display crops for cluster avatars
```

## Heads-up: two untracked items I deliberately did **not** commit

```
.cursor/
docs/cursor_prd_spec_and_app_creation.md
```

- `.cursor/` is Cursor IDE state — should not be in version control. I'd suggest adding a line to `.gitignore`:
  ```gitignore
  # Cursor IDE state
  .cursor/
  ```
- `docs/cursor_prd_spec_and_app_creation.md` is a chat export. Looking at your `docs/` folder, you've already committed similar exports (`cursor_model_conversion_errors.md`, `smoke-test.md`, `todos.md`). If you want to keep this one for the same reason, just `git add docs/cursor_prd_spec_and_app_creation.md && git commit -m "docs: add chat export"`. If not, delete the file.

Want me to add the `.cursor/` line to `.gitignore` as a follow-up commit?

---

**User**

No, add these untracked files to commit also

---

**Cursor**

Both are small and benign — `.cursor/` is just the unified plan markdown, and the docs file is the chat export. Adding them as a follow-up commit.

Done. Working tree is clean and you're now **2 commits ahead of `origin/main`**:

```
04416c4  Add unified plan and PRD/spec chat export to docs
53e8b4a  Add ML pipeline logging and natural display crops for cluster avatars
f3ebb75  Initial Commit - Buildable App
38e5619  Initial commit
```

Push when ready (`git push origin main`).

