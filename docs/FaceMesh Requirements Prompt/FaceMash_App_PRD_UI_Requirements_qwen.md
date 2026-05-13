# FaceMash App PRD/UI Requirements

---

### 我

I want to a build a native Kotlin Android app, using latest Android SDK version, API version, build tools etc. I will give some requirements, functional and technical. What I want you to do is create a detailed PRD or BRD. Feel free to ask any question.

App Name: FaceMash
Package Name: com.alifesoftware.facemash

1. App starts with a home screen - Face Mash.
2. If the user does not have any saved faces, which is called as "Auto Capture List", then there is a cool looking "+" (add) icon on the center.
3. When user clicks on the "+" icon, it allows them to pick photos from their phone's photo gallery. Users can select photos multiple times, and from multiple folders - although to keep things easy, they can only pick photos from one folder or album at one time. They will need to click on + again if they want to pick photos from a different folder.
4. Screen will show a message "N photos selected" when user selected any photos, and we can show an overlapping tile format below + icon to show thumbnails of the last 3-4 photos as a good UX. Think of it as three playing cards in aa fan or accordion style
5. There will be a button at bottom called "Clusterify"
6. Clicking on the Clusterify button, will run all images thru a TFLite model, that runs using TFLite GPU Play Services library. (More on this later in technical details)
7. What this does is, created "clusters" or "groups" of photos of the same person. So if we detected 5 unique faces, we expected 5 "clusters", each cluster will have pictures of the same person. Does that makes sense? We will also need to save each "cluster" in app storage - in technical requirements, we will see what this means.
8. Visual representation of a cluster is a round image view of one picture of the person that the cluster belongs to. It is similar to how many apps show Contacts or faces. Once clusters are created, there will will be a horizontally scrollable round image views (clusters) at the bottom of the screen, and the last image view would be a + icon in a round image view. The center + icon will not be gone, and instead a new icon with Camera icon will be there in the center, let's call it "Camera" icon
9. There would be a selectable check box next to each "cluster" at the bottom, and users can select/unselect clusters
10. Camera icon will be grayed out/disabled when no cluster is checked
11. When 1 or more clusters are selected/checked, camera icon will be enabled
12. Clicking on camera button will allow the user to pick images from photo gallery, users can select upto 15 images this time, no more than that.
13. When user selected these images, the Camera icon will change to a Filter icon. And there will be a clear button on top right. Clear button will clear the photos selected (not the clusters, but selected photos), change filter icon to camera icon, and gray out camera icon again. There will also be a Reset button on top right. Reset button will clear all app storage including all the clusters and the app will go to is initial state (blank state).
14. When user selects the images, and taps on Filter icon, the selected images will be compared to each "cluster". If any of the face in selected images match any of the cluster, that image is a "Keeper", otherwise that image is filtered out. 
15. When all images are processed, the images that matched, i.e. our "keepers" are returns, and showed in an image view, which swipable.
Let's start with these.

I want you to do 2 things:
1. Come up with a BRD/PRD/Spec of the app
2. Come up with the user interface requirements.

Technical details are attached.

### Qwen

# 📘 FaceMash - Product Requirements Document (PRD) & Business Requirements Document (BRD)
**Package:** `com.alifesoftware.facemash`  
**Version:** 1.0  
**Target Platform:** Native Android (Kotlin)  
**Target SDK:** API 35 (Android 15) | **Min SDK:** API 26 (Android 8.0)  
**Build Stack:** AGP 8.4+, Kotlin 2.0+, Jetpack Compose (Recommended), Coroutines, Flow, ViewModel, Room

---

## 🔍 1. Executive Summary & Objectives
**FaceMash** is a privacy-first, on-device Android application that automates face clustering and photo filtering. Users can upload a batch of photos to automatically group unique faces into "Auto Capture Clusters." They can then select a second batch of photos (up to 15) and filter them against chosen clusters, retaining only images that contain matching faces ("Keepers"). All AI inference runs locally using TensorFlow Lite with the GPU Play Services delegate, ensuring zero cloud dependency for face data and strict adherence to a `+500KB` APK growth limit.

**Core Objectives:**
- ✅ 100% on-device face detection, embedding, clustering, and matching
- ✅ Sub-1.5s latency per face inference
- ✅ APK size growth capped at `500KB` (heavy models downloaded as sidecars)
- ✅ Intuitive, state-driven UI for photo selection, clustering, and filtering
- ✅ Lightweight, pure-Kotlin math pipeline for DBSCAN & Cosine Similarity

---

## 🧩 2. Functional Requirements

| ID | Feature | Description | Acceptance Criteria |
|----|---------|-------------|---------------------|
| FR-1 | **Initial State & "+" Icon** | Home screen shows centered `+` when no clusters exist. | `+` is visible only when cluster count = 0. Tapping opens system photo picker. |
| FR-2 | **Single-Album Selection** | Users pick photos one album/folder at a time. | System picker restricts to single album. User must tap `+` again to add from another folder. |
| FR-3 | **Selection Feedback & Thumbnails** | UI shows `N photos selected` + accordion/fan thumbnails of last 3-4 picks. | Count updates instantly. Thumbnails stack with slight rotation/offset overlap. |
| FR-4 | **Clusterify Action** | Bottom `Clusterify` button triggers on-device face clustering pipeline. | Button shows loading state. Pipeline runs asynchronously. Handles 0-face or error states gracefully. |
| FR-5 | **Cluster Persistence** | Generated clusters (centroids + representative image) are saved locally. | Clusters survive app restarts. Stored in Room DB with serialized embeddings. |
| FR-6 | **Cluster Display & Selection** | Horizontal scrollable round avatars + trailing `+`. Checkboxes beside each. | Round avatars show representative face. Checkboxes toggle selection. `Camera` icon replaces center `+`. |
| FR-7 | **Camera State Logic** | `Camera` icon is disabled if 0 clusters selected. Enabled if ≥1 selected. | Real-time UI state syncs with checkbox selection. Haptic feedback on toggle. |
| FR-8 | **Filter Photo Selection** | Tapping `Camera` opens picker for up to 15 images. | Enforces 15-image hard limit. UI shows selected count. |
| FR-9 | **Clear & Reset Controls** | Top-right: `Clear` (resets photos selection), `Reset` (wipes all local data). | `Clear` reverts `Filter` → `Camera`, disables icon, clears selection. `Reset` requires confirmation, clears Room/DataStore, returns to empty state. |
| FR-10 | **Filtering & Matching** | Tapping `Filter` runs embedding comparison against selected clusters. | Matches flagged as `Keepers`. Non-matches discarded. Progress indicator shown during processing. |
| FR-11 | **Keeper Display** | Matched images displayed in a swipeable gallery. | Horizontal pager or swipeable `ViewPager2`. Empty state if 0 keepers found. |

---

## 🛠️ 3. Technical Architecture & Constraints (Integrated from KB)

### 3.1 Core Pipeline
1. **Detection:** `BlazeFace.tflite` (~200KB) → Returns bounding boxes + 6 landmarks.
2. **Alignment:** Pure Kotlin affine transform using landmarks → Crops to `112x112`, eyes aligned horizontally.
3. **Embedding:** `GhostFaceNet-V1` or `MobileFaceNet` (FP16 TFLite) → Outputs 128-d normalized vectors.
4. **Clustering (Phase 1):** Custom Kotlin DBSCAN on embeddings → Groups by density → Computes centroid (mean vector) per cluster.
5. **Matching (Phase 2):** Cosine Similarity/Dot Product against selected centroids → Threshold `≥0.8` = Keeper.

### 3.2 Runtime & Delegates
- **Runtime:** `tensorflow-lite-select-tf-ops` + Play Services `InterpreterApi`
- **Delegate:** `GpuDelegateFactory` with fallback to `XNNPACK` (CPU)
- **Threading:** `Dispatchers.Default` for math, `WorkManager`/`CoroutineScope` for batch TFLite inference. Main thread strictly for UI.

### 3.3 Storage & Persistence
| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Room DB** | Cluster metadata, centroid vectors, timestamps | `@Entity Cluster`, `@Dao ClusterDao`, TypeConverters for `FloatArray` |
| **DataStore** | UI state, app flags, selected cluster IDs | `Proto DataStore` or `Preferences DataStore` |
| **App Files** | Sidecar models (`blazeface.tflite`, `ghostfacenet.tflite`) | Downloaded on first run to `filesDir/ml_models/` |
| **Media URIs** | Reference to gallery images (no copies stored) | `content://` URIs persisted in Room |

### 3.4 APK Size Constraint Strategy
- **Zero third-party CV libraries** in `build.gradle`
- **Models:** Downloaded sidecar (~4MB total), stored externally
- **APK Growth:** ≤500KB (UI, business logic, Room, Coroutines only)
- **R8/ProGuard:** `minifyEnabled true`, `shrinkResources true`, strip debug metadata

---

## 🎨 4. UI/UX Requirements

### 4.1 Layout Structure
```
┌─────────────────────────────────────┐
│  [App Title: Face Mash]             │
│                                     │
│          ┌─────────────┐            │
│          │  + / Camera │  (Center)  │
│          └─────────────┘            │
│                                     │
│   [N photos selected]               │
│   🖼️ 🖼️ 🖼️ 🖼️  (Accordion Stack)  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │       [Clusterify / Filter]   │  │
│  └───────────────────────────────┘  │
│                                     │
│  ◀️ [🔵 Cluster1] [⬜ Cluster2] ... [+] ▶️ │
│  [Checkbox] [Checkbox]              │
└─────────────────────────────────────┘
```

### 4.2 Screen States & Interactions
| State | UI Elements Visible | Behavior |
|-------|---------------------|----------|
| **Empty** | Center `+`, Title | Tap `+` → Open `PhotoPicker` |
| **Selecting** | Count text, Accordion thumbnails, `Clusterify` | Thumbnails update live. `Clusterify` enabled if `N > 0` |
| **Processing** | Full-screen loading overlay + progress | Blocks UI. Shows `% faces processed` or spinner |
| **Clustered** | Horizontal cluster list, Checkboxes, `Camera` (center) | `Camera` disabled until ≥1 checkbox checked |
| **Filter Selected** | `Filter` icon, `Clear`, `Reset` (top-right) | Tap `Filter` → Start matching pipeline |
| **Results** | Swipeable `Keepers` gallery, `Back`/`Done` | Swipe left/right. Pull-to-dismiss or tap to save/share (future) |

### 4.3 Micro-Interactions & Accessibility
- **Haptics:** Light click on checkbox, medium thud on state change
- **Animations:** `SharedElementTransition` for thumbnail fan, `Scale` on `Clusterify` tap
- **Accessibility:** `contentDescription` for all icons, `TalkBack` support for cluster names
- **Error Handling:** Toast/Snackbar for `No faces detected`, `GPU not supported`, `Permission denied`

---

## 📐 5. Data Model (Simplified)
```kotlin
@Entity(tableName = "clusters")
data class FaceCluster(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val centroidEmbedding: ByteArray, // Serialized 128-float vector
    val representativeImageUri: String,
    val faceCount: Int,
    val createdAt: Long = System.currentTimeMillis()
)

data class AppState(
    val initialSelectionUris: List<String> = emptyList(),
    val filterSelectionUris: List<String> = emptyList(),
    val selectedClusterIds: Set<String> = emptySet(),
    val isProcessing: Boolean = false,
    val keepersUris: List<String> = emptyList()
)
```

---

## ⚙️ 6. Non-Functional Requirements
| Category | Requirement |
|----------|-------------|
| **Performance** | ≤1.5s per face inference. Batch processing ≤3s for typical 10-face input. |
| **Memory** | OOM prevention via sequential GPU buffer reuse. Max heap ≤512MB. |
| **Privacy** | 100% offline inference. No telemetry, no network calls after initial model download. |
| **Compatibility** | API 26+. Handles `READ_MEDIA_IMAGES` (API 33+). GPU delegate gracefully falls back to CPU. |
| **Testing** | Unit tests for DBSCAN/Cosine, Instrumentation tests for TFLite pipeline, UI tests for state transitions. |

---

## ❓ 7. Clarifying Questions for Next Iteration
1. **Keeper Output:** Should matched photos be saved to a dedicated `FaceMash/Keepers` folder, or is in-app viewing sufficient for v1?
2. **DBSCAN Noise Handling:** How should we handle faces that DBSCAN marks as outliers (not grouped into any cluster)? Should they be shown as "Unclassified" or silently ignored?
3. **Reset Confirmation:** Should `Reset` show a native dialog (`AlertDialog`) to prevent accidental data loss?
4. **Model Download Trigger:** Should models download automatically on first launch, or only when user taps `Clusterify` for the first time?
5. **UI Framework Preference:** Confirming Jetpack Compose as the baseline. Should I provide XML/ViewBinding equivalents for legacy compatibility?
6. **Threshold Tuning:** Should we expose a hidden debug slider for Cosine Similarity threshold (`0.75–0.85`) for QA/testing, or hardcode it to `0.80`?

---

✅ **Next Steps:**  
Reply with answers to the clarifying questions, and I will generate:
1. A complete `build.gradle.kts` configuration (optimized for `<500KB` growth)
2. Kotlin architecture blueprint (ViewModel + UseCase + TFLite Manager + Math Utils)
3. Compose UI code scaffold matching the state machine above
4. Detailed TFLite GPU delegate initialization + sidecar download logic

### 我

Before that, let us really break down all the image processing inference and algorithms (not code, but in text) so we know what is going on.

### Qwen

Here is a complete, code-free breakdown of the image processing pipeline, algorithms, and data flow for FaceMash, structured strictly around your two core use cases and technical constraints.

---

### 🔄 High-Level Pipeline Architecture
The system operates as a **two-phase, on-device computer vision pipeline**. All inference is executed locally via the TFLite GPU Play Services delegate. Mathematical operations, filtering logic, and state management are handled in pure Kotlin. The pipeline is split into:
1. **Phase 1: Discovery & Clustering** (`Clusterify` button flow)
2. **Phase 2: Matching & Filtering** (`Filter` button flow)

Both phases share the same foundational extraction steps, then diverge into different algorithmic branches.

---

### 📸 Phase 1: Discovery & Clustering (`Clusterify` Flow)

#### Step 1: Face Detection (Candidate Generation)
- **Input:** Raw gallery images selected by the user.
- **Engine:** `BlazeFace.tflite` (downloaded as a sidecar file, ~200KB).
- **Process:** The detector scans each image and outputs bounding boxes, six facial landmarks (eyes, nose, mouth, ears), and a confidence score per detection.
- **False-Positive Mitigation:** 
  - **Confidence Filtering:** Discard detections below a strict threshold (e.g., `0.75`).
  - **Geometric Validation:** Verify that the eye-to-face proportion matches human anatomy.
  - **Size-Based Outlier Removal:** Calculate the median face width in the image. Discard detections that deviate significantly (e.g., `>20%` from median) to ignore background patterns, posters, or distant artifacts.
- **Output:** Validated face regions with precise landmark coordinates.

#### Step 2: Face Alignment & Preprocessing
- **Purpose:** Recognition models degrade significantly if faces are tilted, scaled inconsistently, or off-center.
- **Process:** Using the six detected landmarks, an affine transformation matrix is computed. This maps the detected eyes and mouth to canonical "ideal" coordinates (e.g., eyes aligned horizontally at fixed relative positions). The original face region is warped, cropped, and resized to a standardized `112x112` pixel square. Pixel values are normalized to the model's expected input range.
- **Output:** Clean, pose-normalized face tensors ready for feature extraction.

#### Step 3: Embedding Generation (Feature Extraction)
- **Engine:** `GhostFaceNet-V1.tflite` (FP16 quantized, downloaded as a sidecar).
- **Process:** The aligned tensor is passed to the GPU-accelerated TFLite interpreter. The model processes the image through depthwise separable convolutions and "Ghost" modules, outputting a fixed-length `128-dimensional` floating-point vector.
- **Normalization:** The output vector is L2-normalized (converted to a unit vector). This ensures that similarity calculations rely on angular distance rather than magnitude, making the representation robust to lighting changes.
- **Output:** A `128-d` "face signature" representing the unique biometric features of that face.

#### Step 4: Clustering Algorithm (DBSCAN)
- **Input:** A batch of `128-d` embeddings extracted from the initial photo set.
- **Algorithm:** **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise).
- **Why DBSCAN?** It does not require knowing the number of people in advance. It naturally groups dense regions of vectors and isolates outliers.
- **Process:**
  1. Compute pairwise distances between all embeddings using cosine distance.
  2. Identify "core points" that have a minimum number of neighbors within a predefined radius (`epsilon` or density threshold).
  3. Expand clusters by linking density-reachable points.
  4. Mark embeddings that don't meet density requirements as "noise" (unclassified faces).
  5. For each valid cluster, compute a **centroid** by averaging the vectors of all faces in that group.
- **Storage:** Centroids, a representative image URI, face count, and timestamps are serialized and persisted in a local Room database.
- **Output:** A list of distinct face clusters, each representing a unique individual. These become the selectable round avatars in the UI.

---

### 🔍 Phase 2: Matching & Filtering (`Filter` Flow)

#### Step 1: Secondary Batch Extraction
- **Input:** Up to 15 new gallery images selected by the user.
- **Process:** Steps 1–3 from Phase 1 (Detection → Alignment → Embedding) are repeated for each face in this new batch. No clustering is performed at this stage.

#### Step 2: Similarity Matching Algorithm
- **Input:** Newly extracted embeddings + a subset of user-selected cluster centroids (from Room DB).
- **Algorithm:** **Cosine Similarity via Dot Product** + **Threshold Decision Logic**.
- **Process:**
  1. Because all vectors are L2-normalized, cosine similarity simplifies to a direct dot product between the new embedding and each selected centroid.
  2. The system computes similarity scores against all selected centroids.
  3. **Decision Threshold:** If the highest similarity score meets or exceeds `0.80`, the face is flagged as a match.
  4. **Multi-Face Handling:** If a single photo contains multiple faces, the image is kept as a "Keeper" if *at least one* face matches any selected cluster.
  5. **Ambiguity Handling:** If the top match is close to the threshold but the second-best match is nearly identical, the system can treat it as ambiguous and filter it out to maintain high precision.
- **Output:** A filtered list of "Keeper" image URIs. Non-matching images are discarded from the pipeline.

#### Step 3: Keeper Presentation
- **Process:** The matched image URIs are passed to the UI layer and rendered in a swipeable gallery. The original "Filter" state is cleared, and the app returns to a ready state for further filtering or session reset.

---

### ⚙️ Cross-Cutting Technical Mechanics

#### GPU Delegate & Memory Management
- The TFLite GPU Play Services delegate is initialized once at app launch. It queries hardware support at runtime and gracefully falls back to the CPU (XNNPACK) if the GPU is unavailable, unsupported, or overheating.
- **Batch Processing Strategy:** Inference runs sequentially or in micro-batches on a background `Coroutine` scope. GPU memory buffers are allocated once and reused to prevent OOM crashes during high-volume clustering. Source bitmaps are aggressively downsampled before detection to minimize heap pressure.

#### APK Growth Compliance (<500KB)
- **Zero Heavy Dependencies:** No OpenCV, no ML Kit SDK, no heavy math libraries. All alignment, DBSCAN, and similarity math are implemented in pure Kotlin using standard Android `Canvas`/`Matrix` and `FloatArray` operations.
- **Sidecar Model Strategy:** `BlazeFace` and `GhostFaceNet` are downloaded externally as `.tflite` files. They never touch the APK. The TFLite Play Services runtime handles their loading directly from internal storage.
- **ProGuard/R8 Optimization:** Full minification strips debug symbols, Kotlin metadata, and unused resources, ensuring the APK size increase remains strictly within the quota.

#### Data Flow Summary
```
Gallery Image → BlazeFace (Detect) → Affine Transform (Align) → GhostFaceNet (Embed)
       ↓
[Branch A: Clusterify] → DBSCAN → Compute Centroids → Save to Room DB → UI Cluster List
[Branch B: Filter] → Cosine Match vs Saved Centroids → Threshold ≥0.8 → Keepers → Swipeable Gallery
```

This architecture guarantees a fully offline, privacy-preserving, sub-1.5-second-per-face pipeline while strictly adhering to your APK growth constraint and Play Services GPU runtime limitation. 

Let me know if you want to dive deeper into any specific algorithmic step (e.g., how the DBSCAN density threshold is calibrated, or how the affine alignment math maps landmarks to canonical coordinates).
