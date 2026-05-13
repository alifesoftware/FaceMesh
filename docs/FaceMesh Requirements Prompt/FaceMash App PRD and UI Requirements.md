> From: https://chat.deepseek.com/a/chat/s/13358b07-4a75-48b1-a55c-2ef1ffaa8957

# you asked

[文件: Gemini-Android Face Recognition TFLite App.md]

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

---

# deepseek response

# FaceMash – Product & Technical Specification

| Document | PRD / BRD |
|----------|-----------|
| App Name | FaceMash |
| Package   | com.alifesoftware.facemash |
| Version   | 1.0 |
| Platform  | Android (native Kotlin) |
| Author    | – |
| Date      | 2026-05-08 |

---

## Table of Contents
1. [Product Overview](#1-product-overview)
2. [Business Requirements](#2-business-requirements)
3. [Functional Requirements](#3-functional-requirements)
4. [User Interface Requirements](#4-user-interface-requirements)
5. [Technical Architecture](#5-technical-architecture)
6. [Data & Storage](#6-data--storage)
7. [Dependencies & Constraints](#7-dependencies--constraints)
8. [Non‑Functional Requirements](#8-nonfunctional-requirements)
9. [Acceptance Criteria](#9-acceptance-criteria)
10. [Open Questions](#10-open-questions)

---

## 1. Product Overview

FaceMash is a **100% on‑device** Android application that lets a user:

1. **Discover people** in a large set of photos by grouping photos of the same person into “clusters” – without any prior knowledge of how many different people appear.
2. **Filter a new batch** of photos against the saved clusters, keeping only those images that contain a face matching one of the previously discovered identities.

The entire processing runs locally, using a custom **TensorFlow Lite (TFLite)** face recognition pipeline accelerated by **GPU Play Services**. The app adds **less than 500 KB** to the APK size and respects strict privacy and latency constraints.

---

## 2. Business Requirements

- **BR-01**: The app must deliver face clustering and face matching on device, with zero reliance on a backend server.
- **BR-02**: Total APK size increase caused by the face‑processing logic (libraries, models, assets) must not exceed 500 KB.
- **BR-03**: Face models (detector + recogniser) shall be downloaded after install and stored securely on the device; they are not included in the APK.
- **BR-04**: All inference must use the **TFLite GPU Play Services** delegate; no other runtime (NNAPI, CPU only, etc.) shall be required, though a fallback is allowed.
- **BR-05**: The user interface must be clean, self‑explanatory, and work equally well on flagship and mid‑range devices (Android 10+, API 29+).

---

## 3. Functional Requirements

### 3.1 Initial State (Empty App)

| ID | Requirement |
|----|-------------|
| FR‑01 | On first launch, the home screen displays the title “FaceMash” and a prominent **“+”** button in the centre. |
| FR‑02 | No other UI elements (clusters, photos, buttons) are visible until the user adds photos. |

### 3.2 Adding Photos for Clustering

| ID | Requirement |
|----|-------------|
| FR‑03 | Tapping the **“+”** button opens the system photo picker restricted to one folder/album at a time. |
| FR‑04 | The user may select images from the same folder only per session; to add images from another folder they must tap “+” again. |
| FR‑05 | While selecting, a counter “**N photos selected**” updates in real time. |
| FR‑06 | After confirming the selection, a visual “fan” of the last 3‑4 selected thumbnails is displayed overlapping behind the “+” button. |
| FR‑07 | The thumbnail fan is updated each time the user adds more photos (the fan always shows the most recent 3‑4). |
| FR‑08 | A **“Clusterify”** button appears at the bottom of the screen as soon as at least one photo has been selected. |

### 3.3 Clustering Phase

| ID | Requirement |
|----|-------------|
| FR‑09 | Tapping **“Clusterify”** triggers the face clustering pipeline: detect faces, generate embeddings, run DBSCAN, and compute cluster centroids. |
| FR‑10 | The TF‑Lite models (BlazeFace detector, GhostFaceNet‑V1 recogniser) are downloaded on first use if not already present. |
| FR‑11 | During processing, the UI shows an indeterminate progress indicator and disables all interactive elements. |
| FR‑12 | If no face is detected in any image, the app displays a “No faces found” message and returns to the home state. |
| FR‑13 | The clusters are saved persistently in local storage. Each cluster stores: unique ID, centroid vector, a representative thumbnail (first valid face), and list of source image URIs. |

### 3.4 Post‑Clustering UI

| ID | Requirement |
|----|-------------|
| FR‑14 | After clustering succeeds, the centre “+” is replaced by a **“Camera”** icon (disabled if no cluster is checked). |
| FR‑15 | A horizontally scrollable row of **cluster avatars** appears at the bottom of the screen. |
| FR‑16 | Each avatar is a circular crop of one representative face from that cluster. |
| FR‑17 | Next to each cluster avatar, a **checkbox** allows selection/deselection of that cluster. |
| FR‑18 | The last item in the row is a circular **“+”** icon that returns the user to the photo‑addition flow (see FR‑03). |
| FR‑19 | The **Camera icon** is enabled only when **at least one** cluster is checked. |

### 3.5 Filtering Phase (Matching Against Clusters)

| ID | Requirement |
|----|-------------|
| FR‑20 | Tapping the enabled **Camera icon** opens the photo picker, limited to **maximum 15 images**. |
| FR‑21 | After images are selected, the Camera icon changes to a **“Filter”** icon. |
| FR‑22 | A **“Clear”** button appears in the top‑right corner. Pressing Clear removes the selected filtering images, reverts the Filter icon back to Camera (disabled), and keeps clusters unchanged. |
| FR‑23 | A **“Reset”** button appears in the top‑right corner. Pressing Reset clears **all** app data (clusters + filtering photos) and returns the app to the empty state (FR‑01). |
| FR‑24 | Tapping the **Filter** icon starts the matching process: each face found in the selected images is compared against the centroids of the **checked** clusters only. |
| FR‑25 | An image is a “Keeper” if **at least one** face in that image matches any checked cluster (distance < threshold). |
| FR‑26 | Images with no matching face are discarded. |
| FR‑27 | The matching process shows a progress indicator and then transitions to a **full‑screen swipeable gallery** of the “Keeper” images. |
| FR‑28 | The user can swipe left/right to browse the matched images. A “Done” or back button returns to the previous screen with the clusters still intact. |

### 3.6 Edge Cases & Error Handling

| ID | Scenario | Expected Behaviour |
|----|----------|--------------------|
| FR‑29 | Model download fails or times out | Retry with exponential backoff; show error message and allow manual retry. |
| FR‑30 | GPU delegate not available (older device) | Fall back to XNNPACK CPU delegate; indicate slightly longer processing. |
| FR‑31 | User selects >15 images in filtering mode | Picker must enforce the limit; if Android picker cannot, show a toast and ignore extras. |
| FR‑32 | No faces found during filtering | Show “No matches found” and return to the cluster screen. |
| FR‑33 | All clusters deselected while Camera icon is enabled | Camera becomes disabled immediately. |
| FR‑34 | Very large originals (e.g., 48 MP) | Down‑scale to reasonable size before detection to avoid OOM. |

---

## 4. User Interface Requirements

### 4.1 Home Screen – Empty State

![Empty State](placeholder)

- **Layout**: Single screen, centred vertically and horizontally.
- **Elements**:
  - App title “FaceMash” in a large, bold, modern font.
  - A circular **“+”** button (48dp touch target) with a subtle shadow.
- **Background**: Gradient or soft solid colour, no clutter.

### 4.2 Home Screen – Photos Selected (Before Clustering)

- **Elements**:
  - **“N photos selected”** text appears above the “+” button.
  - A **fan‑shaped stack** of 3‑4 round‑cornered thumbnails sits behind the “+” button, each rotated slightly (e.g. ±8°). The stack updates as photos are added.
  - A **“Clusterify”** button at the bottom, full width, rounded corners, primary accent colour.
- **Behaviour**: 
  - Tapping “Clusterify” triggers the clustering pipeline.
  - The “+” button remains visible; tapping it again adds more photos.

### 4.3 Home Screen – Clusters Present (Waiting for Filtering)

- **Elements**:
  - **Centre**: “Camera” icon (circular, outline camera shape).  
    - *Disabled state*: greyed out, not interactive.  
    - *Enabled state*: filled accent colour, tappable.
  - **Bottom bar**: Horizontal `RecyclerView` with:
    - Cluster avatars (64dp circle) with a small checkbox overlay at the top‑left corner of each avatar.
    - Final item is a circular “+” (add more photos).
  - **Top‑right corner**:
    - “Clear” (visible only when filter images exist).
    - “Reset” (always visible) – icon or text button.
- **Animations**: 
  - Smooth fade/slide when clusters appear.
  - Checkbox toggle with a short colour fill animation.

### 4.4 Camera → Filter Transition

- When the user picks filter photos:
  - The Camera icon morphs into a **filter‑funnel** icon.
  - “Clear” button fades in.
- Tapping “Clear”: Filter icon morphs back to Camera (disabled), selected images cleared.
- Tapping “Reset”: Entire UI fades out and returns to empty state.

### 4.5 Filtering Progress

- Full‑screen translucent overlay with:
  - Indeterminate progress ring.
  - Text “Filtering… X/Y images”.
  - Cancel button (stops processing, discards results, returns to home with clusters intact).

### 4.6 Keeper Gallery

- **Full‑screen** `ViewPager2` or similar with smooth swipe.
- Each page shows one image centred with aspect‑ratio preserving scaling.
- Thumbnail indicator dots at the bottom (if more than 1 image).
- Navigation bar with “Back” arrow and optional “Save / Share” action.

### 4.7 Visual Design Guidelines

- **Colours**: Primary (#1565C0), Accent (#FF6F00), Surface (#FFFFFF), Error (#B00020).
- **Typography**: System default (Roboto), title size 24sp, body 14sp.
- **Icons**: Material Symbols (Google Fonts) to keep APK small.
- **Scale**: All touch targets ≥48dp.

---

## 5. Technical Architecture

### 5.1 Model Pipeline

```
Image → [BlazeFace Detector] → Crop & Align → [GhostFaceNet Recogniser] → 128‑d Embedding
```

**Detection**: BlazeFace (TFLite) – provides bounding boxes and 6 landmarks.  
**Recognition**: GhostFaceNet‑V1 (Float16 TFLite) – 128‑d embedding.

Both models are downloaded together in a single archive after first launch.

**Reasoning**:  
- BlazeFace is lighter (~200 KB) and more accurate than the Android‑native `FaceDetector`.  
- GhostFaceNet‑V1 offers the best speed/accuracy trade‑off for mobile GPUs, performing in <20 ms even on mid‑range Adreno GPUs.  
- All ops are supported by the Play Services TFLite GPU delegate.

### 5.2 Clustering Algorithm

**DBSCAN (Density‑Based Spatial Clustering of Applications with Noise)**

- Distance metric: **Cosine similarity** (simplified to dot product for normalised vectors).
- Parameters: `eps` (distance threshold) and `min_samples` (minimum faces to form a cluster) – tuned empirically and stored in the cloud‑downloaded config.
- DBSCAN automatically determines the number of clusters and marks outliers as noise.

### 5.3 Cluster Storage

**Schema (Room Database)**:

```kotlin
@Entity
data class Cluster(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val centroid: FloatArray,       // 128 floats, stored as blob
    val representativeImageUri: String,
    val createdAt: Long
)

@Entity
data class ClusterImage(
    @PrimaryKey val imageUri: String,
    val clusterId: String,          // foreign key to Cluster
    val embedding: FloatArray       // 128 floats, stored once per face
)
```

**Centroid Updating** (future enhancement): weighted average of new matching embeddings.

### 5.4 Matching Logic (Filtering)

1. For each selected image:
   - Detect all faces.
   - For each face, compute embedding.
   - Compare embedding to each **checked** cluster centroid using cosine similarity.
   - If `similarity > threshold` (e.g., 0.8) for any cluster, mark image as Keeper.
2. Stop early if one face matches (image is a Keeper), but process all faces to handle multiple people.

### 5.5 Threading & Performance

- All heavy work (inference, clustering) runs on a background thread (`Coroutine` with `Dispatchers.Default` or `WorkManager` if long‑running).
- Bitmap decoding is performed with `inSampleSize` to keep memory under control.
- Maximum concurrent GPU operations carefully managed to avoid OOM.

### 5.6 Model Delivery & APK Size

- **Zero models in APK**.  
- On first use, the app downloads a ZIP containing:
  - `blazeface.tflite` (~200 KB)
  - `facemeshnet_v1_fp16.tflite` (~2.5 MB)
  - `config.json` (thresholds, input sizes)
- APK size increase from face‑recognition logic consists only of:
  - Kotlin code (a few tens of KB after R8 minification).
  - Room library dependency (already justified if using Room; otherwise a custom file‑based storage could be even slimmer).
- This easily fits in the **500 KB** quota.

---

## 6. Data & Storage

| Data | Location | Persistence |
|------|----------|-------------|
| Clusters + centroids | Room DB (private internal storage) | Permanent |
| Source image URIs | Room (reference only) | Permanent |
| Downloaded models | `filesDir/models/` | Permanent |
| Temporary filter images | In‑memory (URIs) | Ephemeral |
| User preferences (thresholds) | SharedPreferences / DataStore | Permanent |

---

## 7. Dependencies & Constraints

### 7.1 Android

- `minSdkVersion`: 29 (Android 10)
- `targetSdkVersion`: 35
- `compileSdkVersion`: 35
- Build Tools: latest stable
- Language: Kotlin 2.x

### 7.2 Libraries

| Purpose | Library | APK Impact |
|---------|---------|------------|
| TFLite Runtime | `org.tensorflow:tensorflow-lite-gpu-play-services` | 0 KB (Play Services) |
| Image processing | Android `Canvas`, `Matrix` | 0 KB |
| Async | Kotlin Coroutines (`org.jetbrains.kotlinx:kotlinx-coroutines-android`) | ~50 KB |
| Clustering math | Custom implementation (no external lib) | 0 KB |
| Local DB | Room or manual Protobuf file | ~100 KB (Room) |
| UI | Material 3 (`com.google.android.material:material`) | ~200 KB (if needed) |

**Total APK growth < 500 KB**.

### 7.3 Constraints

- Must run offline after model download.
- Must not require Google ML Kit or MediaPipe (to avoid APK bloat).
- Must use only standard Play Services TFLite delegate ops.
- Processing latency target: cluster creation < 1.5 s per image (average), filtering < 2 s total for up to 15 images.

---

## 8. Non‑Functional Requirements

| NFR‑01 | **Privacy**: All facial data stays on device; no face data leaves the app. |
| NFR‑02 | **Resilience**: Graceful degradation if GPU delegate unavailable; fallback to CPU. |
| NFR‑03 | **Usability**: A person with no technical knowledge should be able to create clusters in under 3 minutes. |
| NFR‑04 | **Performance**: App must not freeze or drop frames during processing; UI always responsive. |
| NFR‑05 | **Storage**: Total app footprint (excluding user photos) < 20 MB. |
| NFR‑06 | **Thermal**: Inference loops must yield periodically to avoid thermal throttling. |

---

## 9. Acceptance Criteria

1. **Empty state**: Launch app → only “+” visible.
2. **Add photos**: Tap “+” → pick photos → “N photos selected” + thumbnail fan shown.
3. **Clusterify**: Tap “Clusterify” → loading → clusters appear as circular avatars; “+” becomes Camera (disabled).
4. **Cluster selection**: Check a cluster → Camera enabled; uncheck all → Camera disabled.
5. **Filter**: Tap Camera → pick ≤15 photos → Camera becomes Filter; Clear/Reset visible.
6. **Filter result**: Tap Filter → processing → swipeable gallery of matched images only.
7. **Clear/Reset**: Clear removes filter photos; Reset wipes everything to empty state.
8. **Edge**: No faces → appropriate message; no match → message.
9. **Model download**: On first “Clusterify” tap, models download; progress shown.

---

## 10. Open Questions

1. **Thumbnail fan**: Should it reflect the most recent 3–4 selected photos **globally** or **per folder**?
2. **Cluster name**: Future requirement? Currently no naming; only visual circles.
3. **Centroid update**: If a new image matches a cluster during filtering, should the centroid be updated automatically? (Suggested: off by default, manual “confirm” mode.)
4. **Delete single cluster**: Not in spec; should we allow swipe‑to‑delete on cluster avatar?
5. **Sharing keepers**: Should we provide “Share all keepers” button in gallery?
6. **Cluster merging**: What if two clusters actually represent the same person? (Out of scope for v1).
7. **Performance feedback**: Should we show per‑image processing time or just overall progress?

---

*End of Document*

---

# you asked

1. "Thumbnail fan: Should it reflect the most recent 3–4 selected photos globally or per folder?" => Globally
2. "Cluster name: Future requirement? Currently no naming; only visual circles." => Yes
3. "Centroid update: If a new image matches a cluster during filtering, should the centroid be updated automatically? (Suggested: off by default, manual “confirm” mode.)" => Sure
4. "Delete single cluster: Not in spec; should we allow swipe‑to‑delete on cluster avatar?" => Yes

---

# deepseek response

# FaceMash – Finalized Product & Technical Specification

**(Updated with User Decisions from 2026-05-08)**

---

## Table of Contents
1. [Product Overview](#1-product-overview)
2. [Business Requirements](#2-business-requirements)
3. [Functional Requirements](#3-functional-requirements)
4. [User Interface Requirements](#4-user-interface-requirements)
5. [Technical Architecture](#5-technical-architecture)
6. [Data & Storage](#6-data--storage)
7. [Dependencies & Constraints](#7-dependencies--constraints)
8. [Non‑Functional Requirements](#8-nonfunctional-requirements)
9. [Acceptance Criteria](#9-acceptance-criteria)
10. [Open Questions / Future Enhancements](#10-open-questions--future-enhancements)

---

## 1. Product Overview

FaceMash is a **100% on‑device** Android application that lets a user:

1. **Discover people** in a large set of photos by grouping photos of the same person into “clusters” – without any prior knowledge of how many different people appear.
2. **Filter a new batch** of photos against the saved clusters, keeping only those images that contain a face matching one of the previously discovered identities.

The entire processing runs locally, using a custom **TensorFlow Lite (TFLite)** face recognition pipeline accelerated by **GPU Play Services**. The app adds **less than 500 KB** to the APK size and respects strict privacy and latency constraints.

---

## 2. Business Requirements

- **BR-01**: The app must deliver face clustering and face matching on device, with zero reliance on a backend server.
- **BR-02**: Total APK size increase caused by the face‑processing logic (libraries, models, assets) must not exceed 500 KB.
- **BR-03**: Face models (detector + recogniser) shall be downloaded after install and stored securely on the device; they are not included in the APK.
- **BR-04**: All inference must use the **TFLite GPU Play Services** delegate; no other runtime (NNAPI, CPU only, etc.) shall be required, though a fallback is allowed.
- **BR-05**: The user interface must be clean, self‑explanatory, and work equally well on flagship and mid‑range devices (Android 10+, API 29+).

---

## 3. Functional Requirements

### 3.1 Initial State (Empty App)

| ID | Requirement |
|----|-------------|
| FR‑01 | On first launch, the home screen displays the title “FaceMash” and a prominent **“+”** button in the centre. |
| FR‑02 | No other UI elements (clusters, photos, buttons) are visible until the user adds photos. |

### 3.2 Adding Photos for Clustering

| ID | Requirement |
|----|-------------|
| FR‑03 | Tapping the **“+”** button opens the system photo picker restricted to one folder/album at a time. |
| FR‑04 | The user may select images from the same folder only per session; to add images from another folder they must tap “+” again. |
| FR‑05 | While selecting, a counter “**N photos selected**” updates in real time. |
| FR‑06 | After confirming the selection, a visual “fan” of the last 3‑4 selected thumbnails is displayed overlapping behind the “+” button. |
| FR‑07 | The thumbnail fan always reflects the **most recent 3‑4 selected photos globally**, independent of which folder they came from. |
| FR‑08 | A **“Clusterify”** button appears at the bottom of the screen as soon as at least one photo has been selected. |

### 3.3 Clustering Phase

| ID | Requirement |
|----|-------------|
| FR‑09 | Tapping **“Clusterify”** triggers the face clustering pipeline: detect faces, generate embeddings, run DBSCAN, and compute cluster centroids. |
| FR‑10 | The TF‑Lite models (BlazeFace detector, GhostFaceNet‑V1 recogniser) are downloaded on first use if not already present. |
| FR‑11 | During processing, the UI shows an indeterminate progress indicator and disables all interactive elements. |
| FR‑12 | If no face is detected in any image, the app displays a “No faces found” message and returns to the home state. |
| FR‑13 | The clusters are saved persistently in local storage. Each cluster stores: unique ID, centroid vector, a representative thumbnail (first valid face), and list of source image URIs. |

### 3.4 Post‑Clustering UI

| ID | Requirement |
|----|-------------|
| FR‑14 | After clustering succeeds, the centre “+” is replaced by a **“Camera”** icon (disabled if no cluster is checked). |
| FR‑15 | A horizontally scrollable row of **cluster avatars** appears at the bottom of the screen. |
| FR‑16 | Each avatar is a circular crop of one representative face from that cluster. |
| FR‑17 | Next to each cluster avatar, a **checkbox** allows selection/deselection of that cluster. |
| FR‑18 | The last item in the row is a circular **“+”** icon that returns the user to the photo‑addition flow (see FR‑03). |
| FR‑19 | The **Camera icon** is enabled only when **at least one** cluster is checked. |
| FR‑20 | **Swipe‑to‑delete** on a cluster avatar removes that cluster permanently after a confirmation dialog. |

### 3.5 Filtering Phase (Matching Against Clusters)

| ID | Requirement |
|----|-------------|
| FR‑21 | Tapping the enabled **Camera icon** opens the photo picker, limited to **maximum 15 images**. |
| FR‑22 | After images are selected, the Camera icon changes to a **“Filter”** icon. |
| FR‑23 | A **“Clear”** button appears in the top‑right corner. Pressing Clear removes the selected filtering images, reverts the Filter icon back to Camera (disabled), and keeps clusters unchanged. |
| FR‑24 | A **“Reset”** button appears in the top‑right corner. Pressing Reset clears **all** app data (clusters + filtering photos) and returns the app to the empty state (FR‑01). |
| FR‑25 | Tapping the **Filter** icon starts the matching process: each face found in the selected images is compared against the centroids of the **checked** clusters only. |
| FR‑26 | An image is a “Keeper” if **at least one** face in that image matches any checked cluster (distance < threshold). |
| FR‑27 | Images with no matching face are discarded. |
| FR‑28 | The matching process shows a progress indicator and then transitions to a **full‑screen swipeable gallery** of the “Keeper” images. |
| FR‑29 | The user can swipe left/right to browse the matched images. A “Done” or back button returns to the previous screen with the clusters still intact. |

### 3.6 Edge Cases & Error Handling

| ID | Scenario | Expected Behaviour |
|----|----------|--------------------|
| FR‑30 | Model download fails or times out | Retry with exponential backoff; show error message and allow manual retry. |
| FR‑31 | GPU delegate not available (older device) | Fall back to XNNPACK CPU delegate; indicate slightly longer processing. |
| FR‑32 | User selects >15 images in filtering mode | Picker must enforce the limit; if Android picker cannot, show a toast and ignore extras. |
| FR‑33 | No faces found during filtering | Show “No matches found” and return to the cluster screen. |
| FR‑34 | All clusters deselected while Camera icon is enabled | Camera becomes disabled immediately. |
| FR‑35 | Very large originals (e.g., 48 MP) | Down‑scale to reasonable size before detection to avoid OOM. |
| FR‑36 | Delete last remaining cluster | Row disappears, centre reverts to “+” (empty state). |

---

## 4. User Interface Requirements

### 4.1 Home Screen – Empty State

- **Layout**: Single screen, centred vertically and horizontally.
- **Elements**:
  - App title “FaceMash” in a large, bold, modern font.
  - A circular **“+”** button (48dp touch target) with a subtle shadow.
- **Background**: Gradient or soft solid colour, no clutter.

### 4.2 Home Screen – Photos Selected (Before Clustering)

- **Elements**:
  - **“N photos selected”** text appears above the “+” button.
  - A **fan‑shaped stack** of 3‑4 round‑cornered thumbnails sits behind the “+” button, each rotated slightly (e.g. ±8°). The stack updates as photos are added, always showing the most recent **globally**.
  - A **“Clusterify”** button at the bottom, full width, rounded corners, primary accent colour.
- **Behaviour**: 
  - Tapping “Clusterify” triggers the clustering pipeline.
  - The “+” button remains visible; tapping it again adds more photos.

### 4.3 Home Screen – Clusters Present (Waiting for Filtering)

- **Elements**:
  - **Centre**: “Camera” icon (circular, outline camera shape).  
    - *Disabled state*: greyed out, not interactive.  
    - *Enabled state*: filled accent colour, tappable.
  - **Bottom bar**: Horizontal `RecyclerView` with:
    - Cluster avatars (64dp circle) with a small checkbox overlay at the top‑left corner of each avatar.
    - Swipe‑left gesture reveals delete option (with confirmation).
    - Final item is a circular “+” (add more photos).
  - **Top‑right corner**:
    - “Clear” (visible only when filter images exist).
    - “Reset” (always visible) – icon or text button.
- **Animations**: 
  - Smooth fade/slide when clusters appear.
  - Checkbox toggle with a short colour fill animation.

### 4.4 Camera → Filter Transition

- When the user picks filter photos:
  - The Camera icon morphs into a **filter‑funnel** icon.
  - “Clear” button fades in.
- Tapping “Clear”: Filter icon morphs back to Camera (disabled), selected images cleared.
- Tapping “Reset”: Entire UI fades out and returns to empty state.

### 4.5 Filtering Progress

- Full‑screen translucent overlay with:
  - Indeterminate progress ring.
  - Text “Filtering… X/Y images”.
  - Cancel button (stops processing, discards results, returns to home with clusters intact).

### 4.6 Keeper Gallery

- **Full‑screen** `ViewPager2` or similar with smooth swipe.
- Each page shows one image centred with aspect‑ratio preserving scaling.
- Thumbnail indicator dots at the bottom (if more than 1 image).
- Navigation bar with “Back” arrow and optional “Save / Share” action.

### 4.7 Visual Design Guidelines

- **Colours**: Primary (#1565C0), Accent (#FF6F00), Surface (#FFFFFF), Error (#B00020).
- **Typography**: System default (Roboto), title size 24sp, body 14sp.
- **Icons**: Material Symbols (Google Fonts) to keep APK small.
- **Scale**: All touch targets ≥48dp.

---

## 5. Technical Architecture

### 5.1 Model Pipeline

```
Image → [BlazeFace Detector] → Crop & Align → [GhostFaceNet Recogniser] → 128‑d Embedding
```

**Detection**: BlazeFace (TFLite) – provides bounding boxes and 6 landmarks.  
**Recognition**: GhostFaceNet‑V1 (Float16 TFLite) – 128‑d embedding.

Both models are downloaded together in a single archive after first launch.

**Reasoning**:  
- BlazeFace is lighter (~200 KB) and more accurate than the Android‑native `FaceDetector`.  
- GhostFaceNet‑V1 offers the best speed/accuracy trade‑off for mobile GPUs, performing in <20 ms even on mid‑range Adreno GPUs.  
- All ops are supported by the Play Services TFLite GPU delegate.

### 5.2 Clustering Algorithm

**DBSCAN (Density‑Based Spatial Clustering of Applications with Noise)**

- Distance metric: **Cosine similarity** (simplified to dot product for normalised vectors).
- Parameters: `eps` (distance threshold) and `min_samples` (minimum faces to form a cluster) – tuned empirically and stored in the cloud‑downloaded config.
- DBSCAN automatically determines the number of clusters and marks outliers as noise.

### 5.3 Cluster Storage

**Schema (Room Database)**:

```kotlin
@Entity
data class Cluster(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val centroid: FloatArray,       // 128 floats, stored as blob
    val representativeImageUri: String,
    val createdAt: Long
)

@Entity
data class ClusterImage(
    @PrimaryKey val imageUri: String,
    val clusterId: String,          // foreign key to Cluster
    val embedding: FloatArray       // 128 floats, stored once per face
)
```

**Future**: Cluster naming (user‑specified label) will be added as a separate field.

### 5.4 Matching Logic (Filtering)

1. For each selected image:
   - Detect all faces.
   - For each face, compute embedding.
   - Compare embedding to each **checked** cluster centroid using cosine similarity.
   - If `similarity > threshold` (e.g., 0.8) for any cluster, mark image as Keeper.
2. Stop early if one face matches (image is a Keeper), but process all faces to handle multiple people.

### 5.5 Centroid Update (Manual Confirm)

When a new image matches a cluster during filtering, the app can optionally offer a “Confirm” mode (future enhancement) where the user can choose to update the cluster centroid with the new embedding, improving future accuracy. This is off by default in v1.

### 5.6 Swipe‑to‑Delete Cluster

- Swipe gesture on a cluster avatar reveals a red “Delete” button.
- A confirmation dialog (“Delete this cluster of N photos?”) prevents accidental removal.
- Deletion cascades: removes all associated images from the database and updates UI.

### 5.7 Threading & Performance

- All heavy work (inference, clustering) runs on a background thread (`Coroutine` with `Dispatchers.Default` or `WorkManager` if long‑running).
- Bitmap decoding is performed with `inSampleSize` to keep memory under control.
- Maximum concurrent GPU operations carefully managed to avoid OOM.

### 5.8 Model Delivery & APK Size

- **Zero models in APK**.  
- On first use, the app downloads a ZIP containing:
  - `blazeface.tflite` (~200 KB)
  - `facemeshnet_v1_fp16.tflite` (~2.5 MB)
  - `config.json` (thresholds, input sizes)
- APK size increase from face‑recognition logic consists only of:
  - Kotlin code (a few tens of KB after R8 minification).
  - Room library dependency (already justified if using Room; otherwise a custom file‑based storage could be even slimmer).
- This easily fits in the **500 KB** quota.

---

## 6. Data & Storage

| Data | Location | Persistence |
|------|----------|-------------|
| Clusters + centroids | Room DB (private internal storage) | Permanent |
| Source image URIs | Room (reference only) | Permanent |
| Downloaded models | `filesDir/models/` | Permanent |
| Temporary filter images | In‑memory (URIs) | Ephemeral |
| User preferences (thresholds) | SharedPreferences / DataStore | Permanent |

---

## 7. Dependencies & Constraints

### 7.1 Android

- `minSdkVersion`: 29 (Android 10)
- `targetSdkVersion`: 35
- `compileSdkVersion`: 35
- Build Tools: latest stable
- Language: Kotlin 2.x

### 7.2 Libraries

| Purpose | Library | APK Impact |
|---------|---------|------------|
| TFLite Runtime | `org.tensorflow:tensorflow-lite-gpu-play-services` | 0 KB (Play Services) |
| Image processing | Android `Canvas`, `Matrix` | 0 KB |
| Async | Kotlin Coroutines (`org.jetbrains.kotlinx:kotlinx-coroutines-android`) | ~50 KB |
| Clustering math | Custom implementation (no external lib) | 0 KB |
| Local DB | Room or manual Protobuf file | ~100 KB (Room) |
| UI | Material 3 (`com.google.android.material:material`) | ~200 KB (if needed) |

**Total APK growth < 500 KB**.

### 7.3 Constraints

- Must run offline after model download.
- Must not require Google ML Kit or MediaPipe (to avoid APK bloat).
- Must use only standard Play Services TFLite delegate ops.
- Processing latency target: cluster creation < 1.5 s per image (average), filtering < 2 s total for up to 15 images.

---

## 8. Non‑Functional Requirements

| NFR‑01 | **Privacy**: All facial data stays on device; no face data leaves the app. |
| NFR‑02 | **Resilience**: Graceful degradation if GPU delegate unavailable; fallback to CPU. |
| NFR‑03 | **Usability**: A person with no technical knowledge should be able to create clusters in under 3 minutes. |
| NFR‑04 | **Performance**: App must not freeze or drop frames during processing; UI always responsive. |
| NFR‑05 | **Storage**: Total app footprint (excluding user photos) < 20 MB. |
| NFR‑06 | **Thermal**: Inference loops must yield periodically to avoid thermal throttling. |
| NFR‑07 | **Offline**: Fully functional without internet after model download. |

---

## 9. Acceptance Criteria

1. **Empty state**: Launch app → only “+” visible.
2. **Add photos**: Tap “+” → pick photos → “N photos selected” + thumbnail fan (global last 3‑4) shown.
3. **Clusterify**: Tap “Clusterify” → loading → clusters appear as circular avatars; “+” becomes Camera (disabled).
4. **Cluster selection**: Check a cluster → Camera enabled; uncheck all → Camera disabled.
5. **Swipe‑to‑delete**: Swipe a cluster → delete confirmation → cluster removed, UI updates accordingly.
6. **Filter**: Tap Camera → pick ≤15 photos → Camera becomes Filter; Clear/Reset visible.
7. **Filter result**: Tap Filter → processing → swipeable gallery of matched images only.
8. **Clear/Reset**: Clear removes filter photos; Reset wipes everything to empty state.
9. **Edge**: No faces → appropriate message; no match → message.
10. **Model download**: On first “Clusterify” tap, models download; progress shown.

---

## 10. Open Questions / Future Enhancements

*Resolved items are marked with a ✓.*

| # | Item | Status |
|---|------|--------|
| 1 | Thumbnail fan reflects **globally** | ✓ Resolved |
| 2 | Cluster naming will be a future feature | Future release |
| 3 | Manual centroid update confirm mode | Optional in v1.1 |
| 4 | Swipe‑to‑delete cluster | ✓ Included |
| 5 | Cluster merging (two clusters same person) | Out of scope |
| 6 | Share keepers button | Nice‑to‑have for v1.1 |
| 7 | Performance breakdown display | Not required |

---

*End of Document*