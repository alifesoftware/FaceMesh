package com.alifesoftware.facemesh.config

/**
 * Centralised configuration for the FaceMesh image pipeline.
 *
 * Every "knob" used by the detection -> filter -> align -> embed -> cluster -> match pipeline
 * lives here, with rich KDoc on each property that the IDE surfaces at every call site
 * (hover-docs, jump-to-definition, find-usages all work).
 *
 * ### Three classes of knob
 *
 *  1. **MODEL CONTRACT** - locked to the on-disk `.tflite` file or anchor table. Changing one
 *     of these without re-exporting / regenerating the corresponding model will break
 *     inference. Treat them as documentation, not as tuning targets.
 *
 *  2. **TUNABLE HEURISTIC** - compile-time default that an engineer adjusts when chasing
 *     recall vs. precision. The corresponding constructor parameters at the call sites still
 *     accept overrides (so unit tests can inject), but the production default lives here.
 *
 *  3. **USER SETTING DEFAULT** - default for a value the user can override at runtime via
 *     [com.alifesoftware.facemesh.data.AppPreferences]. Each one also exposes a `min*` /
 *     `max*` so any future Settings UI slider (or downloaded `manifest.json` config override)
 *     can clamp from the same source.
 *
 * Each property's KDoc follows a fixed shape so the docs stay scannable:
 *
 * ```
 *   - Class:    one of [MODEL CONTRACT | TUNABLE HEURISTIC | USER SETTING DEFAULT]
 *   - Used by:  the call site(s) that read it
 *   - Range:    valid range (where one exists)
 *   - Lower:    behavioural effect of decreasing the value
 *   - Higher:   behavioural effect of increasing the value
 *   - Why:      rationale for the current value
 * ```
 */
object PipelineConfig {

    // ---------------------------------------------------------------------------------------
    // Decode
    // ---------------------------------------------------------------------------------------

    /**
     * [com.alifesoftware.facemesh.media.BitmapDecoder] controls.
     */
    object Decode {
        /**
         * Cap on the longer edge of a decoded source bitmap (after EXIF rotation, before
         * detection / alignment).
         *
         *   - Class:    TUNABLE HEURISTIC
         *   - Used by:  `BitmapDecoder.decode`
         *   - Range:    512..4096 (powers of two preferred so `inSampleSize` halving lands clean)
         *   - Lower:    faster decode, less peak memory, but **smaller faces in source pixels**
         *               so the affine-aligned 112x112 embedder input loses high-frequency
         *               detail on already-small faces -> noisier embeddings, looser clusters.
         *               Detector itself is unaffected (it always letterboxes to
         *               [Detector.inputSize]).
         *   - Higher:   sharper embeddings on small faces in group photos; slower decode,
         *               higher peak RAM during processing.
         *   - Why 1280: empirically the smallest cap that keeps phone-photo faces aligned at
         *               >= 80px on input while staying under the SPEC NFR-04 mem budget.
         */
        const val maxLongEdgePx: Int = 1280
    }

    // ---------------------------------------------------------------------------------------
    // Detector (BlazeFace short-range)
    // ---------------------------------------------------------------------------------------

    /**
     * BlazeFace face-detection knobs ([com.alifesoftware.facemesh.ml.BlazeFaceDetector] +
     * [com.alifesoftware.facemesh.ml.BlazeFaceDecoder]).
     */
    object Detector {
        /**
         * Detector input edge size in pixels. The source bitmap is letterbox-resized into
         * [inputSize] x [inputSize] x 3.
         *
         *   - Class:    MODEL CONTRACT (short-range BlazeFace ships at 128)
         *   - Used by:  `BlazeFaceDetector` (input buffer + letterbox), `BlazeFaceDecoder`
         *               (anchor unprojection)
         *   - Note:     the ManifestConfig.detectorInput from the downloaded model bundle
         *               overrides this at runtime via `MlPipelineProvider.ensureProcessor`.
         *               Do NOT change this constant unless you are also swapping the model
         *               file (e.g. to `face_detection_full_range_sparse.tflite` at 192).
         */
        const val inputSize: Int = 128

        /**
         * Number of anchors emitted by the detector. Hard-coded by the SSD anchor calculator
         * config used to generate [com.alifesoftware.facemesh.ml.BlazeFaceAnchors.FRONT_128].
         *
         *   - Class:    MODEL CONTRACT (anchor-table contract)
         *   - Used by:  `BlazeFaceDetector` output buffer pre-allocation (init path asserts
         *               this matches `BlazeFaceAnchors.FRONT_128.size`).
         *   - Note:     the full-range BlazeFace bundle ships a different anchor count
         *               (typically 2304); a model swap requires a matching anchor table and
         *               a new value here.
         */
        const val numAnchors: Int = 896

        /**
         * Bytes per anchor in the regressor output: 4 box (dx, dy, w, h) + 12 landmark
         * (6 x,y pairs) = 16 floats.
         *
         *   - Class:    MODEL CONTRACT
         *   - Used by:  `BlazeFaceDecoder` regressor unpacking, `BlazeFaceDetector` output
         *               buffer shape.
         */
        const val regStride: Int = 16

        /**
         * Minimum sigmoid score for a candidate anchor to survive the decoder pass.
         *
         *   - Class:    TUNABLE HEURISTIC
         *   - Used by:  `BlazeFaceDecoder.decode`, also re-applied at
         *               [Filters.confidenceThreshold] (kept aligned).
         *   - Range:    0.30..0.95
         *   - Lower:    recovers small / distant / blurry faces (their logits are squashed
         *               into the 0.50..0.70 range); risk of more false positives that
         *               downstream FaceFilters must clean up.
         *   - Higher:   sharper precision but distant faces in group photos disappear.
         *   - Why 0.55: chosen 2026-05-09 to fix group-photo recall after observing real-world
         *               faraway faces peaking at sigmoid 0.55..0.70. Pre-2026-05-09 default
         *               was 0.75.
         */
        const val scoreThreshold: Float = 0.55f

        /**
         * Intersection-over-union threshold above which two candidate boxes are merged in
         * MediaPipe-style weighted NMS.
         *
         *   - Class:    TUNABLE HEURISTIC
         *   - Used by:  `BlazeFaceDecoder.weightedNms`
         *   - Range:    0.20..0.60
         *   - Lower:    fewer duplicates; legitimate side-by-side faces (cheek-to-cheek group
         *               shots, twins) can fuse into one detection.
         *   - Higher:   tolerates near-overlap; risk of stacked duplicates from multiple
         *               anchors firing on the same face surviving as separate detections.
         */
        const val nmsIouThreshold: Float = 0.30f

        /**
         * Worker thread count for the CPU/XNNPACK delegate path. Ignored when the GPU
         * delegate is active.
         *
         *   - Class:    TUNABLE HEURISTIC
         *   - Used by:  `TfLiteRuntime.newInterpreter`
         *   - Range:    1..available cores
         *   - Lower:    less CPU contention with UI (useful on big.LITTLE phones).
         *   - Higher:   faster inference up to the model's parallelism ceiling, more battery.
         */
        const val interpreterThreads: Int = 2
    }

    // ---------------------------------------------------------------------------------------
    // Filters (post-detection false-positive heuristics)
    // ---------------------------------------------------------------------------------------

    /**
     * [com.alifesoftware.facemesh.ml.FaceFilters] post-detection heuristics.
     *
     * These run AFTER NMS and BEFORE alignment / embedding, so they're the cheapest place to
     * cull false positives without spending compute on alignment + embedding.
     */
    object Filters {
        /**
         * Re-applied confidence floor on top of the decoder's [Detector.scoreThreshold]. Kept
         * aligned with the decoder so this stage does not silently undo the decoder's recall
         * recovery.
         *
         *   - Class:    TUNABLE HEURISTIC
         *   - Used by:  `FaceFilters.apply`
         *   - Range:    0.30..0.95 (typically equal to or stricter than [Detector.scoreThreshold])
         */
        const val confidenceThreshold: Float = 0.55f

        /**
         * Lower bound on `eyeDistance / boxWidth`. Real human faces cluster around ~0.40
         * (front-on) and degrade gracefully toward the edges of this band as the face turns
         * away from the camera.
         *
         *   - Class:    TUNABLE HEURISTIC
         *   - Used by:  `FaceFilters.apply` geometry stage
         *   - Range:    0.10..0.40
         *   - Lower:    accepts more profile/turned faces; risk of accepting non-face artefacts
         *               with poor landmark predictions.
         *   - Higher:   stricter front-facing-only filter.
         */
        const val eyeWidthRatioMin: Float = 0.25f

        /**
         * Upper bound on `eyeDistance / boxWidth`.
         *
         *   - Class:    TUNABLE HEURISTIC
         *   - Used by:  `FaceFilters.apply` geometry stage
         *   - Range:    0.50..0.85
         *   - Lower:    stricter; rejects faces where eyes appear unusually far apart relative
         *               to the box (often a sign of a botched bounding box).
         *   - Higher:   tolerates wider-than-typical eye placement.
         */
        const val eyeWidthRatioMax: Float = 0.65f

        /**
         * `+/- sizeBandFraction * medianWidth` is the size band a face must fall within to
         * survive the size-outlier stage.
         *
         *   - Class:    TUNABLE HEURISTIC
         *   - Used by:  `FaceFilters.apply` size-band stage (only when 2 <= n < [groupPhotoSizeBandSkipThreshold])
         *   - Range:    0.30..3.00
         *   - Lower:    aggressively culls outliers; in 2-3-person frames a near subject
         *               (e.g. 200px) and a distant subject (60px) can no longer coexist.
         *   - Higher:   tolerates wider variance at the cost of letting more false-positive
         *               anchors through.
         *   - Why 2.0:  widened from 1.0 -> 2.0 on 2026-05-09 to recover near+far pairs.
         */
        const val sizeBandFraction: Float = 2.0f

        /**
         * When this many faces or more survive the geometry stage, the size-band stage is
         * skipped entirely - the median is no longer a reliable "real face" anchor and we
         * prefer to over-keep here and let DBSCAN sort it out downstream.
         *
         *   - Class:    TUNABLE HEURISTIC
         *   - Used by:  `FaceFilters.apply` size-band stage
         *   - Range:    3..10
         *   - Lower:    skips the stage on smaller groups (more recall on group photos, less
         *               precision overall).
         *   - Higher:   keeps the size-band gate active for larger groups (more precision, but
         *               group photos get pruned again).
         *   - Why 4:    a clean "trio = handle as portrait, quartet+ = group" cutoff.
         */
        const val groupPhotoSizeBandSkipThreshold: Int = 4
    }

    // ---------------------------------------------------------------------------------------
    // Aligner (ArcFace template)
    // ---------------------------------------------------------------------------------------

    /**
     * [com.alifesoftware.facemesh.ml.FaceAligner] settings.
     *
     * This is almost entirely MODEL CONTRACT - the embedder ([Embedder]) was trained on inputs
     * posed exactly to this template. Changing any of these values without retraining the
     * embedder produces a silent embedding-quality regression.
     */
    object Aligner {
        /**
         * Edge size of the affinely-aligned face crop fed into the embedder. Must equal
         * [Embedder.inputSize].
         *
         *   - Class:    MODEL CONTRACT
         *   - Used by:  `FaceAligner.align`, `FaceProcessor.process`
         */
        const val outputSize: Int = 112

        /**
         * The 4-point ArcFace canonical landmark template, normalised to [outputSize]
         * (=112 px). Order is `[rightEyeX, rightEyeY, leftEyeX, leftEyeY, noseX, noseY,
         * mouthCenterX, mouthCenterY]`.
         *
         * Derived from the ArcFace authors' 5-point reference template at 112 x 112 by
         * averaging the original two mouth-corner points into a single mouth-centre point
         * (Android's `Matrix.setPolyToPoly` caps at 4 source/destination points).
         *
         *   - Class:    MODEL CONTRACT (locked to GhostFaceNet training distribution)
         *   - Used by:  `FaceAligner.align`
         *
         * Source: https://insightface.ai/arcface
         *   right_eye    = (38.2946, 51.6963)
         *   left_eye     = (73.5318, 51.5014)
         *   nose         = (56.0252, 71.7366)
         *   mouth_center = mean( (41.5493, 92.3655), (70.7299, 92.2041) ) = (56.1396, 92.2848)
         */
        val canonicalLandmarkTemplate: FloatArray = floatArrayOf(
            38.2946f, 51.6963f,
            73.5318f, 51.5014f,
            56.0252f, 71.7366f,
            56.1396f, 92.2848f,
        )
    }

    // ---------------------------------------------------------------------------------------
    // Embedder (GhostFaceNet-V1 FP16)
    // ---------------------------------------------------------------------------------------

    /**
     * [com.alifesoftware.facemesh.ml.FaceEmbedder] (GhostFaceNet-V1 FP16) shape contract.
     *
     * Confirmed against the FP32 ONNX reference at
     * https://github.com/alifesoftware/ModelZoo/blob/master/GhostFaceNet/Model/ghostface_fp32.onnx
     */
    object Embedder {
        /**
         * Edge size of the embedder input bitmap. Must equal [Aligner.outputSize].
         *
         *   - Class:    MODEL CONTRACT (GhostFaceNet ships at 112)
         *   - Used by:  `FaceEmbedder` input buffer + dimension assertion
         */
        const val inputSize: Int = 112

        /**
         * Length of the embedding vector emitted per face.
         *
         *   - Class:    MODEL CONTRACT (GhostFaceNet-V1 emits 512-d vectors)
         *   - Used by:  `FaceEmbedder` output buffer; downstream cosine math in
         *               `EmbeddingMath`, `Dbscan`, `FilterAgainstClustersUseCase`.
         *   - Note:     the original SPEC referenced 128-d MobileFaceNet; one stale comment
         *               in `ClusterEntity` still says "128-d centroid". Both the runtime and
         *               the persisted DB blobs are 512 floats.
         */
        const val embeddingDim: Int = 512
    }

    // ---------------------------------------------------------------------------------------
    // Display crop (cluster avatar thumbnail)
    // ---------------------------------------------------------------------------------------

    /**
     * `FaceProcessor.cropDisplayThumbnail` knobs - the natural-pose square crop used as the
     * cluster avatar (distinct from the affine-aligned embedder input).
     */
    object DisplayCrop {
        /**
         * Padding added on EACH side of the face bounding box, expressed as a fraction of the
         * bbox's max side. Final crop side = `bboxMaxSide * (1 + 2 * paddingFraction)`.
         *
         *   - Class:    TUNABLE HEURISTIC
         *   - Used by:  `FaceProcessor.cropDisplayThumbnail`
         *   - Range:    0.10..0.60
         *   - Lower:    tighter avatar, mostly face. Loses contextual hair/shoulders that help
         *               humans distinguish similar people.
         *   - Higher:   roomier avatar with more context, but the face becomes smaller within
         *               the thumbnail.
         */
        const val paddingFraction: Float = 0.30f

        /**
         * Cap on the largest output dimension of the saved display crop (PNG on disk).
         *
         *   - Class:    TUNABLE HEURISTIC
         *   - Used by:  `FaceProcessor.cropDisplayThumbnail`
         *   - Range:    96..512
         *   - Lower:    smaller PNG files, but visible aliasing on high-DPI cluster grid cells.
         *   - Higher:   crisper avatars, more storage per cluster (~1-50 KB scales as N^2).
         */
        const val maxOutputDim: Int = 256
    }

    // ---------------------------------------------------------------------------------------
    // Clustering (DBSCAN)
    // ---------------------------------------------------------------------------------------

    /**
     * [com.alifesoftware.facemesh.ml.cluster.Dbscan] hyperparameters.
     */
    object Clustering {
        /**
         * Default maximum cosine distance (= 1 - dot for L2-normed vectors) for two
         * embeddings to be considered DBSCAN neighbours.
         *
         *   - Class:    USER SETTING DEFAULT (overridden at runtime via
         *               `AppPreferences.dbscanEps` / downloaded `manifest.json`)
         *   - Used by:  `Dbscan.regionQuery`
         *   - Range:    [minEps]..[maxEps]
         *   - Lower:    tighter clusters, more "same person" splits across pose / lighting /
         *               expression; safer when face library is small and clean.
         *   - Higher:   looser clusters, more "different people" merges. Useful when faces
         *               are noisier (small / blurry / aged-out).
         *   - Why 0.35: empirical sweet spot on GhostFaceNet 512-d L2-normed embeddings.
         */
        const val defaultEps: Float = 0.35f
        const val minEps: Float = 0.10f
        const val maxEps: Float = 0.80f

        /**
         * Default minimum neighbour count (including self) for a point to qualify as a DBSCAN
         * core point.
         *
         *   - Class:    USER SETTING DEFAULT (overridden at runtime via
         *               `AppPreferences.dbscanMinPts`)
         *   - Used by:  `Dbscan.run`
         *   - Range:    [minMinPts]..[maxMinPts]
         *   - Lower:    minPts=1 reduces DBSCAN to "every face is its own cluster". minPts=2
         *               is the lowest useful value: any two faces above [defaultEps] cosine
         *               similarity form a cluster.
         *   - Higher:   harder to start a cluster -> single-photo people get marked as noise
         *               and dropped. Useful when the input is dominated by strangers.
         *   - Why 2:    a single duplicate of a person is enough to surface that person as a
         *               cluster - matches the small-library assumption of the v1 UX.
         */
        const val defaultMinPts: Int = 2
        const val minMinPts: Int = 1
        const val maxMinPts: Int = 10

        /**
         * Sentinel labels the algorithm uses internally. Not user-tunable - exposed here so
         * call sites have a single source of truth.
         *
         *   - Class:    ALGORITHM CONSTANT
         */
        const val unvisitedLabel: Int = -2
        const val noiseLabel: Int = -1
    }

    // ---------------------------------------------------------------------------------------
    // Match (filter phase: per-image vs cluster centroids)
    // ---------------------------------------------------------------------------------------

    /**
     * [com.alifesoftware.facemesh.domain.FilterAgainstClustersUseCase] thresholds.
     */
    object Match {
        /**
         * Default minimum cosine similarity (= dot product for L2-normed vectors) for a
         * candidate image's face to count as a match against a selected cluster's centroid.
         * An image is a Keeper iff at least one of its faces beats this against any selected
         * centroid.
         *
         *   - Class:    USER SETTING DEFAULT (overridden at runtime via
         *               `AppPreferences.matchThreshold`)
         *   - Used by:  `FilterAgainstClustersUseCase.run`
         *   - Range:    [minThreshold]..[maxThreshold]
         *   - Lower:    more keepers (recall up, precision down). Same-person faces that
         *               diverged in pose/lighting from the centroid get included.
         *   - Higher:   fewer keepers (precision up, recall down). Strangers that look
         *               passingly similar get rejected.
         *   - Why 0.65: GhostFaceNet 512-d "same person" similarity typically lives in
         *               0.55..0.85; 0.65 picks up most pose variance without leaking
         *               look-alikes.
         */
        const val defaultThreshold: Float = 0.65f
        const val minThreshold: Float = 0.30f
        const val maxThreshold: Float = 0.95f
    }
}
