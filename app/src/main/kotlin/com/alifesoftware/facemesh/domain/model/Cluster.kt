package com.alifesoftware.facemesh.domain.model

import android.net.Uri

/**
 * A face cluster as exposed to the UI / domain layers.
 *
 * Persistence-layer counterpart lives in [com.alifesoftware.facemesh.data.Cluster] (added in
 * Phase 2). Centroid is intentionally not exposed here \u2014 the UI only needs a stable id, the
 * representative thumbnail, the face count, and an optional name for v1.1.
 */
data class Cluster(
    val id: String,
    val representativeImageUri: Uri,
    val faceCount: Int,
    val name: String? = null,
)
