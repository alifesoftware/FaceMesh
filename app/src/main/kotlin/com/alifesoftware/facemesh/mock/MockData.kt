package com.alifesoftware.facemesh.mock

import android.net.Uri
import com.alifesoftware.facemesh.domain.model.Cluster

/**
 * Deterministic fixtures for Phase 1 (UI shell only) and previews.
 *
 * Replaced in Phase 5 by the real [com.alifesoftware.facemesh.domain.ClusterifyUseCase] output.
 */
object MockData {

    private fun placeholder(seed: Int): Uri =
        Uri.parse("https://picsum.photos/seed/facemesh-$seed/400/400")

    val photosBatchOne: List<Uri> = (1..7).map { placeholder(it) }
    val photosBatchTwo: List<Uri> = (8..14).map { placeholder(it) }

    val mockClusters: List<Cluster> = listOf(
        Cluster(id = "c1", representativeImageUri = placeholder(101), faceCount = 12),
        Cluster(id = "c2", representativeImageUri = placeholder(102), faceCount = 7),
        Cluster(id = "c3", representativeImageUri = placeholder(103), faceCount = 4),
        Cluster(id = "c4", representativeImageUri = placeholder(104), faceCount = 2),
    )

    val mockKeepers: List<Uri> = (200..207).map { placeholder(it) }
}
