package com.alifesoftware.facemesh.viewmodel

import android.net.Uri
import com.alifesoftware.facemesh.domain.model.Cluster

/**
 * Single source of truth for the home screen. Each variant maps 1:1 to a SPEC \u00a74.2 state.
 *
 * Invariants preserved by [HomeViewModel]:
 *   \u2022 `selectedPhotos` is empty in [Empty]/[Clustered].
 *   \u2022 `clusters` is empty in [Empty]/[Selecting].
 *   \u2022 `filterPhotos` is empty unless we're in [FilterReady] or transitioning to [Matching].
 */
sealed interface HomeUiState {

    /** Latest 0..4 photo URIs to render in the thumbnail fan, newest first. */
    val recentFan: List<Uri>

    /** SPEC \u00a74.2.1 \u2014 first launch or post-Reset / no clusters. */
    data object Empty : HomeUiState {
        override val recentFan: List<Uri> = emptyList()
    }

    /** SPEC \u00a74.2.2 \u2014 user picked photos but hasn't tapped Clusterify yet. */
    data class Selecting(
        val selectedPhotos: List<Uri>,
        override val recentFan: List<Uri>,
    ) : HomeUiState

    /** SPEC \u00a74.2.3 \u2014 Clusterify pipeline is in flight. */
    data class Processing(
        val processed: Int,
        val total: Int,
        override val recentFan: List<Uri>,
        /** Optional caption override for the model-download sub-phase (SPEC \u00a78.2 step 3). */
        val downloadFraction: Float? = null,
    ) : HomeUiState

    /** SPEC \u00a74.2.4 \u2014 we have at least one cluster, no filter photos selected yet. */
    data class Clustered(
        val clusters: List<Cluster>,
        val selectedClusterIds: Set<String>,
        override val recentFan: List<Uri> = emptyList(),
    ) : HomeUiState {
        val hasAnySelected: Boolean get() = selectedClusterIds.isNotEmpty()
    }

    /** SPEC \u00a74.2.5 \u2014 user picked filter photos; ready to run Filter. */
    data class FilterReady(
        val clusters: List<Cluster>,
        val selectedClusterIds: Set<String>,
        val filterPhotos: List<Uri>,
        override val recentFan: List<Uri> = emptyList(),
    ) : HomeUiState {
        val canFilter: Boolean get() = filterPhotos.isNotEmpty() && selectedClusterIds.isNotEmpty()
    }

    /** SPEC \u00a74.2.6 \u2014 Filter pipeline is in flight. */
    data class Matching(
        val processed: Int,
        val total: Int,
        val clusters: List<Cluster>,
        val selectedClusterIds: Set<String>,
        val filterPhotos: List<Uri>,
        override val recentFan: List<Uri> = emptyList(),
    ) : HomeUiState
}

/** Transient, one-shot UI messages (snackbars/toasts/dialogs) the screen should consume. */
sealed interface UiMessage {
    data object NoFacesFound : UiMessage
    data object NoMatchesFound : UiMessage
    data object UsingCpuAcceleration : UiMessage
    data object MaxFilterImagesReached : UiMessage
    data object ModelDownloadFailed : UiMessage
    data class NavigateToKeepers(val sessionId: String) : UiMessage
}
