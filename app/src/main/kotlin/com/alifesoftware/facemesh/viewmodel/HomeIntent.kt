package com.alifesoftware.facemesh.viewmodel

import android.net.Uri

/** User- and pipeline-driven events the [HomeViewModel] reduces into [HomeUiState] transitions. */
sealed interface HomeIntent {

    // \u2500\u2500 user intents \u2500\u2500
    data object AddPhotosTapped : HomeIntent
    data class PhotosPicked(val uris: List<Uri>) : HomeIntent
    data object ClusterifyTapped : HomeIntent
    data class ToggleClusterChecked(val clusterId: String) : HomeIntent
    data object CameraTapped : HomeIntent
    data class FilterPhotosPicked(val uris: List<Uri>) : HomeIntent
    data object FilterTapped : HomeIntent
    data object ClearFilterTapped : HomeIntent
    data object ResetConfirmed : HomeIntent
    data class DeleteClusterConfirmed(val clusterId: String) : HomeIntent
    data object CancelProcessingTapped : HomeIntent
    data object ReturnedFromKeepers : HomeIntent

    // \u2500\u2500 pipeline events (emitted by use-case side-effects, see Phase 5/6) \u2500\u2500
    data class ClusterifyProgress(val processed: Int, val total: Int) : HomeIntent
    data class ClusterifyFinished(val clusters: List<com.alifesoftware.facemesh.domain.model.Cluster>) : HomeIntent
    data object ClusterifyNoFaces : HomeIntent
    data class FilterProgress(val processed: Int, val total: Int) : HomeIntent
    data class FilterFinished(val keepers: List<Uri>) : HomeIntent
    data object FilterNoMatches : HomeIntent
}
