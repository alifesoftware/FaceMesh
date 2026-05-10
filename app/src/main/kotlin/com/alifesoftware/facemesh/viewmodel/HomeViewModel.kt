package com.alifesoftware.facemesh.viewmodel

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.alifesoftware.facemesh.data.AppPreferences
import com.alifesoftware.facemesh.data.ClusterRepository
import com.alifesoftware.facemesh.domain.ClusterifyUseCase
import com.alifesoftware.facemesh.domain.FilterAgainstClustersUseCase
import com.alifesoftware.facemesh.domain.model.Cluster
import com.alifesoftware.facemesh.ml.MlPipelineProvider
import com.alifesoftware.facemesh.ml.TfLiteRuntime
import com.alifesoftware.facemesh.ml.download.ModelDownloadManager
import com.alifesoftware.facemesh.mock.MockData
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.UUID

/**
 * Single state-machine ViewModel for the home screen (SPEC \u00a75.3).
 *
 * Phase 1 implementation: state transitions are real, ML side-effects are mocked with delays.
 * Phase 2 will replace mock pickers with the system Photo Picker; Phase 5 / 6 will swap the
 * mocked progress for real Use-Case calls.
 */
class HomeViewModel(
    private val clusterRepository: ClusterRepository? = null,
    private val preferences: AppPreferences? = null,
    private val modelDownloader: ModelDownloadManager? = null,
    private val mlPipelineProvider: MlPipelineProvider? = null,
) : ViewModel() {

    private val _state: MutableStateFlow<HomeUiState> = MutableStateFlow(HomeUiState.Empty)
    val state: StateFlow<HomeUiState> = _state.asStateFlow()

    init {
        // Surface persisted clusters as the initial state on cold start (SPEC FR-13 persistence).
        clusterRepository?.let { repo ->
            viewModelScope.launch {
                val saved = repo.loadClusters()
                if (saved.isNotEmpty() && _state.value is HomeUiState.Empty) {
                    _state.value = HomeUiState.Clustered(
                        clusters = saved,
                        selectedClusterIds = emptySet(),
                    )
                }
            }
        }
    }

    private val _messages: MutableSharedFlow<UiMessage> = MutableSharedFlow(extraBufferCapacity = 4)
    val messages: SharedFlow<UiMessage> = _messages.asSharedFlow()

    private val _keepers: MutableStateFlow<Map<String, List<Uri>>> = MutableStateFlow(emptyMap())
    val keepersBySession: StateFlow<Map<String, List<Uri>>> = _keepers.asStateFlow()

    private var pipelineJob: Job? = null

    fun handle(intent: HomeIntent) {
        when (intent) {
            HomeIntent.AddPhotosTapped -> { /* handled by UI launching the picker */ }
            is HomeIntent.PhotosPicked -> onPhotosPicked(intent.uris)
            HomeIntent.ClusterifyTapped -> onClusterifyTapped()
            is HomeIntent.ToggleClusterChecked -> onToggleCluster(intent.clusterId)
            HomeIntent.CameraTapped -> { /* handled by UI launching picker (max 15) */ }
            is HomeIntent.FilterPhotosPicked -> onFilterPhotosPicked(intent.uris)
            HomeIntent.FilterTapped -> onFilterTapped()
            HomeIntent.ClearFilterTapped -> onClearFilter()
            HomeIntent.ResetConfirmed -> onReset()
            is HomeIntent.DeleteClusterConfirmed -> onDeleteCluster(intent.clusterId)
            HomeIntent.CancelProcessingTapped -> onCancelProcessing()
            HomeIntent.ReturnedFromKeepers -> onReturnedFromKeepers()
            is HomeIntent.ClusterifyProgress -> updateProcessing(intent.processed, intent.total)
            is HomeIntent.ClusterifyFinished -> finishClusterify(intent.clusters)
            HomeIntent.ClusterifyNoFaces -> handleNoFaces()
            is HomeIntent.FilterProgress -> updateMatching(intent.processed, intent.total)
            is HomeIntent.FilterFinished -> finishMatching(intent.keepers)
            HomeIntent.FilterNoMatches -> handleNoMatches()
        }
    }

    // \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 transitions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    private fun onPhotosPicked(uris: List<Uri>) {
        if (uris.isEmpty()) return
        _state.update { current ->
            when (current) {
                is HomeUiState.Empty -> HomeUiState.Selecting(
                    selectedPhotos = uris,
                    recentFan = uris.takeLast(MAX_FAN).reversed(),
                )
                is HomeUiState.Selecting -> {
                    val merged = current.selectedPhotos + uris
                    HomeUiState.Selecting(
                        selectedPhotos = merged,
                        recentFan = merged.takeLast(MAX_FAN).reversed(),
                    )
                }
                is HomeUiState.Clustered -> HomeUiState.Selecting(
                    selectedPhotos = uris,
                    recentFan = uris.takeLast(MAX_FAN).reversed(),
                )
                else -> current
            }
        }
    }

    private suspend fun maybeShowGpuFallbackToastOnce() {
        val provider = mlPipelineProvider ?: return
        val prefs = preferences ?: return
        if (provider.activeDelegate == TfLiteRuntime.Delegate.XNNPACK) {
            val alreadyShown = prefs.gpuFallbackToastShown.first()
            if (!alreadyShown) {
                _messages.emit(UiMessage.UsingCpuAcceleration)
                prefs.setGpuFallbackToastShown(true)
            }
        }
    }

    private fun onClusterifyTapped() {
        val current = _state.value as? HomeUiState.Selecting ?: return
        val total = current.selectedPhotos.size
        _state.value = HomeUiState.Processing(processed = 0, total = total, recentFan = current.recentFan)

        if (modelDownloader == null || mlPipelineProvider == null) {
            // Tests / Phase-1 mock path
            runMockClusterify(current.selectedPhotos, total)
            return
        }

        pipelineJob?.cancel()
        pipelineJob = viewModelScope.launch {
            ensureModelsAvailable() ?: return@launch
            val useCase = mlPipelineProvider.clusterifyUseCase()
            maybeShowGpuFallbackToastOnce()
            useCase.run(current.selectedPhotos).collect { event ->
                when (event) {
                    is ClusterifyUseCase.Event.Progress ->
                        handle(HomeIntent.ClusterifyProgress(event.processed, event.total))
                    ClusterifyUseCase.Event.NoFaces ->
                        handle(HomeIntent.ClusterifyNoFaces)
                    is ClusterifyUseCase.Event.Done ->
                        handle(HomeIntent.ClusterifyFinished(event.clusters))
                }
            }
        }
    }

    /**
     * Drives the model download (SPEC \u00a78.2). Returns null on hard failure (we already pushed an
     * Empty state + Snackbar in that case).
     */
    private suspend fun ensureModelsAvailable(): Boolean? {
        val downloader = modelDownloader ?: return true
        var ok = false
        downloader.ensureAvailable().collect { progress ->
            when (progress) {
                is ModelDownloadManager.Progress.AlreadyAvailable -> {
                    ok = true
                }
                is ModelDownloadManager.Progress.Started -> {
                    _state.update { s -> if (s is HomeUiState.Processing) s.copy(downloadFraction = 0f) else s }
                }
                is ModelDownloadManager.Progress.Downloading -> {
                    _state.update { s ->
                        if (s is HomeUiState.Processing) s.copy(downloadFraction = progress.approxFraction) else s
                    }
                }
                is ModelDownloadManager.Progress.Done -> {
                    ok = true
                    _state.update { s -> if (s is HomeUiState.Processing) s.copy(downloadFraction = null) else s }
                }
                is ModelDownloadManager.Progress.Failed -> {
                    ok = false
                }
            }
        }
        if (!ok) {
            _messages.emit(UiMessage.ModelDownloadFailed)
            _state.value = HomeUiState.Empty
            return null
        }
        return true
    }

    private fun runMockClusterify(uris: List<Uri>, total: Int) {
        pipelineJob?.cancel()
        pipelineJob = viewModelScope.launch {
            uris.forEachIndexed { index, _ ->
                delay(120)
                handle(HomeIntent.ClusterifyProgress(processed = index + 1, total = total))
            }
            delay(150)
            handle(HomeIntent.ClusterifyFinished(MockData.mockClusters))
        }
    }

    private fun updateProcessing(processed: Int, total: Int) {
        _state.update { s ->
            when (s) {
                is HomeUiState.Processing -> s.copy(processed = processed, total = total)
                is HomeUiState.Matching -> s
                else -> s
            }
        }
    }

    private fun finishClusterify(clusters: List<Cluster>) {
        _state.value = HomeUiState.Clustered(
            clusters = clusters,
            selectedClusterIds = emptySet(),
        )
    }

    private fun handleNoFaces() {
        viewModelScope.launch { _messages.emit(UiMessage.NoFacesFound) }
        _state.value = HomeUiState.Empty
    }

    private fun onToggleCluster(id: String) {
        _state.update { s ->
            when (s) {
                is HomeUiState.Clustered -> s.copy(
                    selectedClusterIds = s.selectedClusterIds.toggle(id),
                )
                is HomeUiState.FilterReady -> s.copy(
                    selectedClusterIds = s.selectedClusterIds.toggle(id),
                )
                else -> s
            }
        }
    }

    private fun onFilterPhotosPicked(uris: List<Uri>) {
        if (uris.isEmpty()) return
        val capped = uris.take(MAX_FILTER_PHOTOS)
        if (uris.size > MAX_FILTER_PHOTOS) {
            viewModelScope.launch { _messages.emit(UiMessage.MaxFilterImagesReached) }
        }
        _state.update { s ->
            when (s) {
                is HomeUiState.Clustered -> HomeUiState.FilterReady(
                    clusters = s.clusters,
                    selectedClusterIds = s.selectedClusterIds,
                    filterPhotos = capped,
                )
                is HomeUiState.FilterReady -> s.copy(filterPhotos = capped)
                else -> s
            }
        }
    }

    private fun onFilterTapped() {
        val current = _state.value as? HomeUiState.FilterReady ?: return
        if (!current.canFilter) return
        val total = current.filterPhotos.size
        _state.value = HomeUiState.Matching(
            processed = 0,
            total = total,
            clusters = current.clusters,
            selectedClusterIds = current.selectedClusterIds,
            filterPhotos = current.filterPhotos,
        )

        if (mlPipelineProvider == null) {
            // Tests / Phase-1 mock path
            runMockMatching(total)
            return
        }

        pipelineJob?.cancel()
        pipelineJob = viewModelScope.launch {
            val useCase = mlPipelineProvider.filterUseCase()
            useCase
                .run(current.filterPhotos, current.selectedClusterIds)
                .collect { event ->
                    when (event) {
                        is FilterAgainstClustersUseCase.Event.Progress ->
                            handle(HomeIntent.FilterProgress(event.processed, event.total))
                        is FilterAgainstClustersUseCase.Event.Done ->
                            if (event.keepers.isEmpty()) handle(HomeIntent.FilterNoMatches)
                            else handle(HomeIntent.FilterFinished(event.keepers))
                    }
                }
        }
    }

    private fun runMockMatching(total: Int) {
        pipelineJob?.cancel()
        pipelineJob = viewModelScope.launch {
            (1..total).forEach { i ->
                delay(80)
                handle(HomeIntent.FilterProgress(processed = i, total = total))
            }
            delay(120)
            handle(HomeIntent.FilterFinished(MockData.mockKeepers))
        }
    }

    private fun updateMatching(processed: Int, total: Int) {
        _state.update { s ->
            if (s is HomeUiState.Matching) s.copy(processed = processed, total = total) else s
        }
    }

    private fun finishMatching(keepers: List<Uri>) {
        if (keepers.isEmpty()) {
            handleNoMatches()
            return
        }
        val current = _state.value as? HomeUiState.Matching ?: return
        val sessionId = UUID.randomUUID().toString()
        _keepers.update { it + (sessionId to keepers) }
        _state.value = HomeUiState.Clustered(
            clusters = current.clusters,
            selectedClusterIds = current.selectedClusterIds,
        )
        viewModelScope.launch { _messages.emit(UiMessage.NavigateToKeepers(sessionId)) }
    }

    private fun handleNoMatches() {
        val current = _state.value as? HomeUiState.Matching ?: return
        _state.value = HomeUiState.Clustered(
            clusters = current.clusters,
            selectedClusterIds = current.selectedClusterIds,
        )
        viewModelScope.launch { _messages.emit(UiMessage.NoMatchesFound) }
    }

    private fun onClearFilter() {
        val current = _state.value as? HomeUiState.FilterReady ?: return
        _state.value = HomeUiState.Clustered(
            clusters = current.clusters,
            selectedClusterIds = current.selectedClusterIds,
        )
    }

    private fun onReset() {
        pipelineJob?.cancel()
        _keepers.value = emptyMap()
        _state.value = HomeUiState.Empty
        viewModelScope.launch {
            // Best-effort wipe; if either is null (e.g. unit tests) just skip.
            clusterRepository?.deleteAll()
            preferences?.clearAll()
        }
    }

    private fun onDeleteCluster(id: String) {
        _state.update { s ->
            when (s) {
                is HomeUiState.Clustered -> {
                    val remaining = s.clusters.filterNot { it.id == id }
                    if (remaining.isEmpty()) HomeUiState.Empty
                    else s.copy(
                        clusters = remaining,
                        selectedClusterIds = s.selectedClusterIds - id,
                    )
                }
                is HomeUiState.FilterReady -> {
                    val remaining = s.clusters.filterNot { it.id == id }
                    when {
                        remaining.isEmpty() -> HomeUiState.Empty
                        else -> s.copy(
                            clusters = remaining,
                            selectedClusterIds = s.selectedClusterIds - id,
                        )
                    }
                }
                else -> s
            }
        }
        viewModelScope.launch { clusterRepository?.deleteCluster(id) }
    }

    private fun onCancelProcessing() {
        pipelineJob?.cancel()
        _state.update { s ->
            when (s) {
                is HomeUiState.Processing -> HomeUiState.Empty
                is HomeUiState.Matching -> HomeUiState.Clustered(
                    clusters = s.clusters,
                    selectedClusterIds = s.selectedClusterIds,
                )
                else -> s
            }
        }
    }

    private fun onReturnedFromKeepers() {
        // No-op: state is preserved as-is. Hook reserved for future analytics.
    }

    private fun Set<String>.toggle(id: String): Set<String> =
        if (contains(id)) this - id else this + id

    companion object {
        const val MAX_FAN: Int = 4
        const val MAX_FILTER_PHOTOS: Int = 15
    }
}
