package com.alifesoftware.facemesh.viewmodel

import android.net.Uri
import android.os.SystemClock
import android.util.Log
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
        Log.i(
            TAG,
            "init: VM created repo=${clusterRepository != null} prefs=${preferences != null} " +
                "downloader=${modelDownloader != null} pipeline=${mlPipelineProvider != null} " +
                "initialState=${_state.value::class.simpleName}",
        )
        // Surface persisted clusters as the initial state on cold start (SPEC FR-13 persistence).
        clusterRepository?.let { repo ->
            viewModelScope.launch {
                Log.i(TAG, "init: loading persisted clusters from repository...")
                val saved = repo.loadClusters()
                Log.i(TAG, "init: repo returned ${saved.size} cluster(s)")
                if (saved.isNotEmpty() && _state.value is HomeUiState.Empty) {
                    transition("init/restorePersistedClusters", HomeUiState.Clustered(
                        clusters = saved,
                        selectedClusterIds = emptySet(),
                    ))
                } else if (saved.isEmpty()) {
                    Log.i(TAG, "init: no persisted clusters; staying in Empty")
                }
            }
        }
    }

    private val _messages: MutableSharedFlow<UiMessage> = MutableSharedFlow(extraBufferCapacity = 4)
    val messages: SharedFlow<UiMessage> = _messages.asSharedFlow()

    private val _keepers: MutableStateFlow<Map<String, List<Uri>>> = MutableStateFlow(emptyMap())
    val keepersBySession: StateFlow<Map<String, List<Uri>>> = _keepers.asStateFlow()

    /**
     * Per-cluster source-photo URI cache populated lazily when the user opens a cluster's
     * gallery via [loadClusterImages]. Keyed by cluster id; value is the full ordered list
     * of distinct source photos that contributed faces. Survives navigation (so back-and-
     * forth between Home and ClusterGallery doesn't re-hit the DB) but is dropped on
     * `clearAll` / Reset.
     */
    private val _clusterImages: MutableStateFlow<Map<String, List<Uri>>> = MutableStateFlow(emptyMap())
    val clusterImagesById: StateFlow<Map<String, List<Uri>>> = _clusterImages.asStateFlow()

    /**
     * Loads the source photos for [clusterId] into [clusterImagesById]. Idempotent: skips
     * the DB hit when the cache already has an entry. Logs counts for the trace.
     */
    fun loadClusterImages(clusterId: String) {
        val repo = clusterRepository ?: run {
            Log.w(TAG, "loadClusterImages: no repository wired; clusterId=$clusterId no-op")
            return
        }
        if (_clusterImages.value.containsKey(clusterId)) {
            Log.i(TAG, "loadClusterImages: id=$clusterId cache hit, skipping DB read")
            return
        }
        viewModelScope.launch {
            Log.i(TAG, "loadClusterImages: id=$clusterId DB read starting")
            val uris = repo.loadImagesForCluster(clusterId)
            _clusterImages.update { it + (clusterId to uris) }
            Log.i(TAG, "loadClusterImages: id=$clusterId loaded ${uris.size} photo(s) into cache")
        }
    }

    private var pipelineJob: Job? = null

    fun handle(intent: HomeIntent) {
        Log.i(TAG, "handle: intent=${describe(intent)} curState=${describe(_state.value)}")
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

    /** Centralised state setter so every transition gets logged with the [reason] tag. */
    private fun transition(reason: String, next: HomeUiState) {
        val prev = _state.value
        if (prev === next || prev == next) {
            Log.i(TAG, "transition[$reason]: no-op (state unchanged) ${describe(prev)}")
            return
        }
        Log.i(TAG, "transition[$reason]: ${describe(prev)} -> ${describe(next)}")
        _state.value = next
    }

    /** Same idea, but for in-place updates that may or may not actually change anything. */
    private fun updateState(reason: String, block: (HomeUiState) -> HomeUiState) {
        val prev = _state.value
        val next = block(prev)
        if (prev === next || prev == next) {
            Log.i(TAG, "updateState[$reason]: no-op (state unchanged) ${describe(prev)}")
            return
        }
        Log.i(TAG, "updateState[$reason]: ${describe(prev)} -> ${describe(next)}")
        _state.value = next
    }

    private suspend fun emitMessage(reason: String, message: UiMessage) {
        Log.i(TAG, "emitMessage[$reason]: $message")
        _messages.emit(message)
    }

    // \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 transitions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    private fun onPhotosPicked(uris: List<Uri>) {
        if (uris.isEmpty()) {
            Log.i(TAG, "onPhotosPicked: empty selection -> ignore")
            return
        }
        Log.i(TAG, "onPhotosPicked: received ${uris.size} uri(s) first=${uris.firstOrNull()}")
        updateState("PhotosPicked") { current ->
            when (current) {
                is HomeUiState.Empty -> HomeUiState.Selecting(
                    selectedPhotos = uris,
                    recentFan = uris.takeLast(MAX_FAN).reversed(),
                )
                is HomeUiState.Selecting -> {
                    val merged = current.selectedPhotos + uris
                    Log.i(
                        TAG,
                        "onPhotosPicked: merging into existing Selecting " +
                            "${current.selectedPhotos.size}+${uris.size}=${merged.size}",
                    )
                    HomeUiState.Selecting(
                        selectedPhotos = merged,
                        recentFan = merged.takeLast(MAX_FAN).reversed(),
                    )
                }
                is HomeUiState.Clustered -> {
                    Log.i(TAG, "onPhotosPicked: leaving Clustered, replacing selection with new batch")
                    HomeUiState.Selecting(
                        selectedPhotos = uris,
                        recentFan = uris.takeLast(MAX_FAN).reversed(),
                    )
                }
                else -> {
                    Log.w(TAG, "onPhotosPicked: ignored in unexpected state ${describe(current)}")
                    current
                }
            }
        }
    }

    private suspend fun maybeShowGpuFallbackToastOnce() {
        val provider = mlPipelineProvider ?: return
        val prefs = preferences ?: return
        Log.i(TAG, "maybeShowGpuFallbackToastOnce: activeDelegate=${provider.activeDelegate}")
        if (provider.activeDelegate == TfLiteRuntime.Delegate.XNNPACK) {
            val alreadyShown = prefs.gpuFallbackToastShown.first()
            Log.i(TAG, "maybeShowGpuFallbackToastOnce: XNNPACK fallback active alreadyShown=$alreadyShown")
            if (!alreadyShown) {
                emitMessage("gpuFallbackToast", UiMessage.UsingCpuAcceleration)
                prefs.setGpuFallbackToastShown(true)
            }
        }
    }

    private fun onClusterifyTapped() {
        val current = _state.value as? HomeUiState.Selecting ?: run {
            Log.w(TAG, "onClusterifyTapped: ignored, state is ${describe(_state.value)} (need Selecting)")
            return
        }
        val total = current.selectedPhotos.size
        Log.i(TAG, "onClusterifyTapped: starting Clusterify on $total photo(s)")
        transition("Clusterify/start", HomeUiState.Processing(processed = 0, total = total, recentFan = current.recentFan))

        if (modelDownloader == null || mlPipelineProvider == null) {
            Log.i(TAG, "onClusterifyTapped: downloader/pipeline null -> running MOCK path")
            runMockClusterify(current.selectedPhotos, total)
            return
        }

        pipelineJob?.cancel()
        Log.i(TAG, "onClusterifyTapped: launching real pipeline coroutine (cancelled any prior job)")
        val launchedAt = SystemClock.elapsedRealtime()
        pipelineJob = viewModelScope.launch {
            ensureModelsAvailable() ?: return@launch
            Log.i(
                TAG,
                "onClusterifyTapped: models OK; building ClusterifyUseCase " +
                    "(${SystemClock.elapsedRealtime() - launchedAt}ms after launch)",
            )
            val useCase = mlPipelineProvider.clusterifyUseCase()
            maybeShowGpuFallbackToastOnce()
            Log.i(TAG, "onClusterifyTapped: starting useCase.run flow collection on ${current.selectedPhotos.size} uri(s)")
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
            Log.i(
                TAG,
                "onClusterifyTapped: useCase flow completed; total wallTime=" +
                    "${SystemClock.elapsedRealtime() - launchedAt}ms",
            )
        }
    }

    /**
     * Drives the model download (SPEC \u00a78.2). Returns null on hard failure (we already pushed an
     * Empty state + Snackbar in that case).
     */
    private suspend fun ensureModelsAvailable(): Boolean? {
        val downloader = modelDownloader ?: run {
            Log.i(TAG, "ensureModelsAvailable: downloader null -> assuming OK (test path)")
            return true
        }
        Log.i(TAG, "ensureModelsAvailable: starting downloader.ensureAvailable()")
        val started = SystemClock.elapsedRealtime()
        var ok = false
        downloader.ensureAvailable().collect { progress ->
            when (progress) {
                is ModelDownloadManager.Progress.AlreadyAvailable -> {
                    Log.i(TAG, "ensureModelsAvailable: AlreadyAvailable v${progress.manifest.version}")
                    ok = true
                }
                is ModelDownloadManager.Progress.Started -> {
                    Log.i(TAG, "ensureModelsAvailable: Started (totalBytes=${progress.totalBytes})")
                    updateState("Download/started") { s ->
                        if (s is HomeUiState.Processing) s.copy(downloadFraction = 0f) else s
                    }
                }
                is ModelDownloadManager.Progress.Downloading -> {
                    // Don't log every byte — the manager already logs per-model start/end.
                    updateState("Download/progress") { s ->
                        if (s is HomeUiState.Processing) s.copy(downloadFraction = progress.approxFraction) else s
                    }
                }
                is ModelDownloadManager.Progress.Done -> {
                    Log.i(TAG, "ensureModelsAvailable: Done v${progress.manifest.version}")
                    ok = true
                    updateState("Download/done") { s ->
                        if (s is HomeUiState.Processing) s.copy(downloadFraction = null) else s
                    }
                }
                is ModelDownloadManager.Progress.Failed -> {
                    Log.e(TAG, "ensureModelsAvailable: Failed target=${progress.target}")
                    ok = false
                }
            }
        }
        Log.i(
            TAG,
            "ensureModelsAvailable: collection done ok=$ok in ${SystemClock.elapsedRealtime() - started}ms",
        )
        if (!ok) {
            emitMessage("Download/failed", UiMessage.ModelDownloadFailed)
            transition("Download/failed", HomeUiState.Empty)
            return null
        }
        return true
    }

    private fun runMockClusterify(uris: List<Uri>, total: Int) {
        Log.i(TAG, "runMockClusterify: total=$total")
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
        // Suppress per-image transition log; just update silently. Pipeline emits its own per-uri lines.
        _state.update { s ->
            when (s) {
                is HomeUiState.Processing -> s.copy(processed = processed, total = total)
                is HomeUiState.Matching -> s
                else -> s
            }
        }
    }

    private fun finishClusterify(clusters: List<Cluster>) {
        Log.i(TAG, "finishClusterify: producing Clustered with ${clusters.size} cluster(s)")
        transition("Clusterify/finished", HomeUiState.Clustered(
            clusters = clusters,
            selectedClusterIds = emptySet(),
        ))
    }

    private fun handleNoFaces() {
        Log.w(TAG, "handleNoFaces: pipeline reported no faces; surfacing toast and Empty")
        viewModelScope.launch { emitMessage("noFaces", UiMessage.NoFacesFound) }
        transition("Clusterify/noFaces", HomeUiState.Empty)
    }

    private fun onToggleCluster(id: String) {
        Log.i(TAG, "onToggleCluster: id=$id")
        updateState("ToggleCluster") { s ->
            when (s) {
                is HomeUiState.Clustered -> s.copy(
                    selectedClusterIds = s.selectedClusterIds.toggle(id),
                )
                is HomeUiState.FilterReady -> s.copy(
                    selectedClusterIds = s.selectedClusterIds.toggle(id),
                )
                else -> {
                    Log.w(TAG, "onToggleCluster: ignored in ${describe(s)}")
                    s
                }
            }
        }
    }

    private fun onFilterPhotosPicked(uris: List<Uri>) {
        if (uris.isEmpty()) {
            Log.i(TAG, "onFilterPhotosPicked: empty -> ignore")
            return
        }
        val capped = uris.take(MAX_FILTER_PHOTOS)
        Log.i(
            TAG,
            "onFilterPhotosPicked: received ${uris.size} (capped to ${capped.size}, max=$MAX_FILTER_PHOTOS)",
        )
        if (uris.size > MAX_FILTER_PHOTOS) {
            Log.w(TAG, "onFilterPhotosPicked: user picked ${uris.size} > $MAX_FILTER_PHOTOS; emitting MaxFilterImagesReached")
            viewModelScope.launch { emitMessage("maxFilterImages", UiMessage.MaxFilterImagesReached) }
        }
        updateState("FilterPhotosPicked") { s ->
            when (s) {
                is HomeUiState.Clustered -> HomeUiState.FilterReady(
                    clusters = s.clusters,
                    selectedClusterIds = s.selectedClusterIds,
                    filterPhotos = capped,
                )
                is HomeUiState.FilterReady -> s.copy(filterPhotos = capped)
                else -> {
                    Log.w(TAG, "onFilterPhotosPicked: ignored in ${describe(s)}")
                    s
                }
            }
        }
    }

    private fun onFilterTapped() {
        val current = _state.value as? HomeUiState.FilterReady ?: run {
            Log.w(TAG, "onFilterTapped: ignored, state is ${describe(_state.value)} (need FilterReady)")
            return
        }
        if (!current.canFilter) {
            Log.w(
                TAG,
                "onFilterTapped: ignored, canFilter=false (filterPhotos=${current.filterPhotos.size}, " +
                    "selectedClusters=${current.selectedClusterIds.size})",
            )
            return
        }
        val total = current.filterPhotos.size
        Log.i(
            TAG,
            "onFilterTapped: starting Filter on $total photo(s) against " +
                "${current.selectedClusterIds.size} cluster(s)=${current.selectedClusterIds}",
        )
        transition("Filter/start", HomeUiState.Matching(
            processed = 0,
            total = total,
            clusters = current.clusters,
            selectedClusterIds = current.selectedClusterIds,
            filterPhotos = current.filterPhotos,
        ))

        if (mlPipelineProvider == null) {
            Log.i(TAG, "onFilterTapped: pipeline null -> running MOCK matcher")
            runMockMatching(total)
            return
        }

        pipelineJob?.cancel()
        val launchedAt = SystemClock.elapsedRealtime()
        pipelineJob = viewModelScope.launch {
            val useCase = mlPipelineProvider.filterUseCase()
            Log.i(TAG, "onFilterTapped: collecting filter use-case flow")
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
            Log.i(
                TAG,
                "onFilterTapped: filter flow completed in ${SystemClock.elapsedRealtime() - launchedAt}ms",
            )
        }
    }

    private fun runMockMatching(total: Int) {
        Log.i(TAG, "runMockMatching: total=$total")
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
            Log.i(TAG, "finishMatching: 0 keepers -> handleNoMatches")
            handleNoMatches()
            return
        }
        val current = _state.value as? HomeUiState.Matching ?: run {
            Log.w(TAG, "finishMatching: ignored, state is ${describe(_state.value)} (need Matching)")
            return
        }
        val sessionId = UUID.randomUUID().toString()
        Log.i(TAG, "finishMatching: ${keepers.size} keeper(s) sessionId=$sessionId")
        _keepers.update { it + (sessionId to keepers) }
        transition("Filter/finished", HomeUiState.Clustered(
            clusters = current.clusters,
            selectedClusterIds = current.selectedClusterIds,
        ))
        viewModelScope.launch { emitMessage("navigateToKeepers", UiMessage.NavigateToKeepers(sessionId)) }
    }

    private fun handleNoMatches() {
        val current = _state.value as? HomeUiState.Matching ?: run {
            Log.w(TAG, "handleNoMatches: ignored, state is ${describe(_state.value)} (need Matching)")
            return
        }
        Log.w(TAG, "handleNoMatches: filter produced no keepers")
        transition("Filter/noMatches", HomeUiState.Clustered(
            clusters = current.clusters,
            selectedClusterIds = current.selectedClusterIds,
        ))
        viewModelScope.launch { emitMessage("noMatches", UiMessage.NoMatchesFound) }
    }

    private fun onClearFilter() {
        val current = _state.value as? HomeUiState.FilterReady ?: run {
            Log.w(TAG, "onClearFilter: ignored, state is ${describe(_state.value)} (need FilterReady)")
            return
        }
        Log.i(TAG, "onClearFilter: discarding ${current.filterPhotos.size} filter photo(s)")
        transition("Filter/clear", HomeUiState.Clustered(
            clusters = current.clusters,
            selectedClusterIds = current.selectedClusterIds,
        ))
    }

    private fun onReset() {
        Log.w(TAG, "onReset: user confirmed Reset; cancelling pipeline + clearing keepers + state")
        pipelineJob?.cancel()
        _keepers.value = emptyMap()
        _clusterImages.value = emptyMap()
        transition("Reset", HomeUiState.Empty)
        viewModelScope.launch {
            // Best-effort wipe; if either is null (e.g. unit tests) just skip.
            Log.i(TAG, "onReset: wiping repository + preferences")
            clusterRepository?.deleteAll()
            preferences?.clearAll()
            Log.i(TAG, "onReset: done")
        }
    }

    private fun onDeleteCluster(id: String) {
        Log.i(TAG, "onDeleteCluster: id=$id")
        _clusterImages.update { it - id }
        updateState("DeleteCluster") { s ->
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
                else -> {
                    Log.w(TAG, "onDeleteCluster: ignored in ${describe(s)}")
                    s
                }
            }
        }
        viewModelScope.launch { clusterRepository?.deleteCluster(id) }
    }

    private fun onCancelProcessing() {
        Log.w(TAG, "onCancelProcessing: cancelling pipeline job; state=${describe(_state.value)}")
        pipelineJob?.cancel()
        updateState("CancelProcessing") { s ->
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
        Log.i(TAG, "onReturnedFromKeepers: user popped Keepers route (state preserved)")
    }

    private fun Set<String>.toggle(id: String): Set<String> =
        if (contains(id)) this - id else this + id

    /** Compact, copy-pasteable description of an intent for log lines. */
    private fun describe(intent: HomeIntent): String = when (intent) {
        HomeIntent.AddPhotosTapped -> "AddPhotosTapped"
        is HomeIntent.PhotosPicked -> "PhotosPicked(n=${intent.uris.size})"
        HomeIntent.ClusterifyTapped -> "ClusterifyTapped"
        is HomeIntent.ToggleClusterChecked -> "ToggleClusterChecked(id=${intent.clusterId})"
        HomeIntent.CameraTapped -> "CameraTapped"
        is HomeIntent.FilterPhotosPicked -> "FilterPhotosPicked(n=${intent.uris.size})"
        HomeIntent.FilterTapped -> "FilterTapped"
        HomeIntent.ClearFilterTapped -> "ClearFilterTapped"
        HomeIntent.ResetConfirmed -> "ResetConfirmed"
        is HomeIntent.DeleteClusterConfirmed -> "DeleteClusterConfirmed(id=${intent.clusterId})"
        HomeIntent.CancelProcessingTapped -> "CancelProcessingTapped"
        HomeIntent.ReturnedFromKeepers -> "ReturnedFromKeepers"
        is HomeIntent.ClusterifyProgress -> "ClusterifyProgress(${intent.processed}/${intent.total})"
        is HomeIntent.ClusterifyFinished -> "ClusterifyFinished(clusters=${intent.clusters.size})"
        HomeIntent.ClusterifyNoFaces -> "ClusterifyNoFaces"
        is HomeIntent.FilterProgress -> "FilterProgress(${intent.processed}/${intent.total})"
        is HomeIntent.FilterFinished -> "FilterFinished(keepers=${intent.keepers.size})"
        HomeIntent.FilterNoMatches -> "FilterNoMatches"
    }

    private fun describe(state: HomeUiState): String = when (state) {
        HomeUiState.Empty -> "Empty"
        is HomeUiState.Selecting -> "Selecting(photos=${state.selectedPhotos.size}, fan=${state.recentFan.size})"
        is HomeUiState.Processing -> "Processing(${state.processed}/${state.total}, dl=${state.downloadFraction})"
        is HomeUiState.Clustered ->
            "Clustered(clusters=${state.clusters.size}, selected=${state.selectedClusterIds.size})"
        is HomeUiState.FilterReady ->
            "FilterReady(clusters=${state.clusters.size}, selected=${state.selectedClusterIds.size}, " +
                "filterPhotos=${state.filterPhotos.size})"
        is HomeUiState.Matching ->
            "Matching(${state.processed}/${state.total}, clusters=${state.clusters.size}, " +
                "selected=${state.selectedClusterIds.size})"
    }

    companion object {
        private const val TAG: String = "FaceMesh.HomeVM"
        const val MAX_FAN: Int = 4
        const val MAX_FILTER_PHOTOS: Int = 15
    }
}
