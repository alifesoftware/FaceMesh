package com.alifesoftware.facemesh.viewmodel

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.alifesoftware.facemesh.config.PipelineConfig
import com.alifesoftware.facemesh.config.PipelineConfig.Detector.DetectorVariant
import com.alifesoftware.facemesh.data.AppPreferences
import com.alifesoftware.facemesh.di.AppContainer
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

/**
 * Backs the [com.alifesoftware.facemesh.ui.screens.SettingsScreen] with read/write access to
 * the in-app preference flags, including the user-tunable pipeline knobs surfaced as sliders
 * (DBSCAN epsilon and the cosine match threshold). See [AppPreferences.Source] for the
 * priority resolution between user override / manifest / config-default.
 */
class SettingsViewModel(
    private val preferences: AppPreferences,
) : ViewModel() {

    init {
        Log.i(TAG, "init: SettingsViewModel created")
    }

    val dynamicColorEnabled: Flow<Boolean> = preferences.dynamicColorEnabled

    /** Currently effective DBSCAN epsilon (drives the slider's position). */
    val dbscanEps: Flow<Float> = preferences.dbscanEps

    /** Where the [dbscanEps] value came from. Drives the "(modified)" / source caption. */
    val dbscanEpsSource: Flow<AppPreferences.Source> = preferences.dbscanEpsSource

    /** True when the user has moved the slider (i.e. an override is active). */
    val dbscanEpsHasUserOverride: Flow<Boolean> = preferences.dbscanEpsUserOverride.map { it != null }

    /** Currently effective cosine match threshold. */
    val matchThreshold: Flow<Float> = preferences.matchThreshold

    val matchThresholdSource: Flow<AppPreferences.Source> = preferences.matchThresholdSource

    val matchThresholdHasUserOverride: Flow<Boolean> =
        preferences.matchThresholdUserOverride.map { it != null }

    /** Active BlazeFace detector variant; drives the SegmentedButton selection in Settings. */
    val detectorVariant: Flow<DetectorVariant> = preferences.detectorVariant

    /**
     * Slider bounds + step come straight from [PipelineConfig] so the UI is the single source
     * of consistency with the underlying algorithms' valid input ranges.
     */
    val dbscanEpsBounds: Bounds = Bounds(
        min = PipelineConfig.Clustering.minEps,
        max = PipelineConfig.Clustering.maxEps,
        step = 0.01f,
    )

    val matchThresholdBounds: Bounds = Bounds(
        min = PipelineConfig.Match.minThreshold,
        max = PipelineConfig.Match.maxThreshold,
        step = 0.01f,
    )

    fun setDynamicColorEnabled(value: Boolean) {
        Log.i(TAG, "setDynamicColorEnabled: user toggled to $value")
        viewModelScope.launch { preferences.setDynamicColorEnabled(value) }
    }

    /**
     * Called as the user drags the eps slider. Writes the value as a user override (highest
     * priority layer); the manifest layer is left untouched so that "Reset" can fall back to
     * it later.
     */
    fun setDbscanEpsUserOverride(value: Float) {
        Log.i(TAG, "setDbscanEpsUserOverride: user moved slider to $value")
        viewModelScope.launch { preferences.setDbscanEpsUserOverride(value) }
    }

    /** Clears the user-override layer for eps; effective value falls back to manifest/default. */
    fun resetDbscanEpsUserOverride() {
        Log.i(TAG, "resetDbscanEpsUserOverride: clearing user override")
        viewModelScope.launch { preferences.setDbscanEpsUserOverride(null) }
    }

    fun setMatchThresholdUserOverride(value: Float) {
        Log.i(TAG, "setMatchThresholdUserOverride: user moved slider to $value")
        viewModelScope.launch { preferences.setMatchThresholdUserOverride(value) }
    }

    fun resetMatchThresholdUserOverride() {
        Log.i(TAG, "resetMatchThresholdUserOverride: clearing user override")
        viewModelScope.launch { preferences.setMatchThresholdUserOverride(null) }
    }

    /**
     * Persists the user's detector-variant choice. The change takes effect on the next
     * `clusterifyUseCase()` / `filterUseCase()` call - `MlPipelineProvider` reads the
     * preference and tears down + rebuilds its detector if the value differs from the
     * cached one.
     */
    fun setDetectorVariant(value: DetectorVariant) {
        Log.i(TAG, "setDetectorVariant: user picked $value")
        viewModelScope.launch { preferences.setDetectorVariant(value) }
    }

    /** Slider configuration shape for [SettingsScreen] consumption. */
    data class Bounds(val min: Float, val max: Float, val step: Float) {
        /** Matches Material3 Slider's expected `steps` parameter (count of intermediate ticks). */
        val sliderSteps: Int
            get() = (((max - min) / step).toInt() - 1).coerceAtLeast(0)
    }

    companion object {
        private const val TAG: String = "FaceMesh.SettingsVM"
    }
}

class SettingsViewModelFactory(private val container: AppContainer) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        require(modelClass.isAssignableFrom(SettingsViewModel::class.java)) {
            "SettingsViewModelFactory does not produce ${modelClass.name}"
        }
        Log.i("FaceMesh.SettingsVM", "factory.create: building SettingsViewModel")
        return SettingsViewModel(preferences = container.preferences) as T
    }
}
