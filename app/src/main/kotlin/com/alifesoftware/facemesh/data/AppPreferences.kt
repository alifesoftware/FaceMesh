package com.alifesoftware.facemesh.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "facemesh_prefs")

/**
 * Wraps the Preferences DataStore for typed access.
 *
 * Defaults match SPEC \u00a77.3 and \u00a76.5/6.6. They are overridable via the downloaded `config.json`
 * (Phase 3) which writes through to the same store at runtime.
 */
class AppPreferences(private val context: Context) {

    val dbscanEps: Flow<Float> = context.dataStore.data.map { it[KEY_DBSCAN_EPS] ?: DEFAULT_DBSCAN_EPS }
    val dbscanMinPts: Flow<Int> = context.dataStore.data.map { it[KEY_DBSCAN_MIN_PTS] ?: DEFAULT_DBSCAN_MIN_PTS }
    val matchThreshold: Flow<Float> = context.dataStore.data.map { it[KEY_MATCH_THRESHOLD] ?: DEFAULT_MATCH_THRESHOLD }
    val modelsVersion: Flow<Int> = context.dataStore.data.map { it[KEY_MODELS_VERSION] ?: 0 }
    val lastModelsCheck: Flow<Long> = context.dataStore.data.map { it[KEY_LAST_MODELS_CHECK] ?: 0L }
    val pendingFilterSession: Flow<String?> = context.dataStore.data.map { it[KEY_PENDING_FILTER_SESSION] }
    val gpuFallbackToastShown: Flow<Boolean> = context.dataStore.data.map { it[KEY_GPU_FALLBACK_TOAST_SHOWN] ?: false }

    /**
     * When true (and the device is Android 12+), the Compose theme overrides the All Black
     * brand palette with a Material You dynamic-color scheme derived from the user's wallpaper.
     * Default is false so that first-launch UX is the deliberate brand palette.
     */
    val dynamicColorEnabled: Flow<Boolean> = context.dataStore.data.map { it[KEY_DYNAMIC_COLOR_ENABLED] ?: false }

    suspend fun setDbscanEps(value: Float) = context.dataStore.edit { it[KEY_DBSCAN_EPS] = value }
    suspend fun setDbscanMinPts(value: Int) = context.dataStore.edit { it[KEY_DBSCAN_MIN_PTS] = value }
    suspend fun setMatchThreshold(value: Float) = context.dataStore.edit { it[KEY_MATCH_THRESHOLD] = value }
    suspend fun setModelsVersion(value: Int) = context.dataStore.edit { it[KEY_MODELS_VERSION] = value }
    suspend fun setLastModelsCheck(value: Long) = context.dataStore.edit { it[KEY_LAST_MODELS_CHECK] = value }
    suspend fun setPendingFilterSession(value: String?) = context.dataStore.edit {
        if (value == null) it.remove(KEY_PENDING_FILTER_SESSION) else it[KEY_PENDING_FILTER_SESSION] = value
    }
    suspend fun setGpuFallbackToastShown(value: Boolean) = context.dataStore.edit {
        it[KEY_GPU_FALLBACK_TOAST_SHOWN] = value
    }
    suspend fun setDynamicColorEnabled(value: Boolean) = context.dataStore.edit {
        it[KEY_DYNAMIC_COLOR_ENABLED] = value
    }

    suspend fun clearAll() = context.dataStore.edit { it.clear() }

    companion object {
        const val DEFAULT_DBSCAN_EPS: Float = 0.35f
        const val DEFAULT_DBSCAN_MIN_PTS: Int = 2

        // GhostFaceNet-V1 emits 512-d embeddings; in that regime "same person" cosine
        // similarity typically lives in 0.55-0.65 (vs ~0.80 for the 128-d MobileFaceNet
        // models the SPEC initially referenced). Calibrate empirically on real photos
        // and override at runtime via the downloaded `manifest.json` config.
        const val DEFAULT_MATCH_THRESHOLD: Float = 0.65f

        private val KEY_DBSCAN_EPS = floatPreferencesKey("dbscan_eps")
        private val KEY_DBSCAN_MIN_PTS = intPreferencesKey("dbscan_min_pts")
        private val KEY_MATCH_THRESHOLD = floatPreferencesKey("match_threshold")
        private val KEY_MODELS_VERSION = intPreferencesKey("models_version")
        private val KEY_LAST_MODELS_CHECK = longPreferencesKey("last_models_check")
        private val KEY_PENDING_FILTER_SESSION = stringPreferencesKey("pending_filter_session")
        private val KEY_GPU_FALLBACK_TOAST_SHOWN = booleanPreferencesKey("gpu_fallback_toast_shown")
        private val KEY_DYNAMIC_COLOR_ENABLED = booleanPreferencesKey("dynamic_color_enabled")
    }
}
