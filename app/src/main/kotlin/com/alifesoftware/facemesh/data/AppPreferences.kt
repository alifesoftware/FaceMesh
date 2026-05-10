package com.alifesoftware.facemesh.data

import android.content.Context
import android.util.Log
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.alifesoftware.facemesh.config.PipelineConfig
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onEach

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "facemesh_prefs")

/**
 * Wraps the Preferences DataStore for typed access.
 *
 * Defaults match SPEC \u00a77.3 and \u00a76.5/6.6. The DataStore-backed manifest values can be
 * further overridden by user adjustments in Settings; see [Source] for the priority resolution
 * applied to [dbscanEps] and [matchThreshold].
 */
class AppPreferences(private val context: Context) {

    /**
     * Where the effective value for a tuned preference came from. Returned alongside each
     * resolved value so the use-cases can include it in their per-run diagnostic logs.
     */
    enum class Source {
        /** User moved the slider in Settings; this is the highest-priority layer. */
        USER,
        /** A model manifest has been ingested and wrote a value into DataStore. */
        MANIFEST,
        /** Cold start before any manifest fetch has succeeded. */
        DEFAULT,
    }

    /**
     * Resolved DBSCAN epsilon (3-layer priority: user override > manifest-written value >
     * [PipelineConfig.Clustering.defaultEps]).
     */
    val dbscanEps: Flow<Float> = context.dataStore.data
        .map { resolveDbscanEps(it).first }
        .observed("dbscanEps")

    /**
     * Source of the value emitted by [dbscanEps]. Useful for diagnostic logging.
     */
    val dbscanEpsSource: Flow<Source> = context.dataStore.data
        .map { resolveDbscanEps(it).second }
        .observed("dbscanEpsSource")

    /**
     * The user override for [dbscanEps], or `null` when the user has not moved the slider.
     * Drives the "Reset" affordance in Settings.
     */
    val dbscanEpsUserOverride: Flow<Float?> = context.dataStore.data
        .map { it[KEY_DBSCAN_EPS_USER_OVERRIDE] }
        .observed("dbscanEpsUserOverride")

    /**
     * No user-override layer for `dbscanMinPts` - it remains manifest/config-only by deliberate
     * design (algorithmic, not intuitive enough for a slider).
     */
    val dbscanMinPts: Flow<Int> = context.dataStore.data
        .map { it[KEY_DBSCAN_MIN_PTS] ?: DEFAULT_DBSCAN_MIN_PTS }
        .observed("dbscanMinPts")

    /**
     * Resolved cosine match threshold (3-layer priority: user override > manifest-written
     * value > [PipelineConfig.Match.defaultThreshold]).
     */
    val matchThreshold: Flow<Float> = context.dataStore.data
        .map { resolveMatchThreshold(it).first }
        .observed("matchThreshold")

    val matchThresholdSource: Flow<Source> = context.dataStore.data
        .map { resolveMatchThreshold(it).second }
        .observed("matchThresholdSource")

    val matchThresholdUserOverride: Flow<Float?> = context.dataStore.data
        .map { it[KEY_MATCH_THRESHOLD_USER_OVERRIDE] }
        .observed("matchThresholdUserOverride")
    val modelsVersion: Flow<Int> = context.dataStore.data
        .map { it[KEY_MODELS_VERSION] ?: 0 }
        .observed("modelsVersion")
    val lastModelsCheck: Flow<Long> = context.dataStore.data
        .map { it[KEY_LAST_MODELS_CHECK] ?: 0L }
        .observed("lastModelsCheck")
    val pendingFilterSession: Flow<String?> = context.dataStore.data
        .map { it[KEY_PENDING_FILTER_SESSION] }
        .observed("pendingFilterSession")
    val gpuFallbackToastShown: Flow<Boolean> = context.dataStore.data
        .map { it[KEY_GPU_FALLBACK_TOAST_SHOWN] ?: false }
        .observed("gpuFallbackToastShown")

    /**
     * When true (and the device is Android 12+), the Compose theme overrides the All Black
     * brand palette with a Material You dynamic-color scheme derived from the user's wallpaper.
     * Default is false so that first-launch UX is the deliberate brand palette.
     */
    val dynamicColorEnabled: Flow<Boolean> = context.dataStore.data
        .map { it[KEY_DYNAMIC_COLOR_ENABLED] ?: false }
        .observed("dynamicColorEnabled")

    /**
     * Manifest path: writes the manifest-supplied value into DataStore. Does NOT clear any
     * existing user override - the override always wins at read time.
     */
    suspend fun setDbscanEps(value: Float) {
        Log.i(TAG, "set dbscanEps=$value (manifest layer)")
        context.dataStore.edit { it[KEY_DBSCAN_EPS] = value }
    }

    /**
     * User path: writes the slider-supplied user override. Pass `null` to clear (i.e. fall
     * back to the manifest value, then to [PipelineConfig.Clustering.defaultEps]).
     */
    suspend fun setDbscanEpsUserOverride(value: Float?) {
        Log.i(TAG, "set dbscanEpsUserOverride=$value (null=clear)")
        context.dataStore.edit { prefs ->
            if (value == null) prefs.remove(KEY_DBSCAN_EPS_USER_OVERRIDE)
            else prefs[KEY_DBSCAN_EPS_USER_OVERRIDE] = value
        }
    }

    suspend fun setDbscanMinPts(value: Int) {
        Log.i(TAG, "set dbscanMinPts=$value")
        context.dataStore.edit { it[KEY_DBSCAN_MIN_PTS] = value }
    }

    /**
     * Manifest path: writes the manifest-supplied value into DataStore. Does NOT clear any
     * existing user override - the override always wins at read time.
     */
    suspend fun setMatchThreshold(value: Float) {
        Log.i(TAG, "set matchThreshold=$value (manifest layer)")
        context.dataStore.edit { it[KEY_MATCH_THRESHOLD] = value }
    }

    /**
     * User path: writes the slider-supplied user override. Pass `null` to clear (i.e. fall
     * back to the manifest value, then to [PipelineConfig.Match.defaultThreshold]).
     */
    suspend fun setMatchThresholdUserOverride(value: Float?) {
        Log.i(TAG, "set matchThresholdUserOverride=$value (null=clear)")
        context.dataStore.edit { prefs ->
            if (value == null) prefs.remove(KEY_MATCH_THRESHOLD_USER_OVERRIDE)
            else prefs[KEY_MATCH_THRESHOLD_USER_OVERRIDE] = value
        }
    }
    suspend fun setModelsVersion(value: Int) {
        Log.i(TAG, "set modelsVersion=$value")
        context.dataStore.edit { it[KEY_MODELS_VERSION] = value }
    }
    suspend fun setLastModelsCheck(value: Long) {
        Log.i(TAG, "set lastModelsCheck=$value")
        context.dataStore.edit { it[KEY_LAST_MODELS_CHECK] = value }
    }
    suspend fun setPendingFilterSession(value: String?) {
        Log.i(TAG, "set pendingFilterSession=$value")
        context.dataStore.edit {
            if (value == null) it.remove(KEY_PENDING_FILTER_SESSION) else it[KEY_PENDING_FILTER_SESSION] = value
        }
    }
    suspend fun setGpuFallbackToastShown(value: Boolean) {
        Log.i(TAG, "set gpuFallbackToastShown=$value")
        context.dataStore.edit { it[KEY_GPU_FALLBACK_TOAST_SHOWN] = value }
    }
    suspend fun setDynamicColorEnabled(value: Boolean) {
        Log.i(TAG, "set dynamicColorEnabled=$value")
        context.dataStore.edit { it[KEY_DYNAMIC_COLOR_ENABLED] = value }
    }

    suspend fun clearAll() {
        Log.w(TAG, "clearAll: wiping ALL preferences (Reset flow)")
        context.dataStore.edit { it.clear() }
    }

    private fun <T> Flow<T>.observed(name: String): Flow<T> = distinctUntilChanged().onEach {
        Log.i(TAG, "observe $name=$it")
    }

    /**
     * Layered resolution for [dbscanEps]:
     *   1. user override (if present) -> [Source.USER]
     *   2. manifest-written value (if present) -> [Source.MANIFEST]
     *   3. [PipelineConfig.Clustering.defaultEps]   -> [Source.DEFAULT]
     */
    private fun resolveDbscanEps(prefs: Preferences): Pair<Float, Source> {
        prefs[KEY_DBSCAN_EPS_USER_OVERRIDE]?.let { return it to Source.USER }
        prefs[KEY_DBSCAN_EPS]?.let { return it to Source.MANIFEST }
        return DEFAULT_DBSCAN_EPS to Source.DEFAULT
    }

    /** Layered resolution for [matchThreshold]; same shape as [resolveDbscanEps]. */
    private fun resolveMatchThreshold(prefs: Preferences): Pair<Float, Source> {
        prefs[KEY_MATCH_THRESHOLD_USER_OVERRIDE]?.let { return it to Source.USER }
        prefs[KEY_MATCH_THRESHOLD]?.let { return it to Source.MANIFEST }
        return DEFAULT_MATCH_THRESHOLD to Source.DEFAULT
    }

    companion object {
        private const val TAG: String = "FaceMesh.Prefs"

        // Defaults for user-overridable settings live in PipelineConfig (single source of
        // truth for both runtime defaults AND any future Settings UI sliders' min/max bounds).
        const val DEFAULT_DBSCAN_EPS: Float = PipelineConfig.Clustering.defaultEps
        const val DEFAULT_DBSCAN_MIN_PTS: Int = PipelineConfig.Clustering.defaultMinPts
        const val DEFAULT_MATCH_THRESHOLD: Float = PipelineConfig.Match.defaultThreshold

        private val KEY_DBSCAN_EPS = floatPreferencesKey("dbscan_eps")
        private val KEY_DBSCAN_EPS_USER_OVERRIDE = floatPreferencesKey("dbscan_eps_user_override")
        private val KEY_DBSCAN_MIN_PTS = intPreferencesKey("dbscan_min_pts")
        private val KEY_MATCH_THRESHOLD = floatPreferencesKey("match_threshold")
        private val KEY_MATCH_THRESHOLD_USER_OVERRIDE = floatPreferencesKey("match_threshold_user_override")
        private val KEY_MODELS_VERSION = intPreferencesKey("models_version")
        private val KEY_LAST_MODELS_CHECK = longPreferencesKey("last_models_check")
        private val KEY_PENDING_FILTER_SESSION = stringPreferencesKey("pending_filter_session")
        private val KEY_GPU_FALLBACK_TOAST_SHOWN = booleanPreferencesKey("gpu_fallback_toast_shown")
        private val KEY_DYNAMIC_COLOR_ENABLED = booleanPreferencesKey("dynamic_color_enabled")
    }
}
