package com.alifesoftware.facemesh.data

import com.alifesoftware.facemesh.config.PipelineConfig
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class AppPreferencesTest {

    private val ctx: android.content.Context = RuntimeEnvironment.getApplication()
    private val prefs = AppPreferences(ctx)

    @Before
    fun resetStateBeforeEachTest() {
        // DataStore caches state in-memory under the Context.dataStore delegate (singleton per
        // Context per name) so just deleting the on-disk file isn't enough to reset between
        // tests - we have to write through clearAll() so the in-memory store is also empty.
        runBlocking { prefs.clearAll() }
    }

    @After
    fun tearDown() {
        runCatching {
            File(ctx.filesDir.parentFile, "datastore").deleteRecursively()
            File(ctx.filesDir, "datastore").deleteRecursively()
        }
    }

    @Test
    fun `dynamicColorEnabled defaults to false`() = runTest {
        assertFalse(prefs.dynamicColorEnabled.first())
    }

    @Test
    fun `setDynamicColorEnabled persists the new value`() = runTest {
        prefs.setDynamicColorEnabled(true)
        assertTrue(prefs.dynamicColorEnabled.first())
        prefs.setDynamicColorEnabled(false)
        assertFalse(prefs.dynamicColorEnabled.first())
    }

    @Test
    fun `match threshold default is the GhostFaceNet value`() = runTest {
        assertEquals(0.65f, prefs.matchThreshold.first(), 1e-6f)
    }

    @Test
    fun `dbscan defaults are present`() = runTest {
        assertEquals(0.35f, prefs.dbscanEps.first(), 1e-6f)
        assertEquals(2, prefs.dbscanMinPts.first())
    }

    // -------- 3-layer priority resolution: user override > manifest > config default --------

    @Test
    fun `dbscanEps source is DEFAULT when neither manifest nor user has written`() = runTest {
        assertEquals(AppPreferences.Source.DEFAULT, prefs.dbscanEpsSource.first())
        assertEquals(PipelineConfig.Clustering.defaultEps, prefs.dbscanEps.first(), 1e-6f)
        assertNull(prefs.dbscanEpsUserOverride.first())
    }

    @Test
    fun `dbscanEps source flips to MANIFEST after a manifest write`() = runTest {
        prefs.setDbscanEps(0.42f)
        assertEquals(AppPreferences.Source.MANIFEST, prefs.dbscanEpsSource.first())
        assertEquals(0.42f, prefs.dbscanEps.first(), 1e-6f)
        assertNull(prefs.dbscanEpsUserOverride.first())
    }

    @Test
    fun `dbscanEps user override beats the manifest layer`() = runTest {
        prefs.setDbscanEps(0.42f)                      // manifest layer
        prefs.setDbscanEpsUserOverride(0.55f)          // user layer
        assertEquals(AppPreferences.Source.USER, prefs.dbscanEpsSource.first())
        assertEquals(0.55f, prefs.dbscanEps.first(), 1e-6f)
        assertEquals(0.55f, prefs.dbscanEpsUserOverride.first()!!, 1e-6f)
    }

    @Test
    fun `clearing the dbscanEps user override falls back to manifest`() = runTest {
        prefs.setDbscanEps(0.42f)
        prefs.setDbscanEpsUserOverride(0.55f)
        prefs.setDbscanEpsUserOverride(null)
        assertEquals(AppPreferences.Source.MANIFEST, prefs.dbscanEpsSource.first())
        assertEquals(0.42f, prefs.dbscanEps.first(), 1e-6f)
        assertNull(prefs.dbscanEpsUserOverride.first())
    }

    @Test
    fun `manifest re-write does not clobber an active user override`() = runTest {
        prefs.setDbscanEpsUserOverride(0.55f)
        prefs.setDbscanEps(0.42f) // simulates a manifest re-download
        assertEquals(AppPreferences.Source.USER, prefs.dbscanEpsSource.first())
        assertEquals(0.55f, prefs.dbscanEps.first(), 1e-6f)
    }

    @Test
    fun `matchThreshold layered priority mirrors dbscanEps`() = runTest {
        assertEquals(AppPreferences.Source.DEFAULT, prefs.matchThresholdSource.first())
        prefs.setMatchThreshold(0.70f)
        assertEquals(AppPreferences.Source.MANIFEST, prefs.matchThresholdSource.first())
        prefs.setMatchThresholdUserOverride(0.80f)
        assertEquals(AppPreferences.Source.USER, prefs.matchThresholdSource.first())
        assertEquals(0.80f, prefs.matchThreshold.first(), 1e-6f)
        prefs.setMatchThresholdUserOverride(null)
        assertEquals(0.70f, prefs.matchThreshold.first(), 1e-6f)
    }
}
