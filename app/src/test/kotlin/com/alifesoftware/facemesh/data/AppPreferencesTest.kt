package com.alifesoftware.facemesh.data

import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
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

    @After
    fun tearDown() {
        // DataStore writes a single file under filesDir/datastore. Wipe between tests for
        // isolation since AppPreferences is a singleton-style object backed by a file.
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
}
