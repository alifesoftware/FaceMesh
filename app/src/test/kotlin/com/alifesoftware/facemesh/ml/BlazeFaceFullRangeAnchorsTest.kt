package com.alifesoftware.facemesh.ml

import com.alifesoftware.facemesh.config.PipelineConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BlazeFaceFullRangeAnchorsTest {

    @Test
    fun `single 48 x 48 grid yields exactly numAnchors entries`() {
        val anchors = BlazeFaceFullRangeAnchors.GRID_192
        assertEquals(2304, anchors.size)
        assertEquals(PipelineConfig.Detector.FullRange.numAnchors, anchors.size)
    }

    @Test
    fun `every anchor centre is in the unit square`() {
        for (a in BlazeFaceFullRangeAnchors.GRID_192) {
            assertTrue("cx ${a.cx} in [0,1]", a.cx in 0f..1f)
            assertTrue("cy ${a.cy} in [0,1]", a.cy in 0f..1f)
        }
    }

    @Test
    fun `top-left anchor lands at half-pixel offset of the 48-cell grid`() {
        // First anchor is for cell (0, 0). With grid size 48, centre = (0.5/48, 0.5/48).
        val first = BlazeFaceFullRangeAnchors.GRID_192[0]
        assertEquals(0.5f / 48f, first.cx, 1e-6f)
        assertEquals(0.5f / 48f, first.cy, 1e-6f)
    }

    @Test
    fun `bottom-right anchor lands at the opposite half-pixel offset`() {
        // Last anchor is for cell (47, 47). Centre = (47.5/48, 47.5/48).
        val last = BlazeFaceFullRangeAnchors.GRID_192.last()
        assertEquals(47.5f / 48f, last.cx, 1e-6f)
        assertEquals(47.5f / 48f, last.cy, 1e-6f)
    }

    @Test
    fun `anchor table is row-major -- adjacent indices share a row when not at the edge`() {
        // Cell (0, 0) -> index 0; cell (1, 0) -> index 1; same y.
        val zero = BlazeFaceFullRangeAnchors.GRID_192[0]
        val one = BlazeFaceFullRangeAnchors.GRID_192[1]
        assertEquals(zero.cy, one.cy, 1e-6f)
        assertTrue("cx must increase by exactly 1/48 across columns", one.cx > zero.cx)
        assertEquals(1f / 48f, one.cx - zero.cx, 1e-6f)

        // Crossing into row 1: index 48 -> cell (0, 1). Same cx as index 0, cy bumped.
        val rowOneStart = BlazeFaceFullRangeAnchors.GRID_192[48]
        assertEquals(zero.cx, rowOneStart.cx, 1e-6f)
        assertEquals(1f / 48f, rowOneStart.cy - zero.cy, 1e-6f)
    }
}
