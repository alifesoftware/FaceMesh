package com.alifesoftware.facemesh.ml

/**
 * Generates anchors for the BlazeFace **full-range** 192 x 192 model.
 *
 * Unlike short-range BlazeFace which uses two SSD prediction layers, the full-range
 * variant uses a **single** layer at stride 4 with one anchor per cell:
 *
 *   - 192 / 4 = 48-cell grid in each axis
 *   - 1 anchor per cell
 *   - **Total: 48 x 48 x 1 = 2304 anchors**
 *
 * Anchor (cx, cy) is the centre of the cell, normalised to [0, 1]. Width and height are
 * 1.0 (the regressor predicts displacements relative to anchor size).
 *
 * Companion to [BlazeFaceShortRangeAnchors]; the two anchor tables are deliberately
 * separate types so the code paths can't accidentally share state.
 */
object BlazeFaceFullRangeAnchors {

    data class Anchor(val cx: Float, val cy: Float)

    /** Single 48 x 48 grid (stride 4 over 192 px input), 1 anchor per cell -> 2304 anchors. */
    val GRID_192: Array<Anchor> by lazy { generateGrid192() }

    private fun generateGrid192(): Array<Anchor> {
        val anchors = ArrayList<Anchor>(2304)
        for (y in 0 until GRID_SIZE) {
            for (x in 0 until GRID_SIZE) {
                val cx = (x + 0.5f) / GRID_SIZE
                val cy = (y + 0.5f) / GRID_SIZE
                anchors.add(Anchor(cx, cy))
            }
        }
        return anchors.toTypedArray()
    }

    private const val GRID_SIZE: Int = 48
}
