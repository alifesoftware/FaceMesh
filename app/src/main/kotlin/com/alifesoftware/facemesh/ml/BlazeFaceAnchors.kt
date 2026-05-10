package com.alifesoftware.facemesh.ml

/**
 * Generates anchors for the front-camera BlazeFace 128x128 model.
 *
 * The model has two SSD prediction layers:
 *   \u2022 stride 8  on a 16\u00d716 grid \u2192 2 anchors per cell  = 512 anchors
 *   \u2022 stride 16 on an 8\u00d78 grid  \u2192 6 anchors per cell  = 384 anchors
 * Total: 896 anchors. Anchor (cx, cy) is the centre of the cell, normalised to [0,1].
 * Width and height are 1.0 (the regressor predicts displacements relative to anchor size).
 */
object BlazeFaceAnchors {

    data class Anchor(val cx: Float, val cy: Float)

    val FRONT_128: Array<Anchor> by lazy { generateFront128() }

    private fun generateFront128(): Array<Anchor> {
        val anchors = ArrayList<Anchor>(896)
        // Layer 1: 16x16 grid, 2 anchors per cell
        appendGrid(anchors, gridSize = 16, anchorsPerCell = 2)
        // Layer 2: 8x8 grid, 6 anchors per cell
        appendGrid(anchors, gridSize = 8, anchorsPerCell = 6)
        return anchors.toTypedArray()
    }

    private fun appendGrid(out: MutableList<Anchor>, gridSize: Int, anchorsPerCell: Int) {
        for (y in 0 until gridSize) {
            for (x in 0 until gridSize) {
                val cx = (x + 0.5f) / gridSize
                val cy = (y + 0.5f) / gridSize
                repeat(anchorsPerCell) {
                    out.add(Anchor(cx, cy))
                }
            }
        }
    }
}
