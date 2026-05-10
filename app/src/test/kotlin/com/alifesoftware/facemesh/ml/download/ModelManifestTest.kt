package com.alifesoftware.facemesh.ml.download

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class ModelManifestTest {

    private val v1SampleJson = """
        {
          "version": 1,
          "models": [
            {
              "type": "detector_blazeface_short_range",
              "name": "face_detection_short_range.tflite",
              "url": "face_detection_short_range.tflite",
              "sha256": "3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633"
            },
            {
              "type": "embedder_ghostfacenet_fp16",
              "name": "ghostface_fp16.tflite",
              "url": "ghostface_fp16.tflite",
              "sha256": "4965f3463d0209a298d5409aa4c08d1f7d6d1e4ae15fbd6aafdaf0ab2478fe1c"
            }
          ],
          "config": {
            "dbscan_eps": 0.35,
            "dbscan_min_pts": 2,
            "match_threshold": 0.65,
            "detector_input": [128, 128],
            "embedder_input": [112, 112]
          }
        }
    """.trimIndent()

    private val v2SampleJson = """
        {
          "version": 2,
          "models": [
            {
              "type": "detector_blazeface_short_range",
              "name": "face_detection_short_range.tflite",
              "url": "face_detection_short_range.tflite",
              "sha256": "3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633"
            },
            {
              "type": "detector_blazeface_full_range",
              "name": "face_detection_full_range.tflite",
              "url": "face_detection_full_range.tflite",
              "sha256": "99bf9494d84f50acc6617d89873f71bf6635a841ea699c17cb3377f9507cfec3"
            },
            {
              "type": "embedder_ghostfacenet_fp16",
              "name": "ghostface_fp16.tflite",
              "url": "ghostface_fp16.tflite",
              "sha256": "4965f3463d0209a298d5409aa4c08d1f7d6d1e4ae15fbd6aafdaf0ab2478fe1c"
            }
          ],
          "config": {
            "dbscan_eps": 0.35,
            "dbscan_min_pts": 2,
            "match_threshold": 0.65,
            "detector_short_range_input": [128, 128],
            "detector_full_range_input": [192, 192],
            "embedder_input": [112, 112]
          }
        }
    """.trimIndent()

    @Test
    fun `v2 manifest -- parses both detector entries and split input keys`() {
        val m = ModelManifest.parse(v2SampleJson)
        assertEquals(2, m.version)
        assertEquals(3, m.models.size)

        val sr = m.models.first { it.type == ModelDescriptor.TYPE_DETECTOR_SHORT_RANGE }
        assertEquals("face_detection_short_range.tflite", sr.name)
        val fr = m.models.first { it.type == ModelDescriptor.TYPE_DETECTOR_FULL_RANGE }
        assertEquals("face_detection_full_range.tflite", fr.name)
        val emb = m.models.first { it.type == ModelDescriptor.TYPE_EMBEDDER }
        assertEquals("ghostface_fp16.tflite", emb.name)

        assertEquals(0.35f, m.config.dbscanEps, 1e-6f)
        assertEquals(2, m.config.dbscanMinPts)
        assertEquals(0.65f, m.config.matchThreshold, 1e-6f)
        assertArrayEquals(intArrayOf(128, 128), m.config.shortRangeDetectorInput)
        assertArrayEquals(intArrayOf(192, 192), m.config.fullRangeDetectorInput)
        assertArrayEquals(intArrayOf(112, 112), m.config.embedderInput)
    }

    @Test
    fun `v1 manifest -- short_range falls back to legacy detector_input, full_range to default`() {
        val m = ModelManifest.parse(v1SampleJson)
        assertEquals(1, m.version)
        assertEquals(2, m.models.size)
        // Legacy `detector_input` key is read as the short-range value.
        assertArrayEquals(intArrayOf(128, 128), m.config.shortRangeDetectorInput)
        // Full-range key absent -> fall back to canonical 192x192 default.
        assertArrayEquals(intArrayOf(192, 192), m.config.fullRangeDetectorInput)
    }

    @Test
    fun `manifest exposes both detector type discriminators`() {
        // Sanity check: callers can still rely on the explicit type constants. The deprecated
        // TYPE_DETECTOR alias must equal the short-range type so old call sites resolve.
        @Suppress("DEPRECATION")
        assertEquals(ModelDescriptor.TYPE_DETECTOR_SHORT_RANGE, ModelDescriptor.TYPE_DETECTOR)
    }

    @Test(expected = org.json.JSONException::class)
    fun `manifest entry without type is rejected`() {
        val json = """
            {
              "version": 2,
              "models": [
                {"name": "x.tflite", "url": "x.tflite", "sha256": "abc"}
              ],
              "config": {
                "dbscan_eps": 0.3, "dbscan_min_pts": 2, "match_threshold": 0.6,
                "detector_short_range_input": [128, 128],
                "detector_full_range_input": [192, 192],
                "embedder_input": [112, 112]
              }
            }
        """.trimIndent()
        ModelManifest.parse(json)
    }
}
