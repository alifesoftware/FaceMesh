package com.alifesoftware.facemesh.ml.download

import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class ModelManifestTest {

    private val sampleJson = """
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

    @Test
    fun `parses production manifest with type discriminators`() {
        val m = ModelManifest.parse(sampleJson)
        assertEquals(1, m.version)
        assertEquals(2, m.models.size)

        val det = m.models.first { it.type == ModelDescriptor.TYPE_DETECTOR }
        assertEquals("face_detection_short_range.tflite", det.name)
        assertEquals(
            "3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633",
            det.sha256,
        )

        val emb = m.models.first { it.type == ModelDescriptor.TYPE_EMBEDDER }
        assertEquals("ghostface_fp16.tflite", emb.name)
        assertEquals(
            "4965f3463d0209a298d5409aa4c08d1f7d6d1e4ae15fbd6aafdaf0ab2478fe1c",
            emb.sha256,
        )

        assertEquals(0.35f, m.config.dbscanEps, 1e-6f)
        assertEquals(2, m.config.dbscanMinPts)
        assertEquals(0.65f, m.config.matchThreshold, 1e-6f)
        assertEquals(128, m.config.detectorInput[0])
        assertEquals(112, m.config.embedderInput[1])
    }

    @Test(expected = org.json.JSONException::class)
    fun `manifest entry without type is rejected`() {
        val json = """
            {
              "version": 1,
              "models": [
                {"name": "x.tflite", "url": "x.tflite", "sha256": "abc"}
              ],
              "config": {
                "dbscan_eps": 0.3, "dbscan_min_pts": 2, "match_threshold": 0.6,
                "detector_input": [128,128], "embedder_input": [112,112]
              }
            }
        """.trimIndent()
        ModelManifest.parse(json)
    }
}
