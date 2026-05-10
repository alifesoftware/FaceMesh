package com.alifesoftware.facemesh.ml.download

import org.json.JSONObject

/**
 * Pure-Kotlin representation of the sidecar bundle manifest (SPEC \u00a78.1).
 *
 * Hand-rolled JSON parsing avoids pulling in Gson / kotlinx-serialization which would push us
 * outside the 500 KB APK budget.
 */
data class ModelManifest(
    val version: Int,
    val models: List<ModelDescriptor>,
    val config: ModelConfig,
) {
    companion object {
        fun parse(json: String): ModelManifest {
            val root = JSONObject(json)
            val modelsArray = root.getJSONArray("models")
            val models = (0 until modelsArray.length()).map { i ->
                val obj = modelsArray.getJSONObject(i)
                ModelDescriptor(
                    type = obj.getString("type"),
                    name = obj.getString("name"),
                    relativeUrl = obj.getString("url"),
                    sha256 = obj.getString("sha256"),
                )
            }
            val configObj = root.getJSONObject("config")
            val inputDet = configObj.getJSONArray("detector_input")
            val inputEmb = configObj.getJSONArray("embedder_input")
            val config = ModelConfig(
                dbscanEps = configObj.getDouble("dbscan_eps").toFloat(),
                dbscanMinPts = configObj.getInt("dbscan_min_pts"),
                matchThreshold = configObj.getDouble("match_threshold").toFloat(),
                detectorInput = intArrayOf(inputDet.getInt(0), inputDet.getInt(1)),
                embedderInput = intArrayOf(inputEmb.getInt(0), inputEmb.getInt(1)),
            )
            return ModelManifest(
                version = root.getInt("version"),
                models = models,
                config = config,
            )
        }
    }
}

data class ModelDescriptor(
    /**
     * Semantic role this artifact fulfils. Stable across re-exports / re-naming so that
     * [com.alifesoftware.facemesh.ml.MlPipelineProvider] can resolve the right file even if
     * the upstream filename changes. Known values: [TYPE_DETECTOR], [TYPE_EMBEDDER].
     */
    val type: String,
    /** On-disk filename; used by [com.alifesoftware.facemesh.ml.download.ModelStore]. */
    val name: String,
    /** URL relative to `BuildConfig.MODEL_BASE_URL`. */
    val relativeUrl: String,
    val sha256: String,
) {
    companion object {
        const val TYPE_DETECTOR: String = "detector_blazeface_short_range"
        const val TYPE_EMBEDDER: String = "embedder_ghostfacenet_fp16"
    }
}

data class ModelConfig(
    val dbscanEps: Float,
    val dbscanMinPts: Int,
    val matchThreshold: Float,
    val detectorInput: IntArray,
    val embedderInput: IntArray,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ModelConfig) return false
        if (dbscanEps != other.dbscanEps) return false
        if (dbscanMinPts != other.dbscanMinPts) return false
        if (matchThreshold != other.matchThreshold) return false
        if (!detectorInput.contentEquals(other.detectorInput)) return false
        if (!embedderInput.contentEquals(other.embedderInput)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = dbscanEps.hashCode()
        result = 31 * result + dbscanMinPts
        result = 31 * result + matchThreshold.hashCode()
        result = 31 * result + detectorInput.contentHashCode()
        result = 31 * result + embedderInput.contentHashCode()
        return result
    }
}
