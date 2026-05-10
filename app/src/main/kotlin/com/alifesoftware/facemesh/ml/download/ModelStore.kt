package com.alifesoftware.facemesh.ml.download

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Filesystem layout for downloaded sidecar models. Single source of truth so other components
 * (TfLiteRuntime, FaceDetector, FaceEmbedder) can resolve paths consistently.
 */
class ModelStore(context: Context) {

    val modelsDir: File = File(context.filesDir, MODELS_SUBDIR).apply { mkdirs() }
    val manifestFile: File = File(modelsDir, MANIFEST_FILENAME)

    init {
        Log.i(
            TAG,
            "init: modelsDir=${modelsDir.absolutePath} manifest=${manifestFile.absolutePath} " +
                "manifestExists=${manifestFile.exists()}",
        )
    }

    fun fileFor(descriptor: ModelDescriptor): File {
        val f = File(modelsDir, descriptor.name)
        Log.i(
            TAG,
            "fileFor: type=${descriptor.type} name=${descriptor.name} -> ${f.absolutePath} " +
                "exists=${f.exists()} size=${if (f.exists()) f.length() else 0L}",
        )
        return f
    }

    fun manifestPresent(): Boolean {
        val present = manifestFile.exists()
        Log.i(TAG, "manifestPresent: $present")
        return present
    }

    fun readManifest(): ModelManifest? {
        if (!manifestPresent()) {
            Log.i(TAG, "readManifest: file missing -> null")
            return null
        }
        return runCatching {
            val text = manifestFile.readText(Charsets.UTF_8)
            Log.i(TAG, "readManifest: parsing ${text.length} bytes from ${manifestFile.absolutePath}")
            ModelManifest.parse(text)
        }.fold(
            onSuccess = { m ->
                Log.i(
                    TAG,
                    "readManifest: parsed v${m.version} models=${m.models.map { it.name }} " +
                        "config={dbscanEps=${m.config.dbscanEps}, minPts=${m.config.dbscanMinPts}, " +
                        "matchTh=${m.config.matchThreshold}, det=${m.config.detectorInput.toList()}, " +
                        "emb=${m.config.embedderInput.toList()}}",
                )
                m
            },
            onFailure = {
                Log.w(TAG, "readManifest: parse failed; treating as missing", it)
                null
            },
        )
    }

    fun allFilesPresent(manifest: ModelManifest): Boolean {
        val results = manifest.models.map { d ->
            val f = File(modelsDir, d.name)
            Triple(d.name, f.exists(), if (f.exists()) f.length() else 0L)
        }
        val ok = results.all { it.second && it.third > 0 }
        Log.i(
            TAG,
            "allFilesPresent: $ok " + results.joinToString(", ") { "${it.first}(exists=${it.second}, size=${it.third})" },
        )
        return ok
    }

    fun clear() {
        val files = modelsDir.listFiles()
        Log.w(TAG, "clear: deleting ${files?.size ?: 0} file(s) in ${modelsDir.absolutePath}")
        files?.forEach {
            val deleted = it.delete()
            Log.i(TAG, "clear: ${it.name} deleted=$deleted")
        }
    }

    companion object {
        private const val TAG: String = "FaceMesh.Store"
        const val MODELS_SUBDIR: String = "models"
        const val MANIFEST_FILENAME: String = "manifest.json"
    }
}
