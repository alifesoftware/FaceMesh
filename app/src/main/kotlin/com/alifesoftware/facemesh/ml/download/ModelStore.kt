package com.alifesoftware.facemesh.ml.download

import android.content.Context
import java.io.File

/**
 * Filesystem layout for downloaded sidecar models. Single source of truth so other components
 * (TfLiteRuntime, FaceDetector, FaceEmbedder) can resolve paths consistently.
 */
class ModelStore(context: Context) {

    val modelsDir: File = File(context.filesDir, MODELS_SUBDIR).apply { mkdirs() }
    val manifestFile: File = File(modelsDir, MANIFEST_FILENAME)

    fun fileFor(descriptor: ModelDescriptor): File = File(modelsDir, descriptor.name)

    fun manifestPresent(): Boolean = manifestFile.exists()

    fun readManifest(): ModelManifest? = if (manifestPresent()) {
        runCatching { ModelManifest.parse(manifestFile.readText(Charsets.UTF_8)) }.getOrNull()
    } else null

    fun allFilesPresent(manifest: ModelManifest): Boolean =
        manifest.models.all { fileFor(it).exists() && fileFor(it).length() > 0 }

    fun clear() {
        modelsDir.listFiles()?.forEach { it.delete() }
    }

    companion object {
        const val MODELS_SUBDIR: String = "models"
        const val MANIFEST_FILENAME: String = "manifest.json"
    }
}
