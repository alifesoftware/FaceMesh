package com.alifesoftware.facemesh.ml.download

import com.alifesoftware.facemesh.data.AppPreferences
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Downloads the sidecar model bundle (SPEC \u00a78). Implementation notes:
 *   \u2022 Pure-Kotlin / pure-JDK \u2014 no OkHttp, no Retrofit \u2014 to stay inside APK budget.
 *   \u2022 Atomic writes via temp file + rename so a partial download never corrupts the cache.
 *   \u2022 SHA-256 verification per file before commit.
 *   \u2022 3 attempts with exponential backoff (1s, 4s, 16s) per SPEC FR-30 / \u00a78.2 step 4.
 *   \u2022 Streams progress through a [Flow] so the UI can render a determinate progress bar.
 */
class ModelDownloadManager(
    private val store: ModelStore,
    private val preferences: AppPreferences,
    private val baseUrl: String,
) {

    /**
     * If the bundle is already on disk and valid, completes with [Progress.AlreadyAvailable].
     * Otherwise emits download progress, then [Progress.Done] (or [Progress.Failed]).
     */
    fun ensureAvailable(): Flow<Progress> = flow {
        val cached = store.readManifest()
        if (cached != null && store.allFilesPresent(cached)) {
            emit(Progress.AlreadyAvailable(cached))
            return@flow
        }

        val manifest = downloadManifestWithRetry()
            ?: run { emit(Progress.Failed("manifest")); return@flow }

        emit(Progress.Started(totalBytes = -1))
        val results = mutableListOf<File>()
        for ((index, descriptor) in manifest.models.withIndex()) {
            try {
                val file = downloadModelWithRetry(descriptor) { downloaded, total ->
                    emit(Progress.Downloading(
                        index = index,
                        totalFiles = manifest.models.size,
                        currentBytes = downloaded,
                        totalBytes = total,
                    ))
                }
                results += file
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (e: Exception) {
                emit(Progress.Failed(descriptor.name))
                return@flow
            }
        }

        // Persist manifest only after every model is on disk.
        store.manifestFile.writeText(manifestJson(manifest), Charsets.UTF_8)
        preferences.setModelsVersion(manifest.version)
        preferences.setLastModelsCheck(System.currentTimeMillis())
        // Mirror config defaults into DataStore so downstream components can read tuned values.
        preferences.setDbscanEps(manifest.config.dbscanEps)
        preferences.setDbscanMinPts(manifest.config.dbscanMinPts)
        preferences.setMatchThreshold(manifest.config.matchThreshold)

        emit(Progress.Done(manifest))
    }.flowOn(Dispatchers.IO)

    private suspend fun downloadManifestWithRetry(): ModelManifest? {
        val url = URL(baseUrl + MANIFEST_PATH)
        return withRetry {
            val text = downloadString(url)
            ModelManifest.parse(text)
        }
    }

    private suspend fun downloadModelWithRetry(
        descriptor: ModelDescriptor,
        onProgress: suspend (downloaded: Long, total: Long) -> Unit,
    ): File {
        val url = URL(baseUrl + descriptor.relativeUrl)
        val target = store.fileFor(descriptor)
        return withRetry {
            val tmp = File(target.parentFile, "${target.name}.tmp")
            downloadToFile(url, tmp, onProgress)
            verifySha256(tmp, descriptor.sha256)
            if (!tmp.renameTo(target)) {
                // Fallback: copy + delete (different filesystem)
                tmp.copyTo(target, overwrite = true)
                tmp.delete()
            }
            target
        } ?: throw IOException("Download failed: ${descriptor.name}")
    }

    private suspend fun <T : Any> withRetry(block: suspend () -> T): T? {
        var lastError: Throwable? = null
        for ((attempt, backoff) in BACKOFF_MS.withIndex()) {
            try {
                return block()
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (t: Throwable) {
                lastError = t
                if (attempt < BACKOFF_MS.lastIndex) delay(backoff)
            }
        }
        return null
    }

    private fun downloadString(url: URL): String {
        val conn = (url.openConnection() as HttpURLConnection).apply {
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
        }
        try {
            require(conn.responseCode in 200..299) { "HTTP ${conn.responseCode} for $url" }
            return conn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
        } finally {
            conn.disconnect()
        }
    }

    private suspend fun downloadToFile(
        url: URL,
        target: File,
        onProgress: suspend (downloaded: Long, total: Long) -> Unit,
    ) {
        val conn = (url.openConnection() as HttpURLConnection).apply {
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
        }
        try {
            require(conn.responseCode in 200..299) { "HTTP ${conn.responseCode} for $url" }
            val total = conn.contentLengthLong
            target.parentFile?.mkdirs()
            conn.inputStream.use { input ->
                target.outputStream().use { output ->
                    val buffer = ByteArray(BUFFER_SIZE)
                    var downloaded = 0L
                    var read = input.read(buffer)
                    while (read != -1) {
                        output.write(buffer, 0, read)
                        downloaded += read
                        onProgress(downloaded, total)
                        read = input.read(buffer)
                    }
                }
            }
        } finally {
            conn.disconnect()
        }
    }

    private fun verifySha256(file: File, expected: String) {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(BUFFER_SIZE)
            var read = input.read(buffer)
            while (read != -1) {
                digest.update(buffer, 0, read)
                read = input.read(buffer)
            }
        }
        val actual = digest.digest().joinToString("") { "%02x".format(it) }
        if (!actual.equals(expected, ignoreCase = true)) {
            file.delete()
            throw IOException("SHA-256 mismatch for ${file.name}: expected $expected, got $actual")
        }
    }

    /** Re-emit the manifest as JSON, useful when persisting the version we just verified. */
    private fun manifestJson(manifest: ModelManifest): String {
        val sb = StringBuilder()
        sb.append('{')
        sb.append("\"version\":").append(manifest.version).append(',')
        sb.append("\"models\":[")
        manifest.models.forEachIndexed { index, m ->
            if (index > 0) sb.append(',')
            sb.append('{')
                .append("\"type\":\"").append(m.type).append("\",")
                .append("\"name\":\"").append(m.name).append("\",")
                .append("\"url\":\"").append(m.relativeUrl).append("\",")
                .append("\"sha256\":\"").append(m.sha256).append("\"")
                .append('}')
        }
        sb.append("],")
        val cfg = manifest.config
        sb.append("\"config\":{")
            .append("\"dbscan_eps\":").append(cfg.dbscanEps).append(',')
            .append("\"dbscan_min_pts\":").append(cfg.dbscanMinPts).append(',')
            .append("\"match_threshold\":").append(cfg.matchThreshold).append(',')
            .append("\"detector_input\":[").append(cfg.detectorInput[0]).append(',').append(cfg.detectorInput[1]).append("],")
            .append("\"embedder_input\":[").append(cfg.embedderInput[0]).append(',').append(cfg.embedderInput[1]).append(']')
            .append('}')
        sb.append('}')
        return sb.toString()
    }

    /** Coroutine-friendly progress events. */
    sealed interface Progress {
        data class Started(val totalBytes: Long) : Progress
        data class Downloading(val index: Int, val totalFiles: Int, val currentBytes: Long, val totalBytes: Long) : Progress {
            val approxFraction: Float
                get() {
                    val perFile = if (totalBytes > 0) currentBytes.toFloat() / totalBytes else 0f
                    val base = index.toFloat() / totalFiles
                    return (base + perFile / totalFiles).coerceIn(0f, 1f)
                }
        }
        data class AlreadyAvailable(val manifest: ModelManifest) : Progress
        data class Done(val manifest: ModelManifest) : Progress
        data class Failed(val target: String) : Progress
    }

    companion object {
        const val MANIFEST_PATH: String = "manifest.json"
        const val CONNECT_TIMEOUT_MS: Int = 15_000
        const val READ_TIMEOUT_MS: Int = 30_000
        const val BUFFER_SIZE: Int = 16 * 1024
        val BACKOFF_MS: LongArray = longArrayOf(1_000L, 4_000L, 16_000L)
    }
}

/**
 * Convenience factory used at the call site so we don't have to thread a custom scope/IO
 * dispatcher through tests; can be replaced in tests with a fake.
 */
suspend fun ModelDownloadManager.ensureAvailableSync(): ModelManifest? = coroutineScope {
    var manifest: ModelManifest? = null
    ensureAvailable().collect {
        when (it) {
            is ModelDownloadManager.Progress.AlreadyAvailable -> manifest = it.manifest
            is ModelDownloadManager.Progress.Done -> manifest = it.manifest
            is ModelDownloadManager.Progress.Failed -> manifest = null
            else -> Unit
        }
    }
    manifest
}
