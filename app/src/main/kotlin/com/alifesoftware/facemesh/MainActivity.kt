package com.alifesoftware.facemesh

import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import com.alifesoftware.facemesh.di.AppContainer
import com.alifesoftware.facemesh.nav.FaceMeshNavHost
import com.alifesoftware.facemesh.ui.theme.FaceMeshTheme
import com.alifesoftware.facemesh.viewmodel.HomeIntent
import com.alifesoftware.facemesh.viewmodel.HomeViewModel
import com.alifesoftware.facemesh.viewmodel.HomeViewModelFactory
import com.alifesoftware.facemesh.viewmodel.SettingsViewModelFactory
import kotlinx.coroutines.launch

/**
 * Single-Activity entry point.
 *
 * Hosts the Compose UI tree and the two photo-picker [androidx.activity.result.ActivityResultLauncher]s.
 * Single-album enforcement (SPEC FR-04) is achieved naturally by allowing one batch per `+`
 * tap; users add more photos by tapping `+` again.
 */
class MainActivity : ComponentActivity() {

    private val container: AppContainer get() = (application as FaceMeshApplication).container
    private val homeVm: HomeViewModel by viewModels { HomeViewModelFactory(container) }
    private val settingsVmFactory: SettingsViewModelFactory by lazy { SettingsViewModelFactory(container) }

    private val pickPhotosForCluster = registerForActivityResult(
        ActivityResultContracts.PickMultipleVisualMedia(maxItems = MAX_CLUSTER_BATCH),
    ) { uris: List<Uri> ->
        Log.i(TAG, "pickPhotosForCluster: result n=${uris.size} (max=$MAX_CLUSTER_BATCH)")
        if (uris.isNotEmpty()) {
            persistUriPermissions(uris)
            homeVm.handle(HomeIntent.PhotosPicked(uris))
        } else {
            Log.i(TAG, "pickPhotosForCluster: user cancelled")
        }
    }

    private val pickPhotosForFilter = registerForActivityResult(
        ActivityResultContracts.PickMultipleVisualMedia(maxItems = HomeViewModel.MAX_FILTER_PHOTOS),
    ) { uris: List<Uri> ->
        Log.i(
            TAG,
            "pickPhotosForFilter: result n=${uris.size} (max=${HomeViewModel.MAX_FILTER_PHOTOS})",
        )
        if (uris.isNotEmpty()) {
            persistUriPermissions(uris)
            homeVm.handle(HomeIntent.FilterPhotosPicked(uris))
        } else {
            Log.i(TAG, "pickPhotosForFilter: user cancelled")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "onCreate: savedInstanceState=${savedInstanceState != null}")
        // Force light system-bar icons against the always-dark "All Black" theme. Using
        // SystemBarStyle.dark() instead of .auto() decouples the system bars from the device
        // day/night setting (the app itself is always dark; see ui.theme.Theme).
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(Color.Transparent.value.toInt()),
            navigationBarStyle = SystemBarStyle.dark(Color.Transparent.value.toInt()),
        )
        super.onCreate(savedInstanceState)
        setContent {
            // Re-renders the entire tree whenever the user toggles the Settings preference.
            val useDynamicColor by container.preferences.dynamicColorEnabled
                .collectAsStateWithLifecycle(initialValue = false)

            FaceMeshTheme(useDynamicColor = useDynamicColor) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    FaceMeshNavHost(
                        homeVm = homeVm,
                        settingsVmFactory = settingsVmFactory,
                        onAddPhotosRequested = ::launchClusterPicker,
                        onPickFilterPhotosRequested = ::launchFilterPicker,
                    )
                }
            }
        }
    }

    override fun onStart() {
        Log.i(TAG, "onStart")
        super.onStart()
    }

    override fun onResume() {
        Log.i(TAG, "onResume")
        super.onResume()
    }

    override fun onPause() {
        Log.i(TAG, "onPause")
        super.onPause()
    }

    override fun onStop() {
        Log.i(TAG, "onStop")
        super.onStop()
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy isFinishing=$isFinishing isChangingConfigurations=$isChangingConfigurations")
        super.onDestroy()
    }

    private fun launchClusterPicker() {
        Log.i(TAG, "launchClusterPicker: opening system photo picker (max=$MAX_CLUSTER_BATCH)")
        pickPhotosForCluster.launch(
            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
        )
    }

    private fun launchFilterPicker() {
        Log.i(TAG, "launchFilterPicker: opening system photo picker (max=${HomeViewModel.MAX_FILTER_PHOTOS})")
        pickPhotosForFilter.launch(
            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
        )
    }

    /**
     * The system Photo Picker grants short-lived URI permissions that survive only for the
     * current process. To make sure embeddings can be re-read on subsequent app launches (e.g.,
     * when persisting representative thumbnails), we attempt to take a persistable read grant.
     * We swallow [SecurityException] because non-`OPEN_DOCUMENT` URIs may not support it.
     */
    private fun persistUriPermissions(uris: List<Uri>) {
        lifecycleScope.launch {
            var ok = 0
            var failed = 0
            uris.forEach { uri ->
                runCatching {
                    contentResolver.takePersistableUriPermission(
                        uri,
                        android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION,
                    )
                }.onSuccess { ok++ }.onFailure {
                    failed++
                    // Most non-OPEN_DOCUMENT URIs throw SecurityException here; that's expected.
                    Log.i(TAG, "persistUriPermissions: not persistable uri=$uri (${it.javaClass.simpleName})")
                }
            }
            Log.i(TAG, "persistUriPermissions: total=${uris.size} persisted=$ok skipped=$failed")
        }
    }

    companion object {
        private const val TAG: String = "FaceMesh.MainActivity"
        // Reasonable cap so the picker doesn't accept a thousand images. Spec is silent on this.
        const val MAX_CLUSTER_BATCH: Int = 100
    }
}
