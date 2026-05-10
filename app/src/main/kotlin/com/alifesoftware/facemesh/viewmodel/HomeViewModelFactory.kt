package com.alifesoftware.facemesh.viewmodel

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.alifesoftware.facemesh.di.AppContainer

/**
 * Default factory wiring the [HomeViewModel] to the [AppContainer]. Used by [com.alifesoftware.facemesh.MainActivity].
 *
 * Tests construct the ViewModel directly with mocks instead.
 */
class HomeViewModelFactory(private val container: AppContainer) : ViewModelProvider.Factory {

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        require(modelClass.isAssignableFrom(HomeViewModel::class.java)) {
            "HomeViewModelFactory does not produce ${modelClass.name}"
        }
        Log.i(TAG, "create: wiring HomeViewModel from AppContainer (triggers lazy DB/prefs/store/pipeline)")
        return HomeViewModel(
            clusterRepository = container.clusterRepository,
            preferences = container.preferences,
            modelDownloader = container.modelDownloadManager,
            mlPipelineProvider = container.mlPipelineProvider,
        ) as T
    }

    companion object {
        private const val TAG: String = "FaceMesh.HomeVM"
    }
}
