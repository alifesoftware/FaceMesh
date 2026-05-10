package com.alifesoftware.facemesh.viewmodel

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
        return HomeViewModel(
            clusterRepository = container.clusterRepository,
            preferences = container.preferences,
            modelDownloader = container.modelDownloadManager,
            mlPipelineProvider = container.mlPipelineProvider,
        ) as T
    }
}
