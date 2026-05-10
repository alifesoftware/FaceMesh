package com.alifesoftware.facemesh.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.alifesoftware.facemesh.data.AppPreferences
import com.alifesoftware.facemesh.di.AppContainer
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch

/**
 * Backs the [com.alifesoftware.facemesh.ui.screens.SettingsScreen] with read/write access to
 * the in-app preference flags.
 */
class SettingsViewModel(
    private val preferences: AppPreferences,
) : ViewModel() {

    val dynamicColorEnabled: Flow<Boolean> = preferences.dynamicColorEnabled

    fun setDynamicColorEnabled(value: Boolean) {
        viewModelScope.launch { preferences.setDynamicColorEnabled(value) }
    }
}

class SettingsViewModelFactory(private val container: AppContainer) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        require(modelClass.isAssignableFrom(SettingsViewModel::class.java)) {
            "SettingsViewModelFactory does not produce ${modelClass.name}"
        }
        return SettingsViewModel(preferences = container.preferences) as T
    }
}
