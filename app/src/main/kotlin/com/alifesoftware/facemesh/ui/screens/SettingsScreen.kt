package com.alifesoftware.facemesh.ui.screens

import android.os.Build
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.ListItemDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import android.util.Log
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.alifesoftware.facemesh.R
import com.alifesoftware.facemesh.viewmodel.SettingsViewModel

private const val SCREEN_TAG: String = "FaceMesh.SettingsScreen"

@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel,
    onBack: () -> Unit,
) {
    val dynamicColorEnabled by viewModel.dynamicColorEnabled.collectAsStateWithLifecycle(initialValue = false)
    val supportsDynamicColor = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S

    DisposableEffect(Unit) {
        Log.i(
            SCREEN_TAG,
            "compose: entered supportsDynamicColor=$supportsDynamicColor (sdk=${Build.VERSION.SDK_INT})",
        )
        onDispose { Log.i(SCREEN_TAG, "compose: disposed") }
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = {
            CenterAlignedTopAppBar(
                title = { Text(stringResource(R.string.settings_title)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(R.string.action_back),
                        )
                    }
                },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                ),
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            verticalArrangement = Arrangement.Top,
            horizontalAlignment = Alignment.Start,
        ) {
            Text(
                text = stringResource(R.string.settings_section_appearance),
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
            )
            ListItem(
                headlineContent = { Text(stringResource(R.string.settings_dynamic_color_title)) },
                supportingContent = {
                    Text(
                        text = if (supportsDynamicColor) {
                            stringResource(R.string.settings_dynamic_color_subtitle)
                        } else {
                            stringResource(R.string.settings_dynamic_color_unavailable)
                        },
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                },
                trailingContent = {
                    Switch(
                        checked = dynamicColorEnabled && supportsDynamicColor,
                        enabled = supportsDynamicColor,
                        onCheckedChange = {
                            Log.i(SCREEN_TAG, "switch: dynamicColor toggled -> $it")
                            viewModel.setDynamicColorEnabled(it)
                        },
                    )
                },
                modifier = Modifier.fillMaxWidth(),
                colors = ListItemDefaults.colors(
                    containerColor = MaterialTheme.colorScheme.background,
                ),
            )
        }
    }
}
