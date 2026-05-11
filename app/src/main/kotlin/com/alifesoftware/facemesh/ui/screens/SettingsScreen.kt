package com.alifesoftware.facemesh.ui.screens

import android.os.Build
import android.util.Log
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.ListItemDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.alifesoftware.facemesh.R
import com.alifesoftware.facemesh.config.PipelineConfig
import com.alifesoftware.facemesh.config.PipelineConfig.Detector.DetectorVariant
import com.alifesoftware.facemesh.data.AppPreferences
import com.alifesoftware.facemesh.viewmodel.SettingsViewModel

private const val SCREEN_TAG: String = "FaceMesh.SettingsScreen"

@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel,
    onBack: () -> Unit,
) {
    val dynamicColorEnabled by viewModel.dynamicColorEnabled.collectAsStateWithLifecycle(initialValue = false)
    val supportsDynamicColor = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S

    val dbscanEps by viewModel.dbscanEps.collectAsStateWithLifecycle(
        initialValue = viewModel.dbscanEpsBounds.min,
    )
    val dbscanEpsSource by viewModel.dbscanEpsSource.collectAsStateWithLifecycle(
        initialValue = AppPreferences.Source.DEFAULT,
    )
    val dbscanEpsModified by viewModel.dbscanEpsHasUserOverride.collectAsStateWithLifecycle(initialValue = false)

    val matchThreshold by viewModel.matchThreshold.collectAsStateWithLifecycle(
        initialValue = viewModel.matchThresholdBounds.min,
    )
    val matchThresholdSource by viewModel.matchThresholdSource.collectAsStateWithLifecycle(
        initialValue = AppPreferences.Source.DEFAULT,
    )
    val matchThresholdModified by viewModel.matchThresholdHasUserOverride.collectAsStateWithLifecycle(
        initialValue = false,
    )

    val detectorVariant by viewModel.detectorVariant.collectAsStateWithLifecycle(
        initialValue = PipelineConfig.Detector.defaultVariant,
    )

    val incrementalMergeIntoExisting by viewModel.incrementalMergeIntoExisting.collectAsStateWithLifecycle(
        initialValue = false,
    )

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
                .padding(padding)
                .verticalScroll(rememberScrollState()),
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

            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

            // -------- Pipeline tuning section --------
            Text(
                text = stringResource(R.string.settings_section_pipeline),
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
            )
            Text(
                text = stringResource(R.string.settings_pipeline_section_caption),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
            )

            TunedValueSlider(
                title = stringResource(R.string.settings_dbscan_eps_title),
                subtitle = stringResource(R.string.settings_dbscan_eps_subtitle),
                value = dbscanEps,
                bounds = viewModel.dbscanEpsBounds,
                source = dbscanEpsSource,
                isModified = dbscanEpsModified,
                onValueChange = { viewModel.setDbscanEpsUserOverride(it) },
                onReset = { viewModel.resetDbscanEpsUserOverride() },
            )
            TunedValueSlider(
                title = stringResource(R.string.settings_match_threshold_title),
                subtitle = stringResource(R.string.settings_match_threshold_subtitle),
                value = matchThreshold,
                bounds = viewModel.matchThresholdBounds,
                source = matchThresholdSource,
                isModified = matchThresholdModified,
                onValueChange = { viewModel.setMatchThresholdUserOverride(it) },
                onReset = { viewModel.resetMatchThresholdUserOverride() },
            )

            ClusterifyPersistenceSection(
                incrementalMergeEnabled = incrementalMergeIntoExisting,
                onIncrementalMergeChange = { viewModel.setIncrementalMergeIntoExisting(it) },
            )

            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

            // -------- Face detection model section --------
            DetectorVariantSection(
                current = detectorVariant,
                onSelect = { viewModel.setDetectorVariant(it) },
            )
        }
    }
}

@Composable
private fun TunedValueSlider(
    title: String,
    subtitle: String,
    value: Float,
    bounds: SettingsViewModel.Bounds,
    source: AppPreferences.Source,
    isModified: Boolean,
    onValueChange: (Float) -> Unit,
    onReset: () -> Unit,
) {
    // Use a local state that follows the resolved value but lets drag updates feel snappy. The
    // ViewModel write per onValueChange persists every step; remembering locally avoids the
    // momentary flicker that happens when the DataStore round-trip lags the gesture.
    var sliderValue by remember(value) { mutableStateOf(value) }
    val sourceLabel = when (source) {
        AppPreferences.Source.USER -> stringResource(R.string.settings_source_user)
        AppPreferences.Source.MANIFEST -> stringResource(R.string.settings_source_manifest)
        AppPreferences.Source.DEFAULT -> stringResource(R.string.settings_source_default)
    }
    val displayValue = stringResource(R.string.settings_value_with_source, value, sourceLabel)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
            )
            Text(
                text = displayValue,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Text(
            text = subtitle,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 4.dp),
        )
        // Material3 Slider's `steps` param is the number of *intermediate* discrete stops
        // between min and max - matching SettingsViewModel.Bounds.sliderSteps.
        Slider(
            value = sliderValue,
            onValueChange = { newValue ->
                sliderValue = newValue
                onValueChange(snapToStep(newValue, bounds))
            },
            valueRange = bounds.min..bounds.max,
            steps = bounds.sliderSteps,
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 8.dp),
        )
        if (isModified) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                TextButton(onClick = onReset) {
                    Text(stringResource(R.string.action_reset))
                }
            }
        }
    }
}

/** Snap the slider's continuous float to the nearest configured step. */
private fun snapToStep(raw: Float, bounds: SettingsViewModel.Bounds): Float {
    if (bounds.step <= 0f) return raw
    val snapped = bounds.min + ((raw - bounds.min) / bounds.step).toInt() * bounds.step
    return snapped.coerceIn(bounds.min, bounds.max)
}

@Composable
private fun ClusterifyPersistenceSection(
    incrementalMergeEnabled: Boolean,
    onIncrementalMergeChange: (Boolean) -> Unit,
) {
    Text(
        text = stringResource(R.string.settings_clusterify_persistence_title),
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
    )
    Text(
        text = stringResource(R.string.settings_clusterify_persistence_subtitle),
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
    )
    val options = listOf(
        false to R.string.settings_clusterify_segment_batch_only,
        true to R.string.settings_clusterify_segment_merge_saved,
    )
    SingleChoiceSegmentedButtonRow(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
    ) {
        options.forEachIndexed { index, (incremental, labelRes) ->
            SegmentedButton(
                selected = incrementalMergeEnabled == incremental,
                onClick = {
                    Log.i(
                        SCREEN_TAG,
                        "segmented: clusterifyPersistence incrementalMerge=$incremental",
                    )
                    onIncrementalMergeChange(incremental)
                },
                shape = SegmentedButtonDefaults.itemShape(index = index, count = options.size),
            ) {
                Text(stringResource(labelRes))
            }
        }
    }
}

@Composable
private fun DetectorVariantSection(
    current: DetectorVariant,
    onSelect: (DetectorVariant) -> Unit,
) {
    Text(
        text = stringResource(R.string.settings_section_detector),
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
    )
    Text(
        text = stringResource(R.string.settings_detector_caption),
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
    )
    val options: List<Pair<DetectorVariant, Pair<Int, Int>>> = listOf(
        DetectorVariant.SHORT_RANGE to (R.string.settings_detector_short_range_label to R.string.settings_detector_short_range_subtitle),
        DetectorVariant.FULL_RANGE  to (R.string.settings_detector_full_range_label  to R.string.settings_detector_full_range_subtitle),
    )
    SingleChoiceSegmentedButtonRow(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
    ) {
        options.forEachIndexed { index, (variant, labels) ->
            SegmentedButton(
                selected = variant == current,
                onClick = {
                    Log.i(SCREEN_TAG, "segmented: detectorVariant -> $variant")
                    onSelect(variant)
                },
                shape = SegmentedButtonDefaults.itemShape(index = index, count = options.size),
            ) {
                Text(stringResource(labels.first))
            }
        }
    }
    val activeSubtitleRes = options.first { it.first == current }.second.second
    Text(
        text = stringResource(activeSubtitleRes),
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
    )
}
