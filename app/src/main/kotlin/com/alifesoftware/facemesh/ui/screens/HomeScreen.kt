package com.alifesoftware.facemesh.ui.screens

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.FilterAlt
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.Face
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FabPosition
import androidx.compose.material3.FloatingActionButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.alifesoftware.facemesh.R
import com.alifesoftware.facemesh.mock.MockData
import com.alifesoftware.facemesh.ui.components.ClusterRow
import com.alifesoftware.facemesh.ui.components.FilterImagesStrip
import com.alifesoftware.facemesh.ui.components.ProcessingOverlay
import com.alifesoftware.facemesh.ui.components.ResetConfirmationDialog
import com.alifesoftware.facemesh.ui.components.ThumbnailFan
import com.alifesoftware.facemesh.ui.theme.FaceMeshTheme
import com.alifesoftware.facemesh.viewmodel.HomeIntent
import com.alifesoftware.facemesh.viewmodel.HomeUiState
import com.alifesoftware.facemesh.viewmodel.HomeViewModel
import com.alifesoftware.facemesh.viewmodel.UiMessage
import kotlinx.coroutines.flow.collectLatest

// Bottom inset reserved beneath body content so a centered ExtendedFAB never overlaps the
// last interactive element (cluster row, filter strip, etc.). Roughly: FAB height (56) +
// breathing room above & below.
private val FAB_BOTTOM_INSET = 96.dp

@Composable
fun HomeScreen(
    viewModel: HomeViewModel = viewModel(),
    onNavigateToKeepers: (String) -> Unit,
    onNavigateToSettings: () -> Unit,
    onAddPhotosRequested: () -> Unit,
    onPickFilterPhotosRequested: () -> Unit,
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val ctx = LocalContext.current
    val snackbarHost = remember { SnackbarHostState() }

    LaunchedEffect(viewModel) {
        viewModel.messages.collectLatest { msg ->
            when (msg) {
                UiMessage.NoFacesFound ->
                    snackbarHost.showSnackbar(ctx.getString(R.string.toast_no_faces_found), duration = SnackbarDuration.Short)
                UiMessage.NoMatchesFound ->
                    snackbarHost.showSnackbar(ctx.getString(R.string.toast_no_matches_found), duration = SnackbarDuration.Short)
                UiMessage.UsingCpuAcceleration ->
                    snackbarHost.showSnackbar(ctx.getString(R.string.toast_using_cpu), duration = SnackbarDuration.Short)
                UiMessage.MaxFilterImagesReached ->
                    snackbarHost.showSnackbar(ctx.getString(R.string.toast_max_filter_images), duration = SnackbarDuration.Short)
                UiMessage.ModelDownloadFailed ->
                    snackbarHost.showSnackbar(ctx.getString(R.string.toast_model_download_failed), duration = SnackbarDuration.Long)
                is UiMessage.NavigateToKeepers -> onNavigateToKeepers(msg.sessionId)
            }
        }
    }

    var showResetDialog by remember { mutableStateOf(false) }
    if (showResetDialog) {
        ResetConfirmationDialog(
            onConfirm = {
                showResetDialog = false
                viewModel.handle(HomeIntent.ResetConfirmed)
            },
            onDismiss = { showResetDialog = false },
        )
    }

    HomeScaffold(
        state = state,
        snackbarHost = snackbarHost,
        onAddPhotos = onAddPhotosRequested,
        onPickFilterPhotos = onPickFilterPhotosRequested,
        onClusterify = { viewModel.handle(HomeIntent.ClusterifyTapped) },
        onFilter = { viewModel.handle(HomeIntent.FilterTapped) },
        onClear = { viewModel.handle(HomeIntent.ClearFilterTapped) },
        onReset = { showResetDialog = true },
        onSettings = onNavigateToSettings,
        onToggleCluster = { viewModel.handle(HomeIntent.ToggleClusterChecked(it)) },
        onSwipeDeleteCluster = { viewModel.handle(HomeIntent.DeleteClusterConfirmed(it)) },
        onCancelProcessing = { viewModel.handle(HomeIntent.CancelProcessingTapped) },
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun HomeScaffold(
    state: HomeUiState,
    snackbarHost: SnackbarHostState,
    onAddPhotos: () -> Unit,
    onPickFilterPhotos: () -> Unit,
    onClusterify: () -> Unit,
    onFilter: () -> Unit,
    onClear: () -> Unit,
    onReset: () -> Unit,
    onSettings: () -> Unit,
    onToggleCluster: (String) -> Unit,
    onSwipeDeleteCluster: (String) -> Unit,
    onCancelProcessing: () -> Unit,
) {
    Scaffold(
        modifier = Modifier.fillMaxSize(),
        snackbarHost = { SnackbarHost(hostState = snackbarHost) },
        topBar = {
            HomeTopBar(
                state = state,
                onClear = onClear,
                onReset = onReset,
                onSettings = onSettings,
            )
        },
        floatingActionButton = {
            HomeFab(
                state = state,
                onAddPhotos = onAddPhotos,
                onClusterify = onClusterify,
                onPickFilterPhotos = onPickFilterPhotos,
                onFilter = onFilter,
            )
        },
        floatingActionButtonPosition = FabPosition.Center,
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            HomeBody(
                state = state,
                onAddPhotos = onAddPhotos,
                onToggleCluster = onToggleCluster,
                onSwipeDeleteCluster = onSwipeDeleteCluster,
            )

            when (state) {
                is HomeUiState.Processing -> ProcessingOverlay(
                    captionResId = R.string.processing_photos,
                    processed = state.processed,
                    total = state.total,
                    onCancel = onCancelProcessing,
                    downloadFraction = state.downloadFraction,
                )
                is HomeUiState.Matching -> ProcessingOverlay(
                    captionResId = R.string.filtering_photos,
                    processed = state.processed,
                    total = state.total,
                    onCancel = onCancelProcessing,
                )
                else -> Unit
            }
        }
    }
}

// ────────────────────────── Top app bar ──────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun HomeTopBar(
    state: HomeUiState,
    onClear: () -> Unit,
    onReset: () -> Unit,
    onSettings: () -> Unit,
) {
    val showClear = state is HomeUiState.FilterReady
    val showReset = state !is HomeUiState.Empty
    CenterAlignedTopAppBar(
        title = {
            Text(
                text = stringResource(R.string.app_name),
                style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.W700),
                modifier = Modifier.semantics { contentDescription = "FaceMesh" },
            )
        },
        actions = {
            if (showClear) {
                TextButton(onClick = onClear) { Text(stringResource(R.string.action_clear)) }
            }
            if (showReset) {
                TextButton(onClick = onReset) { Text(stringResource(R.string.action_reset)) }
            }
            IconButton(onClick = onSettings) {
                Icon(
                    imageVector = Icons.Filled.Settings,
                    contentDescription = stringResource(R.string.cd_open_settings),
                )
            }
        },
        colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
            containerColor = MaterialTheme.colorScheme.background,
            titleContentColor = MaterialTheme.colorScheme.onSurface,
            actionIconContentColor = MaterialTheme.colorScheme.onSurface,
        ),
    )
}

// ────────────────────────── Floating action button ──────────────────────────

private data class FabSpec(
    val label: String,
    val icon: ImageVector,
    val onClick: () -> Unit,
    val enabled: Boolean,
)

@Composable
private fun HomeFab(
    state: HomeUiState,
    onAddPhotos: () -> Unit,
    onClusterify: () -> Unit,
    onPickFilterPhotos: () -> Unit,
    onFilter: () -> Unit,
) {
    val spec: FabSpec = when (state) {
        is HomeUiState.Empty -> FabSpec(
            label = stringResource(R.string.action_add_photos),
            icon = Icons.Filled.Add,
            onClick = onAddPhotos,
            enabled = true,
        )
        is HomeUiState.Selecting -> FabSpec(
            label = stringResource(R.string.action_clusterify),
            icon = Icons.Filled.AutoAwesome,
            onClick = onClusterify,
            enabled = state.selectedPhotos.isNotEmpty(),
        )
        is HomeUiState.Clustered -> FabSpec(
            label = stringResource(R.string.action_pick_filter_photos),
            icon = Icons.Filled.PhotoCamera,
            onClick = onPickFilterPhotos,
            enabled = state.hasAnySelected,
        )
        is HomeUiState.FilterReady -> FabSpec(
            label = stringResource(R.string.action_run_filter),
            icon = Icons.Filled.FilterAlt,
            onClick = onFilter,
            enabled = state.canFilter,
        )
        is HomeUiState.Processing,
        is HomeUiState.Matching -> return
    }

    val container =
        if (spec.enabled) MaterialTheme.colorScheme.primary
        else MaterialTheme.colorScheme.surfaceVariant
    val content =
        if (spec.enabled) MaterialTheme.colorScheme.onPrimary
        else MaterialTheme.colorScheme.onSurfaceVariant

    ExtendedFloatingActionButton(
        onClick = { if (spec.enabled) spec.onClick() },
        containerColor = container,
        contentColor = content,
        elevation = FloatingActionButtonDefaults.elevation(
            defaultElevation = if (spec.enabled) 6.dp else 0.dp,
            pressedElevation = if (spec.enabled) 8.dp else 0.dp,
        ),
        modifier = Modifier.semantics { contentDescription = spec.label },
    ) {
        AnimatedContent(
            targetState = spec.icon to spec.label,
            transitionSpec = {
                (scaleIn(tween(180), initialScale = 0.7f) + fadeIn(tween(180))) togetherWith
                    (scaleOut(tween(140), targetScale = 0.7f) + fadeOut(tween(140)))
            },
            label = "homeFabMorph",
        ) { (icon, label) ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(imageVector = icon, contentDescription = null)
                Spacer(Modifier.width(12.dp))
                Text(text = label, fontWeight = FontWeight.W600)
            }
        }
    }
}

// ────────────────────────── Body per state ──────────────────────────

@Composable
private fun HomeBody(
    state: HomeUiState,
    onAddPhotos: () -> Unit,
    onToggleCluster: (String) -> Unit,
    onSwipeDeleteCluster: (String) -> Unit,
) {
    when (state) {
        is HomeUiState.Empty -> EmptyBody()
        is HomeUiState.Selecting -> SelectingBody(state = state, onAddPhotos = onAddPhotos)
        is HomeUiState.Clustered -> ClusteredBody(
            clusters = state.clusters,
            selectedIds = state.selectedClusterIds,
            filterPhotos = emptyList(),
            onAddPhotos = onAddPhotos,
            onToggleCluster = onToggleCluster,
            onSwipeDeleteCluster = onSwipeDeleteCluster,
        )
        is HomeUiState.FilterReady -> ClusteredBody(
            clusters = state.clusters,
            selectedIds = state.selectedClusterIds,
            filterPhotos = state.filterPhotos,
            onAddPhotos = onAddPhotos,
            onToggleCluster = onToggleCluster,
            onSwipeDeleteCluster = onSwipeDeleteCluster,
        )
        is HomeUiState.Matching -> ClusteredBody(
            clusters = state.clusters,
            selectedIds = state.selectedClusterIds,
            filterPhotos = state.filterPhotos,
            onAddPhotos = onAddPhotos,
            onToggleCluster = onToggleCluster,
            onSwipeDeleteCluster = onSwipeDeleteCluster,
        )
        is HomeUiState.Processing -> Box(modifier = Modifier.fillMaxSize())
    }
}

@Composable
private fun EmptyBody() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 32.dp, vertical = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Icon(
            imageVector = Icons.Outlined.Face,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.size(96.dp),
        )
        Spacer(Modifier.height(16.dp))
        Text(
            text = stringResource(R.string.empty_state_headline),
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.onSurface,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = stringResource(R.string.app_tagline),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(FAB_BOTTOM_INSET))
    }
}

@Composable
private fun SelectingBody(
    state: HomeUiState.Selecting,
    onAddPhotos: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Spacer(Modifier.weight(1f))
        ThumbnailFan(uris = state.recentFan)
        Spacer(Modifier.height(16.dp))
        Text(
            text = stringResource(R.string.photos_selected_count, state.selectedPhotos.size),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(4.dp))
        TextButton(onClick = onAddPhotos) {
            Text(stringResource(R.string.action_add_more_photos))
        }
        Spacer(Modifier.weight(1f))
        Spacer(Modifier.height(FAB_BOTTOM_INSET))
    }
}

@Composable
private fun ClusteredBody(
    clusters: List<com.alifesoftware.facemesh.domain.model.Cluster>,
    selectedIds: Set<String>,
    filterPhotos: List<android.net.Uri>,
    onAddPhotos: () -> Unit,
    onToggleCluster: (String) -> Unit,
    onSwipeDeleteCluster: (String) -> Unit,
) {
    Column(modifier = Modifier.fillMaxSize()) {
        Spacer(Modifier.weight(1f))

        if (filterPhotos.isNotEmpty()) {
            Text(
                text = stringResource(R.string.photos_selected_count, filterPhotos.size),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 24.dp, bottom = 4.dp),
            )
            FilterImagesStrip(uris = filterPhotos)
            Spacer(Modifier.height(16.dp))
        }

        ClusterRow(
            clusters = clusters,
            selectedIds = selectedIds,
            onToggleSelected = onToggleCluster,
            onAddMore = onAddPhotos,
            onSwipeDelete = onSwipeDeleteCluster,
        )
        Spacer(Modifier.height(FAB_BOTTOM_INSET))
    }
}

// ────────────────────────── Previews per state ──────────────────────────

@Preview(name = "Empty", showBackground = true, widthDp = 360, heightDp = 720)
@Composable
private fun PreviewEmpty() = FaceMeshTheme {
    HomeScaffold(
        state = HomeUiState.Empty,
        snackbarHost = remember { SnackbarHostState() },
        onAddPhotos = {}, onPickFilterPhotos = {}, onClusterify = {}, onFilter = {},
        onClear = {}, onReset = {}, onSettings = {}, onToggleCluster = {},
        onSwipeDeleteCluster = {}, onCancelProcessing = {},
    )
}

@Preview(name = "Selecting", showBackground = true, widthDp = 360, heightDp = 720)
@Composable
private fun PreviewSelecting() = FaceMeshTheme {
    val photos = MockData.photosBatchOne
    HomeScaffold(
        state = HomeUiState.Selecting(selectedPhotos = photos, recentFan = photos.takeLast(4).reversed()),
        snackbarHost = remember { SnackbarHostState() },
        onAddPhotos = {}, onPickFilterPhotos = {}, onClusterify = {}, onFilter = {},
        onClear = {}, onReset = {}, onSettings = {}, onToggleCluster = {},
        onSwipeDeleteCluster = {}, onCancelProcessing = {},
    )
}

@Preview(name = "Clustered", showBackground = true, widthDp = 360, heightDp = 720)
@Composable
private fun PreviewClustered() = FaceMeshTheme {
    HomeScaffold(
        state = HomeUiState.Clustered(clusters = MockData.mockClusters, selectedClusterIds = setOf("c1", "c3")),
        snackbarHost = remember { SnackbarHostState() },
        onAddPhotos = {}, onPickFilterPhotos = {}, onClusterify = {}, onFilter = {},
        onClear = {}, onReset = {}, onSettings = {}, onToggleCluster = {},
        onSwipeDeleteCluster = {}, onCancelProcessing = {},
    )
}

@Preview(name = "FilterReady", showBackground = true, widthDp = 360, heightDp = 720)
@Composable
private fun PreviewFilterReady() = FaceMeshTheme {
    HomeScaffold(
        state = HomeUiState.FilterReady(
            clusters = MockData.mockClusters,
            selectedClusterIds = setOf("c1"),
            filterPhotos = MockData.photosBatchTwo,
        ),
        snackbarHost = remember { SnackbarHostState() },
        onAddPhotos = {}, onPickFilterPhotos = {}, onClusterify = {}, onFilter = {},
        onClear = {}, onReset = {}, onSettings = {}, onToggleCluster = {},
        onSwipeDeleteCluster = {}, onCancelProcessing = {},
    )
}
