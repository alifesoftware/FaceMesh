package com.alifesoftware.facemesh.ui.components

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.SwipeToDismissBox
import androidx.compose.material3.SwipeToDismissBoxValue
import androidx.compose.material3.rememberSwipeToDismissBoxState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.alifesoftware.facemesh.R
import com.alifesoftware.facemesh.domain.model.Cluster
import com.alifesoftware.facemesh.mock.MockData
import com.alifesoftware.facemesh.ui.theme.FaceMeshTheme

private val AvatarSize = 64.dp

@Composable
fun ClusterRow(
    clusters: List<Cluster>,
    selectedIds: Set<String>,
    onToggleSelected: (String) -> Unit,
    onClusterTapped: (String) -> Unit,
    onAddMore: () -> Unit,
    onSwipeDelete: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    LazyRow(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
    ) {
        items(items = clusters, key = { it.id }) { cluster ->
            SwipeableClusterAvatar(
                cluster = cluster,
                checked = cluster.id in selectedIds,
                index = clusters.indexOf(cluster),
                total = clusters.size,
                onCheckedChange = { onToggleSelected(cluster.id) },
                onAvatarTapped = { onClusterTapped(cluster.id) },
                onSwipeDelete = { onSwipeDelete(cluster.id) },
            )
        }
        item(key = "add") {
            AddMoreAvatar(onClick = onAddMore)
        }
    }
}

@Composable
private fun SwipeableClusterAvatar(
    cluster: Cluster,
    checked: Boolean,
    index: Int,
    total: Int,
    onCheckedChange: () -> Unit,
    onAvatarTapped: () -> Unit,
    onSwipeDelete: () -> Unit,
) {
    var requestedDelete by remember(cluster.id) { mutableStateOf(false) }

    val dismissState = rememberSwipeToDismissBoxState(
        confirmValueChange = { value ->
            if (value == SwipeToDismissBoxValue.EndToStart) {
                requestedDelete = true
                false
            } else {
                false
            }
        },
    )

    if (requestedDelete) {
        DeleteClusterDialog(
            faceCount = cluster.faceCount,
            onConfirm = {
                requestedDelete = false
                onSwipeDelete()
            },
            onDismiss = { requestedDelete = false },
        )
    }

    LaunchedEffect(requestedDelete) {
        if (!requestedDelete) {
            dismissState.reset()
        }
    }

    SwipeToDismissBox(
        state = dismissState,
        enableDismissFromStartToEnd = false,
        backgroundContent = {
            Box(
                modifier = Modifier
                    .size(AvatarSize + 16.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.errorContainer),
                contentAlignment = Alignment.Center,
            ) {
                Icon(
                    imageVector = Icons.Filled.Delete,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onErrorContainer,
                )
            }
        },
    ) {
        ClusterAvatar(
            cluster = cluster,
            checked = checked,
            onCheckedChange = onCheckedChange,
            onAvatarTapped = onAvatarTapped,
            index = index,
            total = total,
        )
    }
}

@Composable
private fun ClusterAvatar(
    cluster: Cluster,
    checked: Boolean,
    onCheckedChange: () -> Unit,
    onAvatarTapped: () -> Unit,
    index: Int,
    total: Int,
) {
    val ctx = LocalContext.current
    val haptics = LocalHapticFeedback.current
    val description = ctx.getString(R.string.cd_cluster_avatar, index + 1, total, cluster.faceCount)
    // The 80 dp box must be opaque against MaterialTheme.colorScheme.background, otherwise the
    // SwipeToDismissBox's red `backgroundContent` (the delete affordance) bleeds through the
    // 8 dp transparent ring around the inner 64 dp avatar circle. Painting the box with the
    // theme background colour is invisible at rest but still hides the red until the user
    // actually swipes.
    Box(modifier = Modifier
        .size(AvatarSize + 16.dp)
        .background(MaterialTheme.colorScheme.background)
        .semantics { contentDescription = description }
    ) {
        Surface(
            shape = CircleShape,
            tonalElevation = 2.dp,
            color = MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier
                .size(AvatarSize)
                .align(Alignment.Center)
                // The avatar surface is the "view this cluster's contents" affordance. The
                // overlaid Checkbox in TopStart owns its own click region, so tapping the
                // checkbox does NOT bubble through to here -- selection vs. drill-down stay
                // independent. Tapping anywhere else on the circle opens the gallery.
                .clip(CircleShape)
                .clickable {
                    Log.i("FaceMesh.ClusterRow", "avatar tapped clusterId=${cluster.id}")
                    onAvatarTapped()
                },
        ) {
            AsyncImage(
                model = cluster.representativeImageUri,
                contentDescription = null,
                // Fit (not Crop) keeps the entire saved thumbnail visible inside the circle.
                // Our display crops are square + padded around the face, so Fit and Crop look
                // identical for the common path \u2014 but Fit also rescues the rare fallback case
                // where the saved Uri is the full source photo (non-square, off-centre face).
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .fillMaxSize()
                    .clip(CircleShape),
            )
        }
        Checkbox(
            checked = checked,
            onCheckedChange = {
                haptics.performHapticFeedback(HapticFeedbackType.LongPress)
                onCheckedChange()
            },
            modifier = Modifier
                .align(Alignment.TopStart)
                .size(28.dp),
            colors = CheckboxDefaults.colors(
                checkedColor = MaterialTheme.colorScheme.secondary,
                uncheckedColor = MaterialTheme.colorScheme.outline,
            ),
        )
    }
}

@Composable
private fun AddMoreAvatar(onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(AvatarSize + 16.dp)
            .wrapContentSize(Alignment.Center),
    ) {
        Surface(
            shape = CircleShape,
            tonalElevation = 1.dp,
            color = MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier.size(AvatarSize),
        ) {
            IconButton(onClick = onClick) {
                Icon(
                    imageVector = Icons.Filled.Add,
                    contentDescription = "Add more photos",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Preview(widthDp = 360, heightDp = 120, showBackground = true)
@Composable
private fun ClusterRowPreview() = FaceMeshTheme {
    ClusterRow(
        clusters = MockData.mockClusters,
        selectedIds = setOf("c1"),
        onToggleSelected = {},
        onClusterTapped = {},
        onAddMore = {},
        onSwipeDelete = {},
    )
}

@Suppress("unused")
private val unusedStableColor = Color.Unspecified
