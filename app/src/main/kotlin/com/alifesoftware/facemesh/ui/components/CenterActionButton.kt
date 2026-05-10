package com.alifesoftware.facemesh.ui.components

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.FilterAlt
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.alifesoftware.facemesh.ui.theme.FaceMeshTheme

/** Which icon the centre button currently presents. SPEC \u00a74.4: morphs Camera \u2194 Filter. */
enum class CenterAction { Add, Camera, Filter }

@Composable
fun CenterActionButton(
    action: CenterAction,
    enabled: Boolean,
    onClick: () -> Unit,
    contentDescription: String,
    modifier: Modifier = Modifier,
) {
    FilledIconButton(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier
            .size(72.dp)
            .semantics { this.contentDescription = contentDescription },
        colors = IconButtonDefaults.filledIconButtonColors(
            containerColor = MaterialTheme.colorScheme.primary,
            contentColor = MaterialTheme.colorScheme.onPrimary,
            disabledContainerColor = MaterialTheme.colorScheme.surfaceVariant,
            disabledContentColor = MaterialTheme.colorScheme.onSurfaceVariant,
        ),
    ) {
        AnimatedContent(
            targetState = action,
            transitionSpec = {
                (scaleIn(tween(180), initialScale = 0.6f) + fadeIn(tween(180))) togetherWith
                    (scaleOut(tween(140), targetScale = 0.6f) + fadeOut(tween(140)))
            },
            label = "centerActionMorph",
        ) { current ->
            Icon(
                imageVector = current.icon(),
                contentDescription = null,
                modifier = Modifier.size(32.dp),
            )
        }
    }
}

private fun CenterAction.icon(): ImageVector = when (this) {
    CenterAction.Add -> Icons.Filled.Add
    CenterAction.Camera -> Icons.Filled.PhotoCamera
    CenterAction.Filter -> Icons.Filled.FilterAlt
}

@Preview
@Composable
private fun CenterActionAddPreview() = FaceMeshTheme {
    CenterActionButton(action = CenterAction.Add, enabled = true, onClick = {}, contentDescription = "Add")
}

@Preview
@Composable
private fun CenterActionCameraDisabledPreview() = FaceMeshTheme {
    CenterActionButton(action = CenterAction.Camera, enabled = false, onClick = {}, contentDescription = "Camera")
}

@Preview
@Composable
private fun CenterActionFilterPreview() = FaceMeshTheme {
    CenterActionButton(action = CenterAction.Filter, enabled = true, onClick = {}, contentDescription = "Filter")
}
