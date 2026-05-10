package com.alifesoftware.facemesh.ui.components

import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.key
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import coil.compose.SubcomposeAsyncImage
import com.alifesoftware.facemesh.mock.MockData
import com.alifesoftware.facemesh.ui.theme.FaceMeshTheme
import android.net.Uri

/**
 * SPEC \u00a74.2.2: a "fan" of the most recent 3\u20134 thumbnails sitting behind the centre action
 * button, each rotated slightly. Newest is on top of the stack.
 */
@Composable
fun ThumbnailFan(
    uris: List<Uri>,
    modifier: Modifier = Modifier,
    tileSize: Dp = 80.dp,
) {
    // Don't reserve any layout space when there's nothing to render. Otherwise the empty-state
    // hero (the centre `+` button) gets visually pushed downward by the fan's idle frame.
    if (uris.isEmpty()) return
    val density = LocalDensity.current
    Box(modifier = modifier.size(tileSize * 1.6f, tileSize * 1.4f), contentAlignment = Alignment.Center) {
        // Render oldest to newest so newest is on top.
        uris.asReversed().forEachIndexed { displayIndex, uri ->
            val depthFromTop = uris.size - 1 - displayIndex
            val angle = ((depthFromTop - (uris.size - 1) / 2f) * 8f).coerceIn(-16f, 16f)
            val xOffset = (depthFromTop - (uris.size - 1) / 2f) * 6f
            key(uri) {
                AnimatedVisibility(
                    visible = true,
                    enter = scaleIn(initialScale = 0.8f, animationSpec = tween(220)) +
                        fadeIn(animationSpec = tween(220)),
                    exit = scaleOut(targetScale = 0.8f, animationSpec = tween(160)) +
                        fadeOut(animationSpec = tween(160)),
                ) {
                    Surface(
                        shape = RoundedCornerShape(16.dp),
                        tonalElevation = 4.dp,
                        modifier = Modifier
                            .size(tileSize)
                            .graphicsLayer {
                                rotationZ = angle
                                translationX = with(density) { xOffset.dp.toPx() }
                            }
                            .shadow(elevation = (4 + depthFromTop * 2).dp, shape = RoundedCornerShape(16.dp)),
                    ) {
                        SubcomposeAsyncImage(
                            model = uri,
                            contentDescription = null,
                            contentScale = ContentScale.Crop,
                            modifier = Modifier
                                .clip(RoundedCornerShape(16.dp))
                                .size(tileSize),
                        )
                    }
                }
            }
        }
    }
}

@Preview(widthDp = 320, heightDp = 200, showBackground = true)
@Composable
private fun ThumbnailFanPreview() {
    FaceMeshTheme {
        Box(modifier = Modifier.size(width = 320.dp, height = 200.dp), contentAlignment = Alignment.Center) {
            ThumbnailFan(uris = MockData.photosBatchOne.take(4))
        }
    }
}
