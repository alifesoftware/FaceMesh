package com.alifesoftware.facemesh.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.alifesoftware.facemesh.R

@Composable
fun ProcessingOverlay(
    captionResId: Int,
    processed: Int,
    total: Int,
    onCancel: () -> Unit,
    modifier: Modifier = Modifier,
    /**
     * If non-null, the overlay shows a model-download caption + determinate progress bar
     * instead of the per-image caption. SPEC \u00a78.2 step 3.
     */
    downloadFraction: Float? = null,
) {
    val ctx = LocalContext.current
    Box(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.scrim)
            .pointerInput(Unit) { /* swallow taps */ },
        contentAlignment = Alignment.Center,
    ) {
        Surface(
            shape = RoundedCornerShape(24.dp),
            color = MaterialTheme.colorScheme.surface,
            tonalElevation = 6.dp,
            modifier = Modifier.padding(32.dp),
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                CircularProgressIndicator(modifier = Modifier.size(48.dp))
                if (downloadFraction != null) {
                    Text(
                        text = stringResource(R.string.downloading_models),
                        style = MaterialTheme.typography.bodyLarge,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(top = 16.dp),
                    )
                    LinearProgressIndicator(
                        progress = { downloadFraction.coerceIn(0f, 1f) },
                        modifier = Modifier
                            .padding(top = 12.dp)
                            .width(180.dp),
                    )
                } else {
                    Text(
                        text = ctx.getString(captionResId, processed, total),
                        style = MaterialTheme.typography.bodyLarge,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(top = 16.dp),
                    )
                    if (total > 0) {
                        LinearProgressIndicator(
                            progress = { (processed.toFloat() / total).coerceIn(0f, 1f) },
                            modifier = Modifier
                                .padding(top = 12.dp)
                                .width(180.dp),
                        )
                    }
                }
                TextButton(
                    onClick = onCancel,
                    modifier = Modifier.padding(top = 8.dp),
                ) { Text(stringResource(R.string.action_cancel)) }
            }
        }
    }
}
