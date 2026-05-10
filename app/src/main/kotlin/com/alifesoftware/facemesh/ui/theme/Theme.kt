package com.alifesoftware.facemesh.ui.theme

import android.os.Build
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext

/**
 * Material 3 color scheme derived from the user-supplied "All Black" palette
 * (SPEC \u00a74.4 / OQ-2 resolution). The app is intentionally **always dark**.
 *
 * Two opt-ins control the actual scheme handed to [MaterialTheme]:
 *   1. [useDynamicColor] \u2014 when true on Android 12+ (API 31+), Material You wallpaper-derived
 *      colors take precedence over the brand palette. Persisted via the in-app Settings screen
 *      (`AppPreferences.dynamicColorEnabled`). Default false so that first-launch UX is the
 *      deliberate brand palette.
 *   2. The minSdk floor: on API 29\u201330 the dynamic-color APIs don't exist, so we silently fall
 *      back to the brand palette regardless of the toggle.
 *
 * Mapping to Material 3 roles (only the slots we actually consume in Compose are populated; the
 * rest fall through to `darkColorScheme()` defaults):
 *   primary / onPrimary               = iOS blue / pure white
 *   secondary / onSecondary           = emerald / black
 *   surface / background / surfaceVariant = pure black with a hair of elevation tint
 *   onSurface / onSurfaceVariant      = #F2F2F2 / #A0A0A0
 *   outline / outlineVariant          = #3A3A3C / #232A34 (subtle borders against pure black)
 *   error / onError / errorContainer  = #FF453A
 */
private val FaceMeshAllBlackScheme: ColorScheme = darkColorScheme(
    primary = FaceMeshPrimary,
    onPrimary = FaceMeshOnSurface,
    primaryContainer = FaceMeshPrimary,
    onPrimaryContainer = FaceMeshOnSurface,
    inversePrimary = FaceMeshInfo,

    secondary = FaceMeshSecondary,
    onSecondary = FaceMeshBackground,
    secondaryContainer = FaceMeshSecondary,
    onSecondaryContainer = FaceMeshBackground,

    tertiary = FaceMeshSuccess,
    onTertiary = FaceMeshBackground,

    background = FaceMeshBackground,
    onBackground = FaceMeshOnSurface,

    surface = FaceMeshSurface,
    onSurface = FaceMeshOnSurface,
    surfaceVariant = FaceMeshSurfaceVariant,
    onSurfaceVariant = FaceMeshOnSurfaceMuted,
    surfaceTint = FaceMeshPrimary,

    inverseSurface = FaceMeshOnSurface,
    inverseOnSurface = FaceMeshBackground,

    outline = FaceMeshOutline,
    outlineVariant = FaceMeshElevation,
    scrim = FaceMeshOverlayScrim,

    error = FaceMeshNegative,
    onError = FaceMeshOnSurface,
    errorContainer = FaceMeshNegative,
    onErrorContainer = FaceMeshOnSurface,
)

@Composable
fun FaceMeshTheme(
    useDynamicColor: Boolean = false,
    content: @Composable () -> Unit,
) {
    val colors: ColorScheme = if (useDynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        dynamicDarkColorScheme(LocalContext.current)
    } else {
        FaceMeshAllBlackScheme
    }
    MaterialTheme(
        colorScheme = colors,
        typography = FaceMeshTypography,
        content = content,
    )
}
