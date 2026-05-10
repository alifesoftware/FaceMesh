package com.alifesoftware.facemesh.ui.theme

import androidx.compose.ui.graphics.Color

// "All Black" OLED-friendly dark palette (SPEC \u00a74.4 / OQ-2 resolution).
// Source palette comes from the user's React app and is mapped onto Material 3 roles below in
// `Theme.kt`. We keep the raw token names here so the mapping is auditable and the palette can
// be regenerated from a JSON if it ever moves.

// \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 brand accents \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
val FaceMeshPrimary       = Color(0xFF007AFF) // iOS-style blue \u2014 maps to MaterialTheme.colorScheme.primary
val FaceMeshSecondary     = Color(0xFF10B981) // emerald          \u2014 maps to MaterialTheme.colorScheme.secondary
val FaceMeshInfo          = Color(0xFF0A84FF)
val FaceMeshSuccess       = Color(0xFF32D74B)
val FaceMeshWarning       = Color(0xFFFF9F0A) // (palette had warning == negative; we tone it down for warning)
val FaceMeshNegative      = Color(0xFFFF453A) // = error / notification
val FaceMeshNeutral       = Color(0xFF98989D)

// \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 neutrals \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
val FaceMeshBackground    = Color(0xFF000000) // background, card, surface, gradientStart/End \u2014 all black
val FaceMeshSurface       = Color(0xFF000000)
val FaceMeshSurfaceVariant = Color(0xFF1C1C1E) // ever-so-slight elevation tint for grouped controls
val FaceMeshElevation     = Color(0xFF232A34) // = rangeSelectorBg; useful for processed surfaces

val FaceMeshOnSurface     = Color(0xFFF2F2F2) // text
val FaceMeshOnSurfaceMuted = Color(0xFFA0A0A0) // subtext
val FaceMeshOutline       = Color(0xFF3A3A3C) // borders are 'invisible' against pure black; this is the next-step-up neutral for stroke / divider use

// \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 scrims / overlays \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
val FaceMeshOverlayScrim  = Color(0x99000000) // = rgba(0,0,0,0.6)
