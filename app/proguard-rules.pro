# --- FaceMesh ProGuard / R8 rules ---
# Goal: keep APK growth from face-recognition logic <500 KB.
# Use full mode (configured in gradle.properties via android.enableR8.fullMode default true).

# Strip log/debug
-assumenosideeffects class android.util.Log {
    public static *** v(...);
    public static *** d(...);
    public static *** i(...);
}

# Keep our entity/data classes used by Room (KSP-generated DAOs reflect on these)
-keep class com.alifesoftware.facemesh.data.** { *; }

# Room base – minimal keep
-keep class androidx.room.** { *; }
-keep @androidx.room.Entity class * { *; }
-keep @androidx.room.Database class * { *; }
-keep @androidx.room.Dao class * { *; }

# TFLite Play Services / GMS
-keep class com.google.android.gms.tflite.** { *; }
-dontwarn com.google.android.gms.**

# Kotlin reflection / metadata
-keep class kotlin.Metadata { *; }

# Coil
-keep class coil.** { *; }
-dontwarn coil.**

# Compose / navigation – generally handled by AndroidX consumer rules

# Strip annotation noise
-keepattributes *Annotation*
