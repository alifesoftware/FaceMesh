package com.alifesoftware.facemesh

import android.app.Application
import android.os.Build
import android.os.SystemClock
import android.util.Log
import com.alifesoftware.facemesh.di.AppContainer

/**
 * Application entry point.
 *
 * Hosts the manually-wired [AppContainer] so we can provide singletons (DB, DataStore, ML runtime)
 * without pulling in Hilt or Koin and inflating the APK budget. Activities access it via
 * `(application as FaceMeshApplication).container`.
 */
class FaceMeshApplication : Application() {

    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        val started = SystemClock.elapsedRealtime()
        Log.i(
            TAG,
            "onCreate: starting FaceMesh process pid=${android.os.Process.myPid()} " +
                "appId=${BuildConfig.APPLICATION_ID} version=${BuildConfig.VERSION_NAME} " +
                "build=${BuildConfig.BUILD_TYPE} debug=${BuildConfig.DEBUG} " +
                "sdk=${Build.VERSION.SDK_INT} device=${Build.MANUFACTURER} ${Build.MODEL} " +
                "abis=${Build.SUPPORTED_ABIS.toList()}",
        )
        super.onCreate()
        container = AppContainer(this)
        Log.i(
            TAG,
            "onCreate: AppContainer constructed in ${SystemClock.elapsedRealtime() - started}ms " +
                "(lazy holders only — heavy graph builds on first access)",
        )
    }

    override fun onLowMemory() {
        Log.w(TAG, "onLowMemory: system reports low memory pressure")
        super.onLowMemory()
    }

    override fun onTrimMemory(level: Int) {
        Log.w(TAG, "onTrimMemory: level=$level")
        super.onTrimMemory(level)
    }

    companion object {
        private const val TAG: String = "FaceMesh.App"
    }
}
