package com.alifesoftware.facemesh

import android.app.Application
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
        super.onCreate()
        container = AppContainer(this)
    }
}
