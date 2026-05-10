package com.alifesoftware.facemesh.nav

import android.util.Log
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.alifesoftware.facemesh.ui.screens.HomeScreen
import com.alifesoftware.facemesh.ui.screens.KeeperGalleryScreen
import com.alifesoftware.facemesh.ui.screens.SettingsScreen
import com.alifesoftware.facemesh.viewmodel.HomeIntent
import com.alifesoftware.facemesh.viewmodel.HomeViewModel
import com.alifesoftware.facemesh.viewmodel.SettingsViewModel
import com.alifesoftware.facemesh.viewmodel.SettingsViewModelFactory

private const val TAG = "FaceMesh.Nav"

object Routes {
    const val Home: String = "home"
    const val Keepers: String = "keepers/{sessionId}"
    const val Settings: String = "settings"
    fun keepers(sessionId: String): String = "keepers/$sessionId"
}

@Composable
fun FaceMeshNavHost(
    homeVm: HomeViewModel,
    settingsVmFactory: SettingsViewModelFactory,
    onAddPhotosRequested: () -> Unit,
    onPickFilterPhotosRequested: () -> Unit,
) {
    val nav = rememberNavController()
    val keepersBySession by homeVm.keepersBySession.collectAsStateWithLifecycle()

    LaunchedEffect(nav) {
        Log.i(TAG, "host: NavController created startDestination=${Routes.Home}")
        nav.currentBackStackEntryFlow.collect { entry ->
            Log.i(
                TAG,
                "host: backstack entry route=${entry.destination.route} " +
                    "args=${entry.arguments?.keySet()?.toList() ?: emptyList<String>()}",
            )
        }
    }

    NavHost(navController = nav, startDestination = Routes.Home) {
        composable(Routes.Home) {
            DisposableEffect(Unit) {
                Log.i(TAG, "route enter: Home")
                onDispose { Log.i(TAG, "route exit:  Home") }
            }
            HomeScreen(
                viewModel = homeVm,
                onNavigateToKeepers = { sessionId ->
                    Log.i(TAG, "Home -> Keepers(sessionId=$sessionId)")
                    nav.navigate(Routes.keepers(sessionId))
                },
                onNavigateToSettings = {
                    Log.i(TAG, "Home -> Settings")
                    nav.navigate(Routes.Settings)
                },
                onAddPhotosRequested = onAddPhotosRequested,
                onPickFilterPhotosRequested = onPickFilterPhotosRequested,
            )
        }
        composable(
            route = Routes.Keepers,
            arguments = listOf(navArgument("sessionId") { type = NavType.StringType }),
        ) { entry ->
            val sessionId = entry.arguments?.getString("sessionId").orEmpty()
            val keepers = remember(sessionId, keepersBySession) {
                keepersBySession[sessionId].orEmpty()
            }
            DisposableEffect(sessionId) {
                Log.i(TAG, "route enter: Keepers sessionId=$sessionId items=${keepers.size}")
                onDispose { Log.i(TAG, "route exit:  Keepers sessionId=$sessionId") }
            }
            KeeperGalleryScreen(
                keepers = keepers,
                onBack = {
                    Log.i(TAG, "Keepers <- back popping to Home")
                    homeVm.handle(HomeIntent.ReturnedFromKeepers)
                    nav.popBackStack()
                },
            )
        }
        composable(Routes.Settings) {
            DisposableEffect(Unit) {
                Log.i(TAG, "route enter: Settings")
                onDispose { Log.i(TAG, "route exit:  Settings") }
            }
            val settingsVm: SettingsViewModel = viewModel(factory = settingsVmFactory)
            SettingsScreen(
                viewModel = settingsVm,
                onBack = {
                    Log.i(TAG, "Settings <- back popping to Home")
                    nav.popBackStack()
                },
            )
        }
    }
}
