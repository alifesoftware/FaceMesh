package com.alifesoftware.facemesh.nav

import androidx.compose.runtime.Composable
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

    NavHost(navController = nav, startDestination = Routes.Home) {
        composable(Routes.Home) {
            HomeScreen(
                viewModel = homeVm,
                onNavigateToKeepers = { sessionId -> nav.navigate(Routes.keepers(sessionId)) },
                onNavigateToSettings = { nav.navigate(Routes.Settings) },
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
            KeeperGalleryScreen(
                keepers = keepers,
                onBack = {
                    homeVm.handle(HomeIntent.ReturnedFromKeepers)
                    nav.popBackStack()
                },
            )
        }
        composable(Routes.Settings) {
            val settingsVm: SettingsViewModel = viewModel(factory = settingsVmFactory)
            SettingsScreen(
                viewModel = settingsVm,
                onBack = { nav.popBackStack() },
            )
        }
    }
}
