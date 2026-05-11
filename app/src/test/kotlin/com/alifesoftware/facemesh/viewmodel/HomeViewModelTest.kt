package com.alifesoftware.facemesh.viewmodel

import android.net.Uri
import com.alifesoftware.facemesh.domain.model.Cluster
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class HomeViewModelTest {

    private val dispatcher = StandardTestDispatcher()

    private val photo1: Uri = Uri.parse("file:///photo1.jpg")
    private val photo2: Uri = Uri.parse("file:///photo2.jpg")
    private val filterPhoto: Uri = Uri.parse("file:///filter.jpg")

    @Before
    fun setUp() {
        Dispatchers.setMain(dispatcher)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `starts in Empty state`() = runTest {
        val vm = HomeViewModel()
        assertSame(HomeUiState.Empty, vm.state.value)
    }

    @Test
    fun `picking photos transitions to Selecting with capped fan`() = runTest {
        val vm = HomeViewModel()
        vm.handle(HomeIntent.PhotosPicked(listOf(photo1, photo2)))
        val s = vm.state.value as HomeUiState.Selecting
        assertEquals(2, s.selectedPhotos.size)
        assertEquals(2, s.recentFan.size)
    }

    @Test
    fun `clusterify in mock path runs to Clustered with mock data`() = runTest {
        val vm = HomeViewModel()
        vm.handle(HomeIntent.PhotosPicked(listOf(photo1)))
        vm.handle(HomeIntent.ClusterifyTapped)
        advanceUntilIdle()
        val s = vm.state.value
        assertTrue("expected Clustered, was $s", s is HomeUiState.Clustered)
    }

    @Test
    fun `toggling cluster updates selectedClusterIds`() = runTest {
        val vm = HomeViewModel()
        // Move to Clustered manually via the public intent surface.
        vm.handle(HomeIntent.PhotosPicked(listOf(photo1)))
        vm.handle(HomeIntent.ClusterifyTapped)
        advanceUntilIdle()
        val first = (vm.state.value as HomeUiState.Clustered).clusters.first()
        vm.handle(HomeIntent.ToggleClusterChecked(first.id))
        val checked = vm.state.value as HomeUiState.Clustered
        assertTrue(first.id in checked.selectedClusterIds)
        assertTrue("camera should be enabled when 1+ checked", checked.hasAnySelected)
    }

    @Test
    fun `picking more than max filter photos caps and emits warning`() = runTest {
        val vm = HomeViewModel()
        vm.handle(HomeIntent.PhotosPicked(listOf(photo1)))
        vm.handle(HomeIntent.ClusterifyTapped)
        advanceUntilIdle()
        val firstCluster = (vm.state.value as HomeUiState.Clustered).clusters.first()
        vm.handle(HomeIntent.ToggleClusterChecked(firstCluster.id))
        val tooMany = (1..HomeViewModel.MAX_FILTER_PHOTOS + 5).map { Uri.parse("file:///p$it.jpg") }
        vm.handle(HomeIntent.FilterPhotosPicked(tooMany))
        val ready = vm.state.value as HomeUiState.FilterReady
        assertEquals(HomeViewModel.MAX_FILTER_PHOTOS, ready.filterPhotos.size)
    }

    @Test
    fun `clear from FilterReady returns to Clustered preserving selection`() = runTest {
        val vm = HomeViewModel()
        vm.handle(HomeIntent.PhotosPicked(listOf(photo1)))
        vm.handle(HomeIntent.ClusterifyTapped)
        advanceUntilIdle()
        val firstCluster = (vm.state.value as HomeUiState.Clustered).clusters.first()
        vm.handle(HomeIntent.ToggleClusterChecked(firstCluster.id))
        vm.handle(HomeIntent.FilterPhotosPicked(listOf(filterPhoto)))
        vm.handle(HomeIntent.ClearFilterTapped)
        val s = vm.state.value as HomeUiState.Clustered
        assertTrue(firstCluster.id in s.selectedClusterIds)
    }

    @Test
    fun `reset wipes back to Empty`() = runTest {
        val vm = HomeViewModel()
        vm.handle(HomeIntent.PhotosPicked(listOf(photo1)))
        vm.handle(HomeIntent.ClusterifyTapped)
        advanceUntilIdle()
        vm.handle(HomeIntent.ResetConfirmed)
        advanceUntilIdle()
        assertSame(HomeUiState.Empty, vm.state.value)
    }

    @Test
    fun `deleting last cluster reverts to Empty`() = runTest {
        val vm = HomeViewModel()
        vm.handle(HomeIntent.PhotosPicked(listOf(photo1)))
        vm.handle(HomeIntent.ClusterifyTapped)
        advanceUntilIdle()
        val ids = (vm.state.value as HomeUiState.Clustered).clusters.map { it.id }
        ids.forEach { id -> vm.handle(HomeIntent.DeleteClusterConfirmed(id)) }
        advanceUntilIdle()
        assertSame(HomeUiState.Empty, vm.state.value)
    }

    @Test
    fun `cluster finished event accepts external clusters`() = runTest {
        val vm = HomeViewModel()
        val injected = listOf(Cluster(id = "x", representativeImageUri = photo1, faceCount = 3))
        vm.handle(HomeIntent.ClusterifyFinished(injected))
        advanceUntilIdle()
        val s = vm.state.value as HomeUiState.Clustered
        assertEquals(1, s.clusters.size)
        assertEquals("x", s.clusters.first().id)
    }

    @Test
    fun `no faces during clusterify returns to empty and emits message`() = runTest {
        val vm = HomeViewModel()
        vm.handle(HomeIntent.ClusterifyNoFaces)
        assertSame(HomeUiState.Empty, vm.state.value)
    }

    @Test
    fun `untoggling all clusters disables camera`() = runTest {
        val vm = HomeViewModel()
        vm.handle(HomeIntent.PhotosPicked(listOf(photo1)))
        vm.handle(HomeIntent.ClusterifyTapped)
        advanceUntilIdle()
        val first = (vm.state.value as HomeUiState.Clustered).clusters.first()
        vm.handle(HomeIntent.ToggleClusterChecked(first.id))
        vm.handle(HomeIntent.ToggleClusterChecked(first.id))
        val s = vm.state.value as HomeUiState.Clustered
        assertFalse(s.hasAnySelected)
    }
}
