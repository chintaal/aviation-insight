<script>
	import { onMount } from 'svelte';
	import PirepHoverDisplay from '$lib/components/PirepHoverDisplay.svelte';

	// State management
	let station = 'KATL';
	let distance = 200;
	let age = 1.5;
	let loading = false;
	let error = null;
	let result = null;
	let showRawData = false;

	// Format date for display
	function formatDate(timestamp) {
		if (!timestamp) return 'N/A';
		return new Date(timestamp).toLocaleString();
	}

	// Get color class based on turbulence intensity
	function getTurbulenceClass(intensity) {
		if (!intensity) return 'bg-gray-100 text-gray-800';
		
		const level = intensity.toLowerCase();
		if (level.includes('sev') || level === 'severe') {
			return 'bg-red-100 text-red-800';
		} else if (level.includes('mod') || level === 'moderate') {
			return 'bg-orange-100 text-orange-800';
		} else if (level.includes('lgt') || level === 'light') {
			return 'bg-yellow-100 text-yellow-800';
		} else {
			return 'bg-gray-100 text-gray-800';
		}
	}

	// Get color class based on icing intensity
	function getIcingClass(intensity) {
		if (!intensity) return 'bg-gray-100 text-gray-800';
		
		const level = intensity.toLowerCase();
		if (level.includes('sev') || level === 'severe') {
			return 'bg-purple-100 text-purple-800';
		} else if (level.includes('mod') || level === 'moderate') {
			return 'bg-blue-100 text-blue-800';
		} else if (level.includes('lgt') || level === 'light' || level.includes('trc') || level === 'trace') {
			return 'bg-green-100 text-green-800';
		} else {
			return 'bg-gray-100 text-gray-800';
		}
	}

	// Fetch PIREP data from the API
	async function fetchPirep() {
		loading = true;
		error = null;
		result = null;

		try {
			const url = `/api/v1/pirep/${station}?distance=${distance}&age=${age}`;
			
			const response = await fetch(url);
			
			if (!response.ok) {
				throw new Error(`API error: ${response.status}`);
			}

			result = await response.json();
		} catch (err) {
			error = err.message || 'Failed to fetch PIREP data';
		} finally {
			loading = false;
		}
	}

	// Auto-fetch data on component mount
	onMount(() => {
		fetchPirep();
	});
</script>

<svelte:head>
	<title>PIREP Data - Aviation Weather API Hub</title>
</svelte:head>

<div class="container mx-auto max-w-4xl">
	<div class="mb-6">
		<h1 class="text-2xl font-bold text-gray-800 mb-2">PIREP Data</h1>
		<p class="text-gray-600">
			Search for and analyze Pilot Reports (PIREPs) near a specific airport. PIREPs provide weather observations
			reported by pilots during flight, including information on turbulence, icing, and other hazards.
		</p>
	</div>

	<!-- Request Form -->
	<div class="bg-white rounded-lg shadow p-6 mb-6">
		<h2 class="text-lg font-semibold text-gray-700 mb-4">Request Parameters</h2>
		
		<div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
			<div>
				<label for="station" class="block text-sm font-medium text-gray-700 mb-1">Airport Code</label>
				<input
					type="text"
					id="station"
					bind:value={station}
					placeholder="e.g. KATL, KJFK"
					class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
				/>
			</div>

			<div>
				<label for="distance" class="block text-sm font-medium text-gray-700 mb-1">Distance (nm)</label>
				<input
					type="number"
					id="distance"
					bind:value={distance}
					min="1"
					max="500"
					class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
				/>
			</div>

			<div>
				<label for="age" class="block text-sm font-medium text-gray-700 mb-1">Age (hours)</label>
				<input
					type="number"
					id="age"
					bind:value={age}
					min="0.5"
					max="24"
					step="0.5"
					class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
				/>
			</div>
		</div>

		<div class="flex justify-end">
			<button
				on:click={fetchPirep}
				class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
			>
				{#if loading}
					<svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
						<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
					</svg>
					Fetching...
				{:else}
					Fetch PIREPs
				{/if}
			</button>
		</div>
	</div>

	<!-- API Documentation -->
	<div class="bg-gray-50 rounded-lg border border-gray-200 p-6 mb-6">
		<h2 class="text-lg font-semibold text-gray-700 mb-2">API Endpoint Reference</h2>
		
		<div class="mb-4">
			<div class="bg-gray-800 rounded p-3 overflow-x-auto">
				<code class="text-xs text-green-400">GET /api/v1/pirep/{station}?distance={distance}&age={age}</code>
			</div>
		</div>
		
		<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
			<div>
				<h3 class="text-md font-medium text-gray-700 mb-1">Parameters</h3>
				<table class="min-w-full text-sm">
					<tbody>
						<tr>
							<td class="py-1 font-mono text-xs font-medium">station</td>
							<td class="py-1 pl-4">ICAO airport code (e.g., KATL)</td>
						</tr>
						<tr>
							<td class="py-1 font-mono text-xs font-medium">distance</td>
							<td class="py-1 pl-4">Search radius in nautical miles (default: 200)</td>
						</tr>
						<tr>
							<td class="py-1 font-mono text-xs font-medium">age</td>
							<td class="py-1 pl-4">Maximum age of reports in hours (default: 1.5)</td>
						</tr>
					</tbody>
				</table>
			</div>
			
			<div>
				<h3 class="text-md font-medium text-gray-700 mb-1">Source API</h3>
				<p class="text-sm">
					This endpoint uses the Aviation Weather Center PIREP API:
					<code class="text-xs break-all">https://aviationweather.gov/api/data/pirep?id=KATL&distance=200&age=1.5&format=raw</code>
				</p>
			</div>
		</div>
	</div>

	<!-- Results -->
	{#if error}
		<div class="bg-red-50 border-l-4 border-red-500 p-4 mb-6">
			<div class="flex">
				<div class="flex-shrink-0">
					<svg class="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
					</svg>
				</div>
				<div class="ml-3">
					<h3 class="text-sm font-medium text-red-800">Error</h3>
					<div class="mt-1 text-sm text-red-700">
						{error}
					</div>
				</div>
			</div>
		</div>
	{/if}

	{#if result}
		<div class="bg-white rounded-lg shadow-md overflow-hidden mb-6">
			<div class="px-6 py-4 bg-gray-50 border-b border-gray-200 flex justify-between items-center">
				<h2 class="text-lg font-semibold text-gray-700">PIREP Results</h2>
				{#if Array.isArray(result) && result.length > 0}
					<span class="text-sm text-gray-500">{result.length} reports found</span>
				{/if}
			</div>

			<div class="p-6">
				{#if !Array.isArray(result)}
					<!-- Handle non-array result -->
					<div class="bg-yellow-50 border-l-4 border-yellow-400 p-4">
						<div class="flex">
							<div class="ml-3">
								<p class="text-sm text-yellow-700">
									Unexpected response format. Expected an array of PIREPs.
								</p>
							</div>
						</div>
					</div>
				{:else if result.length === 0}
					<!-- Handle empty array -->
					<div class="text-center py-8">
						<svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
						</svg>
						<h3 class="mt-2 text-sm font-medium text-gray-900">No PIREPs Found</h3>
						<p class="mt-1 text-sm text-gray-500">
							No pilot reports were found matching your search criteria.
						</p>
					</div>
				{:else}
					<!-- Display PIREP results -->
					<div class="space-y-6">
						{#each result as pirep, index}
							<div class="bg-gray-50 p-4 rounded-md border border-gray-200">
								<!-- Interactive PIREP display with hover functionality -->
								<PirepHoverDisplay pirepString={pirep.raw_text} locationCode={pirep.location} />
								
								<!-- Additional PIREP details -->
								<div class="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-gray-200 pt-4">
									<div>
										<h4 class="text-sm font-medium text-gray-700 mb-2">Report Information</h4>
										<table class="w-full text-sm">
											<tbody>
												<tr>
													<td class="py-1 text-gray-600">Location</td>
													<td class="py-1 font-medium">{pirep.location || 'Unknown'}</td>
												</tr>
												{#if pirep.report_type}
													<tr>
														<td class="py-1 text-gray-600">Report Type</td>
														<td class="py-1">
															<span class="bg-gray-100 text-gray-800 text-xs font-medium px-2 py-0.5 rounded">
																{pirep.report_type}
															</span>
														</td>
													</tr>
												{/if}
												{#if pirep.aircraft_type}
													<tr>
														<td class="py-1 text-gray-600">Aircraft</td>
														<td class="py-1">{pirep.aircraft_type}</td>
													</tr>
												{/if}
												{#if pirep.altitude !== undefined}
													<tr>
														<td class="py-1 text-gray-600">Altitude</td>
														<td class="py-1">
															{typeof pirep.altitude === 'string' ? pirep.altitude : `${pirep.altitude} ft`}
														</td>
													</tr>
												{/if}
											</tbody>
										</table>
									</div>
									
									<div>
										<h4 class="text-sm font-medium text-gray-700 mb-2">Weather Hazards</h4>
										<div class="space-y-3">
											{#if pirep.turbulence}
												<div>
													<span class="text-sm font-medium text-gray-600 block mb-1">Turbulence</span>
													<span class={`px-2 py-1 rounded text-xs font-medium ${getTurbulenceClass(pirep.turbulence.intensity)}`}>
														{pirep.turbulence.intensity || 'Reported'}
													</span>
													{#if pirep.turbulence.frequency}
														<span class="text-xs text-gray-500 ml-2">{pirep.turbulence.frequency}</span>
													{/if}
													{#if pirep.turbulence.altitude}
														<span class="text-xs text-gray-500 ml-2">
															at {pirep.turbulence.altitude} ft
														</span>
													{/if}
												</div>
											{/if}
											
											{#if pirep.icing}
												<div>
													<span class="text-sm font-medium text-gray-600 block mb-1">Icing</span>
													<span class={`px-2 py-1 rounded text-xs font-medium ${getIcingClass(pirep.icing.intensity)}`}>
														{pirep.icing.intensity || 'Reported'}
													</span>
													{#if pirep.icing.type}
														<span class="text-xs text-gray-500 ml-2">{pirep.icing.type}</span>
													{/if}
												</div>
											{/if}
											
											{#if pirep.sky_conditions}
												<div>
													<span class="text-sm font-medium text-gray-600 block mb-1">Sky Conditions</span>
													<span class="text-xs">{pirep.sky_conditions}</span>
												</div>
											{/if}
											
											{#if pirep.remarks}
												<div>
													<span class="text-sm font-medium text-gray-600 block mb-1">Remarks</span>
													<span class="text-xs">{pirep.remarks}</span>
												</div>
											{/if}
											
											{#if !pirep.turbulence && !pirep.icing && !pirep.sky_conditions && !pirep.remarks}
												<p class="text-sm text-gray-500">No specific hazards reported</p>
											{/if}
										</div>
									</div>
								</div>
							</div>
						{/each}
					</div>
				{/if}

				<!-- Raw data toggle -->
				<div class="mt-6 pt-4 border-t border-gray-200">
					<button
						on:click={() => showRawData = !showRawData}
						class="text-sm text-blue-600 hover:text-blue-800"
					>
						{showRawData ? 'Hide' : 'Show'} Raw Data
					</button>

					{#if showRawData}
						<div class="mt-2 bg-gray-50 p-3 rounded overflow-x-auto">
							<pre class="text-xs">{JSON.stringify(result, null, 2)}</pre>
						</div>
					{/if}
				</div>
			</div>
		</div>
	{/if}
</div>