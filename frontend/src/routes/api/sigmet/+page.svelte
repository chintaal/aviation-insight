<script>
	import { onMount } from 'svelte';

	// State management
	let region = 'all';
	let source = 'awc';
	let showMap = true;
	let productType = 'sigmet'; // 'sigmet' or 'airmet'
	let loading = false;
	let error = null;
	let result = null;
	let showRawData = false;

	// Available regions
	const regions = [
		{ id: 'all', name: 'All Regions' },
		{ id: 'conus', name: 'Continental US' },
		{ id: 'pacific', name: 'Pacific' },
		{ id: 'atlantic', name: 'Atlantic' }
	];

	// Available data sources
	const sources = [
		{ id: 'awc', name: 'Aviation Weather Center (AWC)' },
		{ id: 'avwx', name: 'AVWX API' }
	];

	// Format date for display
	function formatDate(timestamp) {
		if (!timestamp) return 'N/A';
		return new Date(timestamp).toLocaleString();
	}

	// Format validity period
	function formatValidityPeriod(from, to) {
		if (!from || !to) return 'N/A';
		return `${formatDate(from)} to ${formatDate(to)}`;
	}

	// Get color class based on phenomenon
	function getPhenomenonClass(phenomenon) {
		if (!phenomenon) return 'bg-gray-100 text-gray-800';
		
		const type = phenomenon.toLowerCase();
		if (type.includes('turb') || type === 'turbulence') {
			return 'bg-orange-100 text-orange-800';
		} else if (type.includes('ice') || type === 'icing') {
			return 'bg-blue-100 text-blue-800';
		} else if (type.includes('mtw') || type === 'mountain wave') {
			return 'bg-purple-100 text-purple-800';
		} else if (type.includes('ifr') || type === 'ifr') {
			return 'bg-red-100 text-red-800';
		} else if (type.includes('convective') || type === 'convective') {
			return 'bg-yellow-100 text-yellow-800';
		} else if (type.includes('ash') || type === 'volcanic ash') {
			return 'bg-gray-700 text-gray-100';
		} else {
			return 'bg-gray-100 text-gray-800';
		}
	}

	// Toggle product type
	function toggleProductType(type) {
		productType = type;
		fetchData();
	}

	// Fetch data from the API
	async function fetchData() {
		loading = true;
		error = null;
		result = null;

		try {
			let url = `/api/v1/${productType}?region=${region}&source=${source}`;
			
			const response = await fetch(url);
			
			if (!response.ok) {
				throw new Error(`API error: ${response.status}`);
			}

			result = await response.json();
		} catch (err) {
			error = err.message || `Failed to fetch ${productType.toUpperCase()} data`;
		} finally {
			loading = false;
		}
	}

	// Auto-fetch data on component mount
	onMount(() => {
		fetchData();
	});
</script>

<svelte:head>
	<title>SIGMET/AIRMET API Testing - Aviation Weather API Hub</title>
</svelte:head>

<div class="container mx-auto max-w-4xl">
	<div class="mb-6">
		<h1 class="text-2xl font-bold text-gray-800 mb-2">SIGMET/AIRMET API Testing</h1>
		<p class="text-gray-600">
			Test the API endpoints for fetching SIGMETs (Significant Meteorological Information) and 
			AIRMETs (Airmen's Meteorological Information) which are issued as advisories to warn pilots of potentially 
			hazardous weather conditions.
		</p>
	</div>

	<!-- Product Type Toggle -->
	<div class="flex justify-center mb-6 bg-white rounded-lg overflow-hidden shadow-sm">
		<button
			on:click={() => toggleProductType('sigmet')}
			class={`flex-1 py-3 px-6 text-sm font-medium ${
				productType === 'sigmet'
					? 'bg-blue-600 text-white'
					: 'bg-white text-gray-600 hover:bg-gray-50'
			}`}
		>
			SIGMET
		</button>
		<button
			on:click={() => toggleProductType('airmet')}
			class={`flex-1 py-3 px-6 text-sm font-medium ${
				productType === 'airmet'
					? 'bg-blue-600 text-white'
					: 'bg-white text-gray-600 hover:bg-gray-50'
			}`}
		>
			AIRMET
		</button>
	</div>

	<!-- Request Form -->
	<div class="bg-white rounded-lg shadow p-6 mb-6">
		<h2 class="text-lg font-semibold text-gray-700 mb-4">Request Parameters</h2>
		
		<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
			<div>
				<label for="region" class="block text-sm font-medium text-gray-700 mb-1">Region</label>
				<select
					id="region"
					bind:value={region}
					class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
				>
					{#each regions as reg}
						<option value={reg.id}>{reg.name}</option>
					{/each}
				</select>
			</div>

			<div>
				<label for="source" class="block text-sm font-medium text-gray-700 mb-1">Data Source</label>
				<select
					id="source"
					bind:value={source}
					class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
				>
					{#each sources as src}
						<option value={src.id}>{src.name}</option>
					{/each}
				</select>
			</div>
		</div>

		<div class="flex justify-end">
			<button
				on:click={fetchData}
				class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
			>
				{#if loading}
					<svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
						<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
					</svg>
					Fetching...
				{:else}
					Fetch {productType.toUpperCase()}s
				{/if}
			</button>
		</div>
	</div>

	<!-- API Documentation -->
	<div class="bg-gray-50 rounded-lg border border-gray-200 p-6 mb-6">
		<h2 class="text-lg font-semibold text-gray-700 mb-2">API Endpoint Reference</h2>
		
		<div class="mb-4">
			<h3 class="text-md font-medium text-gray-700">SIGMET Endpoint</h3>
			<div class="bg-gray-800 rounded p-3 overflow-x-auto mb-2">
				<code class="text-xs text-green-400">GET /api/v1/sigmet?region={region}&source={source}</code>
			</div>
		</div>
		
		<div class="mb-4">
			<h3 class="text-md font-medium text-gray-700">AIRMET Endpoint</h3>
			<div class="bg-gray-800 rounded p-3 overflow-x-auto mb-2">
				<code class="text-xs text-green-400">GET /api/v1/airmet?region={region}&source={source}</code>
			</div>
		</div>
		
		<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
			<div>
				<h3 class="text-md font-medium text-gray-700 mb-1">Parameters</h3>
				<table class="min-w-full text-sm">
					<tbody>
						<tr>
							<td class="py-1 font-mono text-xs font-medium">region</td>
							<td class="py-1 pl-4">Region: 'all', 'conus', 'pacific', or 'atlantic'</td>
						</tr>
						<tr>
							<td class="py-1 font-mono text-xs font-medium">source</td>
							<td class="py-1 pl-4">Data source: 'awc' or 'avwx' (default: awc)</td>
						</tr>
					</tbody>
				</table>
			</div>
			
			<div>
				<h3 class="text-md font-medium text-gray-700 mb-1">Response Format</h3>
				<div class="bg-gray-800 rounded p-3 overflow-x-auto">
				{@html `<pre class="text-xs text-blue-400"><code>[
  {
    "source": "AWC",
    "id": "WSJC31",
    "raw_text": "WSJC31 KJAX 231245\\nSIGMET JULIET 1...",
    "phenomenon": "CONVECTIVE",
    "valid_from": "2025-04-23T12:45:00Z",
    "valid_to": "2025-04-23T16:45:00Z",
    "area": [
      {"lat": 30.5, "lon": -81.2},
      {"lat": 31.5, "lon": -80.5},
      /* ... */
    ],
    "altitude": {
      "lower": 10000,
      "upper": 35000
    }
  }
]</code></pre>`}
				</div>
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
				<h2 class="text-lg font-semibold text-gray-700">{productType.toUpperCase()} Results</h2>
				{#if Array.isArray(result) && result.length > 0}
					<span class="text-sm text-gray-500">{result.length} advisories found</span>
				{/if}
			</div>

			<div class="p-6">
				{#if !Array.isArray(result)}
					<!-- Handle non-array result -->
					<div class="bg-yellow-50 border-l-4 border-yellow-400 p-4">
						<div class="flex">
							<div class="ml-3">
								<p class="text-sm text-yellow-700">
									Unexpected response format. Expected an array of advisories.
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
						<h3 class="mt-2 text-sm font-medium text-gray-900">No Advisories Found</h3>
						<p class="mt-1 text-sm text-gray-500">
							No {productType.toUpperCase()}s were found for the selected region.
						</p>
					</div>
				{:else}
					<!-- Display results -->
					<div class="space-y-6">
						{#each result as advisory, index}
							<div class="bg-gray-50 p-4 rounded-md border border-gray-200">
								<div class="flex flex-wrap justify-between items-center mb-3">
									<div class="flex items-center space-x-2 mb-2 sm:mb-0">
										<span class="font-medium text-gray-700">Source:</span>
										<span class="bg-blue-100 text-blue-800 text-xs font-medium px-2 py-0.5 rounded">
											{advisory.source}
										</span>
										
										{#if advisory.id}
											<span class="font-medium text-gray-700">ID:</span>
											<span class="font-mono text-xs bg-gray-100 px-2 py-0.5 rounded">
												{advisory.id}
											</span>
										{/if}
									</div>
									
									{#if advisory.phenomenon}
										<span class={`px-2 py-1 rounded text-xs font-medium ${getPhenomenonClass(advisory.phenomenon)}`}>
											{advisory.phenomenon}
										</span>
									{/if}
								</div>

								{#if advisory.raw_text}
									<div class="bg-gray-100 p-3 rounded mb-4 font-mono text-sm whitespace-pre-wrap">{advisory.raw_text}</div>
								{/if}

								<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
									<div>
										<h4 class="text-sm font-medium text-gray-700 mb-2">Advisory Information</h4>
										<table class="w-full text-sm">
											<tbody>
												{#if advisory.valid_from && advisory.valid_to}
													<tr>
														<td class="py-1 text-gray-600">Valid</td>
														<td class="py-1">{formatValidityPeriod(advisory.valid_from, advisory.valid_to)}</td>
													</tr>
												{/if}
												{#if advisory.altitude}
													<tr>
														<td class="py-1 text-gray-600">Altitude</td>
														<td class="py-1">
															{#if advisory.altitude.lower !== undefined && advisory.altitude.upper !== undefined}
																{advisory.altitude.lower}-{advisory.altitude.upper} ft
															{:else if advisory.altitude.lower !== undefined}
																Above {advisory.altitude.lower} ft
															{:else if advisory.altitude.upper !== undefined}
																Below {advisory.altitude.upper} ft
															{:else}
																Undefined
															{/if}
														</td>
													</tr>
												{/if}
											</tbody>
										</table>
									</div>
									
									<div>
										{#if advisory.area && advisory.area.length > 0}
											<div class="flex justify-between items-center mb-2">
												<h4 class="text-sm font-medium text-gray-700">Area Coordinates</h4>
												<button
													on:click={() => showMap = !showMap}
													class="text-xs text-blue-600 hover:text-blue-800"
												>
													{showMap ? 'Hide Map' : 'Show Map'}
												</button>
											</div>
											
											{#if showMap}
												<div class="bg-gray-100 p-2 rounded">
													<p class="text-xs text-gray-500 mb-2">Map would be displayed here in a production environment.</p>
													<div class="h-32 flex items-center justify-center border border-gray-300 bg-white">
														<span class="text-sm text-gray-500">Geographic visualization of advisory area</span>
													</div>
												</div>
											{:else}
												<div class="bg-gray-100 p-2 rounded">
													<div class="max-h-32 overflow-y-auto">
														<table class="w-full text-xs">
															<thead>
																<tr class="bg-gray-200">
																	<th class="py-1 px-2 text-left">Lat</th>
																	<th class="py-1 px-2 text-left">Lon</th>
																</tr>
															</thead>
															<tbody>
																{#each advisory.area as coord, i}
																	{#if i < 10}
																		<tr class={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
																			<td class="py-1 px-2">{coord.lat}</td>
																			<td class="py-1 px-2">{coord.lon}</td>
																		</tr>
																	{:else if i === 10}
																		<tr>
																			<td colspan="2" class="py-1 px-2 text-center text-gray-500">
																				...and {advisory.area.length - 10} more points
																			</td>
																		</tr>
																	{/if}
																{/each}
															</tbody>
														</table>
													</div>
												</div>
											{/if}
										{:else}
											<h4 class="text-sm font-medium text-gray-700 mb-2">Area Information</h4>
											<p class="text-sm text-gray-500">No detailed area information available</p>
										{/if}
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