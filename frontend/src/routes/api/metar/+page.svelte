<script>
	import { onMount } from 'svelte';

	// State management
	let station = 'KPHX';
	let hours = 1;
	let loading = false;
	let error = null;
	let result = null;
	let showRawData = false;

	// Get class based on flight category
	function getFlightCategoryClass(category) {
		const classMap = {
			'VFR': 'bg-green-100 text-green-800',
			'MVFR': 'bg-blue-100 text-blue-800',
			'IFR': 'bg-red-100 text-red-800',
			'LIFR': 'bg-purple-100 text-purple-800'
		};
		return classMap[category] || 'bg-gray-100 text-gray-800';
	}

	// Format date for display
	function formatDate(timestamp) {
		if (!timestamp) return 'N/A';
		return new Date(timestamp).toLocaleString();
	}

	// Fetch METAR data from the API
	async function fetchMetar() {
		loading = true;
		error = null;
		result = null;

		try {
			const url = `/api/v1/metar/${station}?hours=${hours}`;
			const response = await fetch(url);
			
			if (!response.ok) {
				throw new Error(`API error: ${response.status}`);
			}

			result = await response.json();
		} catch (err) {
			error = err.message || 'Failed to fetch METAR data';
		} finally {
			loading = false;
		}
	}

	// Auto-fetch data on component mount
	onMount(() => {
		fetchMetar();
	});
</script>

<svelte:head>
	<title>METAR API Testing - Aviation Weather API Hub</title>
</svelte:head>

<div class="container mx-auto max-w-4xl">
	<div class="mb-6">
		<h1 class="text-2xl font-bold text-gray-800 mb-2">METAR API Testing</h1>
		<p class="text-gray-600">
			Test the API endpoint for fetching METAR (Meteorological Aerodrome Report) data from Aviation Weather Center. These reports provide 
			current surface weather observations at airports around the world.
		</p>
	</div>

	<!-- Request Form -->
	<div class="bg-white rounded-lg shadow p-6 mb-6">
		<h2 class="text-lg font-semibold text-gray-700 mb-4">Request Parameters</h2>
		
		<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
			<div>
				<label for="station" class="block text-sm font-medium text-gray-700 mb-1">Airport Code</label>
				<input
					type="text"
					id="station"
					bind:value={station}
					placeholder="e.g. KPHX, KJFK"
					class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
				/>
			</div>

			<div>
				<label for="hours" class="block text-sm font-medium text-gray-700 mb-1">Hours</label>
				<input
					type="number"
					id="hours"
					bind:value={hours}
					min="1"
					max="24"
					class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
				/>
			</div>
		</div>

		<div class="flex justify-end">
			<button
				on:click={fetchMetar}
				class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
			>
				{#if loading}
					<svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
						<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
					</svg>
					Fetching...
				{:else}
					Fetch METAR
				{/if}
			</button>
		</div>
	</div>

	<!-- API Documentation -->
	<div class="bg-gray-50 rounded-lg border border-gray-200 p-6 mb-6">
		<h2 class="text-lg font-semibold text-gray-700 mb-2">API Endpoint Reference</h2>
		
		<div class="mb-4">
			<h3 class="text-md font-medium text-gray-700">METAR Endpoint</h3>
			<div class="bg-gray-800 rounded p-3 overflow-x-auto">
				<code class="text-xs text-green-400">GET /api/v1/metar/{station}?hours={hours}</code>
			</div>
		</div>
		
		<div class="grid grid-cols-1 gap-4 mb-4">
			<div>
				<h3 class="text-md font-medium text-gray-700 mb-1">Parameters</h3>
				<table class="min-w-full text-sm">
					<tbody>
						<tr>
							<td class="py-1 font-mono text-xs font-medium">station</td>
							<td class="py-1 pl-4">ICAO airport code (e.g., KPHX)</td>
						</tr>
						<tr>
							<td class="py-1 font-mono text-xs font-medium">hours</td>
							<td class="py-1 pl-4">Hours of history (1-24, default: 1)</td>
						</tr>
					</tbody>
				</table>
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
			<div class="px-6 py-4 bg-gray-50 border-b border-gray-200">
				<h2 class="text-lg font-semibold text-gray-700">METAR Results</h2>
			</div>

			<div class="p-6">
				<!-- Single result -->
				<div>
					{#if result.raw_text}
						<div class="bg-gray-100 p-3 rounded mb-4 font-mono text-sm whitespace-pre-wrap">{result.raw_text}</div>
					{/if}

					<div class="flex flex-wrap mb-4">
						{#if result.flight_category}
							<div class="mr-4 mb-2">
								<span class="text-sm font-medium text-gray-600 block mb-1">Flight Category</span>
								<span class={`px-2 py-1 rounded-full text-xs font-medium ${getFlightCategoryClass(result.flight_category)}`}>
									{result.flight_category}
								</span>
							</div>
						{/if}
						{#if result.timestamp}
							<div class="mb-2">
								<span class="text-sm font-medium text-gray-600 block mb-1">Observation Time</span>
								<span class="text-sm">{formatDate(result.timestamp)}</span>
							</div>
						{/if}
					</div>

					<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
						<div>
							<h3 class="text-sm font-medium text-gray-700 mb-2">Weather Conditions</h3>
							<table class="w-full text-sm">
								<tbody>
									{#if result.temperature !== undefined || result.dewpoint !== undefined}
										<tr>
											<td class="py-1 text-gray-600">Temperature/Dewpoint</td>
											<td class="py-1 font-medium">
												{result.temperature !== undefined ? `${result.temperature}°C` : '-'} / 
												{result.dewpoint !== undefined ? `${result.dewpoint}°C` : '-'}
											</td>
										</tr>
									{/if}
									{#if result.wind_direction !== undefined || result.wind_speed !== undefined}
										<tr>
											<td class="py-1 text-gray-600">Wind</td>
											<td class="py-1 font-medium">
												{result.wind_direction !== undefined ? `${result.wind_direction}°` : '-'} at 
												{result.wind_speed !== undefined ? `${result.wind_speed} knots` : '-'}
											</td>
										</tr>
									{/if}
									{#if result.visibility !== undefined}
										<tr>
											<td class="py-1 text-gray-600">Visibility</td>
											<td class="py-1 font-medium">{result.visibility} SM</td>
										</tr>
									{/if}
								</tbody>
							</table>
						</div>
					
						<div>
							<h3 class="text-sm font-medium text-gray-700 mb-2">Cloud Information</h3>
							{#if result.clouds && result.clouds.length > 0}
								<table class="w-full text-sm">
									<thead>
										<tr class="bg-gray-50">
											<th class="py-1 text-left font-medium text-gray-600">Coverage</th>
											<th class="py-1 text-left font-medium text-gray-600">Height (ft)</th>
											<th class="py-1 text-left font-medium text-gray-600">Type</th>
										</tr>
									</thead>
									<tbody>
										{#each result.clouds as cloud}
											<tr>
												<td class="py-1">{cloud.cover || cloud.type || '-'}</td>
												<td class="py-1">{cloud.base || cloud.altitude || cloud.height || '-'}</td>
												<td class="py-1">{cloud.type || '-'}</td>
											</tr>
										{/each}
									</tbody>
								</table>
							{:else}
								<p class="text-sm text-gray-500">No cloud information available</p>
							{/if}

							{#if result.ceiling !== undefined}
								<div class="mt-3">
									<span class="text-sm font-medium text-gray-600 block mb-1">Ceiling</span>
									<span class="text-sm">{result.ceiling} ft</span>
								</div>
							{/if}
						</div>
					</div>
				</div>

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