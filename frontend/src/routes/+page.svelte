<script>
	// API categories for the cards
	const apiCategories = [
		{
			name: 'METAR',
			description: 'Current airport weather observations',
			path: '/api/metar',
			icon: 'weather-sunny'
		},
		{
			name: 'TAF',
			description: 'Terminal aerodrome forecasts',
			path: '/api/taf',
			icon: 'weather-cloudy'
		},
		{
			name: 'PIREP',
			description: 'Pilot weather reports',
			path: '/api/pirep',
			icon: 'airplane'
		},
		{
			name: 'SIGMET/AIRMET',
			description: 'Significant meteorological advisories',
			path: '/api/sigmet',
			icon: 'warning'
		}
	];

	// Sample statistics
	const stats = [
		{ name: 'API Endpoints', value: '16+' },
		{ name: 'Data Sources', value: '2' },
		{ name: 'Weather Products', value: '4' },
		{ name: 'Response Time', value: '<500ms' }
	];

	let station = ''; // Add initialization for station variable if used in this file
</script>

<svelte:head>
	<title>Aviation Weather API Hub</title>
</svelte:head>

<div class="container mx-auto max-w-6xl">
	<!-- Hero section -->
	<div class="text-center mb-12">
		<h1 class="text-4xl font-extrabold text-gray-900 mb-4">Aviation Weather API Hub</h1>
		<p class="text-xl text-gray-600 max-w-3xl mx-auto">
			A unified API for accessing aviation weather data from multiple providers, with consistent formatting
			and comprehensive data coverage.
		</p>
	</div>

	<!-- Stats section -->
	<div class="bg-white shadow rounded-lg p-6 mb-10">
		<dl class="grid grid-cols-1 gap-x-8 gap-y-6 lg:grid-cols-4">
			{#each stats as stat}
				<div class="text-center">
					<dt class="text-sm font-medium text-gray-500">{stat.name}</dt>
					<dd class="mt-1 text-3xl font-semibold text-blue-600">{stat.value}</dd>
				</div>
			{/each}
		</dl>
	</div>

	<!-- API categories -->
	<h2 class="text-2xl font-bold text-gray-800 mb-6">API Testing Interfaces</h2>
	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
		{#each apiCategories as category}
			<a
				href={category.path}
				class="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300 p-6 border border-gray-100 hover:border-blue-200"
			>
				<div class="flex flex-col items-center text-center">
					<div class="mb-4 p-3 bg-blue-50 rounded-full">
						<svg class="w-8 h-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							{#if category.icon === 'weather-sunny'}
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
							{:else if category.icon === 'weather-cloudy'}
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
							{:else if category.icon === 'airplane'}
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
							{:else if category.icon === 'warning'}
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
							{/if}
						</svg>
					</div>
					<h3 class="text-lg font-semibold text-gray-900 mb-1">{category.name}</h3>
					<p class="text-sm text-gray-600">{category.description}</p>
				</div>
			</a>
		{/each}
	</div>

	<!-- Quick API reference -->
	<div class="bg-gray-50 rounded-lg p-6 border border-gray-200">
		<h2 class="text-xl font-semibold text-gray-800 mb-4">Quick API Reference</h2>
		<div class="overflow-x-auto">
			<table class="min-w-full bg-white border border-gray-200">
				<thead>
					<tr>
						<th class="py-2 px-4 border-b border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Endpoint</th>
						<th class="py-2 px-4 border-b border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Description</th>
						<th class="py-2 px-4 border-b border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Example</th>
					</tr>
				</thead>
				<tbody>
					<tr>
						<td class="py-3 px-4 border-b border-gray-200 font-mono text-sm">/api/v1/metar/{station}</td>
						<td class="py-3 px-4 border-b border-gray-200 text-sm">Get METAR for an airport</td>
						<td class="py-3 px-4 border-b border-gray-200 text-sm font-mono">/api/v1/metar/KPHX</td>
					</tr>
					<tr>
						<td class="py-3 px-4 border-b border-gray-200 font-mono text-sm">/api/v1/taf/{station}</td>
						<td class="py-3 px-4 border-b border-gray-200 text-sm">Get TAF for an airport</td>
						<td class="py-3 px-4 border-b border-gray-200 text-sm font-mono">/api/v1/taf/KPHX</td>
					</tr>
					<tr>
						<td class="py-3 px-4 border-b border-gray-200 font-mono text-sm">/api/v1/pirep/station/{station}</td>
						<td class="py-3 px-4 border-b border-gray-200 text-sm">Get PIREPs near an airport</td>
						<td class="py-3 px-4 border-b border-gray-200 text-sm font-mono">/api/v1/pirep/station/KPHX</td>
					</tr>
					<tr>
						<td class="py-3 px-4 border-b border-gray-200 font-mono text-sm">/api/v1/sigmet</td>
						<td class="py-3 px-4 border-b border-gray-200 text-sm">Get all current SIGMETs</td>
						<td class="py-3 px-4 border-b border-gray-200 text-sm font-mono">/api/v1/sigmet?region=conus</td>
					</tr>
					<tr>
						<td class="py-3 px-4 border-b border-gray-200 font-mono text-sm">/api/v1/airmet</td>
						<td class="py-3 px-4 border-b border-gray-200 text-sm">Get all current AIRMETs</td>
						<td class="py-3 px-4 border-b border-gray-200 text-sm font-mono">/api/v1/airmet?region=conus</td>
					</tr>
				</tbody>
			</table>
		</div>
	</div>

	<!-- About section -->
	<div class="mt-12 bg-white rounded-lg shadow p-6">
		<h2 class="text-xl font-semibold text-gray-800 mb-3">About Aviation Weather API Hub</h2>
		<p class="text-gray-600 mb-4">
			This API hub provides access to aviation weather data from multiple sources including the NOAA
			Aviation Weather Center (AWC) and AVWX service. It normalizes data formats and provides a consistent 
			interface for accessing different types of aviation weather information.
		</p>
		<h3 class="text-lg font-medium text-gray-700 mt-4">Key Features</h3>
		<ul class="list-disc pl-6 mt-2 text-gray-600">
			<li>Multi-source data with consistent response schemas</li>
			<li>Support for METAR, TAF, PIREP, and SIGMET/AIRMET products</li>
			<li>Flexible querying by airport code, coordinates, or regions</li>
			<li>Robust error handling and fallbacks</li>
			<li>Comprehensive documentation</li>
		</ul>
	</div>
</div>
