<script>
	const apiServices = [
		{
			name: 'METAR',
			path: '/api/metar',
			description: 'Meteorological Aerodrome Reports provide current surface weather observations at airports worldwide.',
			icon: 'M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z',
			iconBackground: 'bg-green-100',
			iconColor: 'text-green-700'
		},
		{
			name: 'TAF',
			path: '/api/taf',
			description: 'Terminal Aerodrome Forecasts provide weather forecasts for airports, typically covering 24-30 hour periods.',
			icon: 'M2.25 15a4.5 4.5 0 004.5 4.5H18a3.75 3.75 0 001.332-7.257 3 3 0 00-3.758-3.848 5.25 5.25 0 00-10.233 2.33A4.502 4.502 0 002.25 15z',
			iconBackground: 'bg-blue-100',
			iconColor: 'text-blue-700'
		},
		{
			name: 'PIREP',
			path: '/api/pirep',
			description: 'Pilot Reports provide observations from pilots about weather conditions encountered during flight.',
			icon: 'M15.75 5.25a3 3 0 013 3m3 0a6 6 0 01-7.029 5.912c-.563-.097-1.159.026-1.563.43L10.5 17.25H8.25v2.25H6v2.25H2.25v-2.818c0-.597.237-1.17.659-1.591l6.499-6.499c.404-.404.527-1 .43-1.563A6 6 0 1121.75 8.25z',
			iconBackground: 'bg-orange-100',
			iconColor: 'text-orange-700'
		},
		{
			name: 'SIGMET/AIRMET',
			path: '/api/sigmet',
			description: 'Significant Meteorological Information and Airmen\'s Meteorological Information advisories warn of hazardous weather conditions.',
			icon: 'M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z',
			iconBackground: 'bg-red-100',
			iconColor: 'text-red-700'
		}
	];

	const examples = [
		{
			title: 'Get METAR for KJFK',
			description: 'Retrieve the current METAR for John F. Kennedy International Airport',
			path: '/api/metar?airport=KJFK'
		},
		{
			title: 'Get TAF for KORD with 24-hour history',
			description: 'Retrieve TAF data for Chicago O\'Hare with a 24-hour history window',
			path: '/api/taf?airport=KORD&hours=24'
		},
		{
			title: 'Get PIREPs near Denver',
			description: 'Search for pilot reports within 100nm of Denver International Airport',
			path: '/api/pirep?airport=KDEN&radius=100'
		},
		{
			title: 'Get SIGMETs for the Continental US',
			description: 'Get active SIGMETs affecting the Continental United States',
			path: '/api/sigmet?region=conus'
		}
	];
</script>

<svelte:head>
	<title>API Testing Dashboard - Aviation Weather API Hub</title>
</svelte:head>

<div class="py-10">
	<!-- Hero section -->
	<div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
		<div class="bg-white shadow-lg rounded-lg overflow-hidden">
			<div class="px-6 py-12 md:px-12 text-center lg:text-left">
				<div class="grid lg:grid-cols-2 gap-12 items-center">
					<div class="mt-12 lg:mt-0">
						<h1 class="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl md:text-6xl mb-6">
							Aviation Weather API Hub
						</h1>
						<p class="mt-3 text-lg text-gray-500 sm:mx-auto sm:mt-5 sm:max-w-xl sm:text-xl lg:mx-0">
							Access comprehensive aviation weather data through our robust API endpoints. Test each service interactively and explore the documentation.
						</p>
						<div class="mt-8 flex flex-col sm:flex-row justify-center lg:justify-start gap-3">
							<a
								href="/api/docs"
								class="rounded-md bg-blue-600 px-5 py-3 text-base font-medium text-white shadow hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
							>
								View Documentation
							</a>
							<a
								href="https://github.com/yourusername/aviation-weather-api"
								target="_blank"
								class="rounded-md bg-gray-100 px-5 py-3 text-base font-medium text-gray-700 shadow hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500"
							>
								GitHub Repository
							</a>
						</div>
					</div>
					<div class="hidden lg:block">
						<img
							class="h-64 w-full object-cover rounded-lg shadow-lg"
							src="https://images.unsplash.com/photo-1464037866556-6812c9d1c72e?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80"
							alt="Aviation weather"
						/>
					</div>
				</div>
			</div>
		</div>
	</div>

	<!-- API Services section -->
	<div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 mt-12">
		<h2 class="text-2xl font-bold text-gray-900 mb-6">API Services</h2>
		<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
			{#each apiServices as service}
				<a href={service.path} class="block">
					<div class="bg-white shadow rounded-lg p-6 hover:shadow-md transition-shadow duration-200">
						<div class="flex items-start">
							<div class={`flex-shrink-0 ${service.iconBackground} p-3 rounded-full`}>
								<svg class={`h-6 w-6 ${service.iconColor}`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d={service.icon} />
								</svg>
							</div>
							<div class="ml-4">
								<h3 class="text-lg font-medium text-gray-900">{service.name}</h3>
								<p class="mt-1 text-sm text-gray-500">{service.description}</p>
								<div class="mt-3 flex items-center text-sm text-blue-600">
									<span>Test this API</span>
									<svg class="ml-1 h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
										<path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
									</svg>
								</div>
							</div>
						</div>
					</div>
				</a>
			{/each}
		</div>
	</div>

	<!-- Quick Examples section -->
	<div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 mt-12">
		<h2 class="text-2xl font-bold text-gray-900 mb-6">Quick Examples</h2>
		<div class="bg-white shadow rounded-lg divide-y">
			{#each examples as example, index}
				<div class="p-6 hover:bg-gray-50">
					<div class="flex flex-col md:flex-row justify-between">
						<div class="mb-4 md:mb-0">
							<h3 class="text-lg font-medium text-gray-900">{example.title}</h3>
							<p class="mt-1 text-sm text-gray-500">{example.description}</p>
						</div>
						<div class="flex items-center">
							<a
								href={example.path}
								class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
							>
								Try it
							</a>
						</div>
					</div>
				</div>
			{/each}
		</div>
	</div>

	<!-- Getting Started section -->
	<div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 mt-12">
		<h2 class="text-2xl font-bold text-gray-900 mb-6">Getting Started</h2>
		<div class="bg-white shadow rounded-lg p-6">
			<div class="prose prose-blue max-w-none">
				<p>The Aviation Weather API Hub provides a simple and unified interface to access various weather data sources for aviation purposes.</p>
				
				<h3>Base URL</h3>
				<div class="bg-gray-800 rounded p-3 overflow-x-auto">
					<code class="text-sm text-green-400">https://api.aviationweather.example.com/api/v1</code>
				</div>
				
				<h3 class="mt-6">Authentication</h3>
				<p>All API endpoints require an API key to be passed in the header. You can obtain an API key by registering on our developer portal.</p>
				<div class="bg-gray-800 rounded p-3 overflow-x-auto mt-2">
					<code class="text-sm text-blue-400">X-API-Key: your-api-key-here</code>
				</div>
				
				<h3 class="mt-6">Example Request</h3>
				<p>Here's an example of how to retrieve METAR data for KPHX using curl:</p>
				<div class="bg-gray-800 rounded p-3 overflow-x-auto mt-2">
<pre class="text-sm text-blue-400"><code>curl -X GET "https://api.aviationweather.example.com/api/v1/metar/KPHX" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json"</code></pre>
				</div>
				
				<p class="mt-6">
					For detailed documentation and examples for each endpoint, visit our
					<a href="/api/docs" class="text-blue-600 hover:text-blue-800">API Documentation</a> page.
				</p>
			</div>
		</div>
	</div>
</div>