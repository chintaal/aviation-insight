<script>
	import { onMount } from 'svelte';
	import MetarHoverDisplay from '$lib/components/MetarHoverDisplay.svelte';
	import TafHoverDisplay from '$lib/components/TafHoverDisplay.svelte';
	import PirepHoverDisplay from '$lib/components/PirepHoverDisplay.svelte';

	// State management for each weather type
	let station = 'KPHX';
	let loading = {
		metar: false,
		taf: false,
		pirep: false
	};
	let error = {
		metar: null,
		taf: null,
		pirep: null
	};
	let results = {
		metar: null,
		taf: null,
		pirep: null
	};
	let summaries = {
		metar: null,
		taf: null,
		pirep: null
	};

	// Fetch METAR data from the API
	async function fetchMetar() {
		loading.metar = true;
		error.metar = null;
		results.metar = null;
		summaries.metar = null;

		try {
			const response = await fetch(`/api/v1/metar/${station}`);
			
			if (!response.ok) {
				throw new Error(`Error ${response.status}: ${await response.text()}`);
			}
			
			const data = await response.json();
			console.log('METAR response:', data);
			
			// Handle both response formats: direct object or nested under data property
			const metarData = data.data || data;
			
			if (metarData && metarData.raw_text) {
				results.metar = metarData;
				// Also fetch the summary for this METAR
				fetchMetarSummary(metarData.raw_text);
			} else {
				throw new Error('No METAR data available or invalid format');
			}
		} catch (err) {
			error.metar = err.message || 'Failed to fetch METAR data';
			console.error("METAR fetch error:", err);
		} finally {
			loading.metar = false;
		}
	}

	// Fetch TAF data from the API
	async function fetchTaf() {
		loading.taf = true;
		error.taf = null;
		results.taf = null;
		summaries.taf = null;

		try {
			const response = await fetch(`/api/v1/taf/${station}`);
			
			if (!response.ok) {
				throw new Error(`Error ${response.status}: ${await response.text()}`);
			}
			
			const data = await response.json();
			console.log('TAF response:', data);
			
			// Handle both response formats: direct object or nested under data property
			const tafData = data.data || data;
			
			if (tafData && tafData.raw_text) {
				results.taf = tafData;
				// Also fetch the summary for this TAF
				fetchTafSummary(tafData.raw_text);
			} else {
				throw new Error('No TAF data available or invalid format');
			}
		} catch (err) {
			error.taf = err.message || 'Failed to fetch TAF data';
			console.error("TAF fetch error:", err);
		} finally {
			loading.taf = false;
		}
	}

	// Fetch PIREP data from the API
	async function fetchPirep() {
		loading.pirep = true;
		error.pirep = null;
		results.pirep = null;
		summaries.pirep = null;

		try {
				// First try the previously fixed format
			const response = await fetch(`/api/v1/pirep/${station}?distance=200&age=2`);
			
			if (!response.ok) {
				throw new Error(`Error ${response.status}: ${await response.text()}`);
			}
			
			const data = await response.json();
			console.log('PIREP response:', data);
			
			// Handle both response formats: direct array or nested under data property
			const pirepData = data.data || data;
			
			if (pirepData && Array.isArray(pirepData) && pirepData.length > 0) {
				results.pirep = pirepData;
				// Also fetch the summary for this PIREP
				fetchPirepSummary(pirepData[0].raw_text, pirepData[0].location || station);
			} else {
				throw new Error('No PIREP data available or invalid format');
			}
		} catch (err) {
			error.pirep = err.message || 'Failed to fetch PIREP data';
			console.error("PIREP fetch error:", err);
			
			// Try alternative endpoint if first format failed
			try {
				const altResponse = await fetch(`/api/v1/pirep?station=${station}&distance=200&age=2`);
				
				if (!altResponse.ok) {
					throw new Error(`Error ${altResponse.status}: ${await altResponse.text()}`);
				}
				
				const altData = await altResponse.json();
				console.log('Alternative PIREP response:', altData);
				
				const pirepData = altData.data || altData;
				
				if (pirepData && Array.isArray(pirepData) && pirepData.length > 0) {
					results.pirep = pirepData;
					error.pirep = null;
					fetchPirepSummary(pirepData[0].raw_text, pirepData[0].location || station);
				}
			} catch (altErr) {
				console.error("Alternative PIREP endpoint also failed:", altErr);
				// Keep original error if both attempts fail
			}
		} finally {
			loading.pirep = false;
		}
	}

	// Fetch summaries using the OpenAI service (GPT-O1 mini)
	async function fetchMetarSummary(metarText) {
		try {
			const response = await fetch('/api/v1/weather-summary/metar', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ text: metarText })
			});
			
			if (!response.ok) {
				throw new Error(`Error ${response.status}`);
			}
			
			const data = await response.json();
			summaries.metar = data;
		} catch (err) {
			console.error("Failed to fetch METAR summary:", err);
			summaries.metar = { error: err.message };
		}
	}

	async function fetchTafSummary(tafText) {
		try {
			const response = await fetch('/api/v1/weather-summary/taf', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ text: tafText })
			});
			
			if (!response.ok) {
				throw new Error(`Error ${response.status}`);
			}
			
			const data = await response.json();
			summaries.taf = data;
		} catch (err) {
			console.error("Failed to fetch TAF summary:", err);
			summaries.taf = { error: err.message };
		}
	}

	async function fetchPirepSummary(pirepText, locationCode) {
		try {
			const response = await fetch('/api/v1/weather-summary/pirep', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ text: pirepText, location: locationCode })
			});
			
			if (!response.ok) {
				throw new Error(`Error ${response.status}`);
			}
			
			const data = await response.json();
			summaries.pirep = data;
		} catch (err) {
			console.error("Failed to fetch PIREP summary:", err);
			summaries.pirep = { error: err.message };
		}
	}

	// Fetch all weather data at once
	function fetchAllWeather() {
		fetchMetar();
		fetchTaf();
		fetchPirep();
	}

	// Auto-fetch data on component mount
	onMount(() => {
		fetchAllWeather();
	});
</script>

<svelte:head>
	<title>Weather Summaries and Hover Display</title>
</svelte:head>

<div class="container mx-auto max-w-7xl px-4 py-8">
	<div class="mb-8">
		<h1 class="text-3xl font-bold text-gray-800 mb-2">Weather Summaries & Interactive Displays</h1>
		<p class="text-gray-600 mb-4">
			View comprehensive weather data with interactive hover functionality and AI-generated summaries.
		</p>
		
		<div class="bg-white rounded-lg shadow p-6 mb-6">
			<div class="flex flex-wrap gap-4 items-end">
				<div class="flex-1">
					<label for="station" class="block text-sm font-medium text-gray-700 mb-1">Airport Code</label>
					<input
						type="text"
						id="station"
						bind:value={station}
						placeholder="e.g. KPHX, KJFK"
						class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
					/>
				</div>

				<button
					on:click={fetchAllWeather}
					class="px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
					disabled={loading.metar || loading.taf || loading.pirep}
				>
					{#if loading.metar || loading.taf || loading.pirep}
						<svg class="animate-spin inline-block -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
							<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
							<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
						</svg>
						Loading...
					{:else}
						Get Weather
					{/if}
				</button>
			</div>
		</div>
	</div>

	<div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
		<!-- METAR Section -->
		<div class="col-span-1">
			<div class="bg-white rounded-lg shadow overflow-hidden">
				<div class="bg-gradient-to-r from-blue-500 to-blue-600 px-6 py-4">
					<h2 class="text-xl font-bold text-white">METAR Report</h2>
					<p class="text-blue-100 text-sm">Current aviation weather observation</p>
				</div>

				<div class="p-6">
					{#if error.metar}
						<div class="bg-red-50 border-l-4 border-red-500 p-4">
							<div class="flex">
								<div class="flex-shrink-0">
									<svg class="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
									</svg>
								</div>
								<div class="ml-3">
									<h3 class="text-sm font-medium text-red-800">Error</h3>
									<div class="mt-1 text-sm text-red-700">{error.metar}</div>
								</div>
							</div>
						</div>
					{:else if loading.metar}
						<div class="flex justify-center py-8">
							<svg class="animate-spin h-8 w-8 text-blue-500" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
						</div>
					{:else if results.metar}
						<!-- Interactive METAR Hover Display -->
						<div class="mb-6">
							<h3 class="font-semibold text-gray-800 mb-3">Interactive METAR Display</h3>
							<MetarHoverDisplay metarString={results.metar.raw_text} />
						</div>

						<!-- METAR Summary Card -->
						<div class="mt-6 bg-blue-50 rounded-lg p-4 border border-blue-100">
							<h3 class="font-semibold text-blue-800 mb-2 flex items-center">
								<svg class="w-5 h-5 mr-1 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
								</svg>
								AI-Generated Summary
							</h3>

							{#if summaries.metar && !summaries.metar.error}
								<div class="space-y-3">
									<div class="bg-white p-3 rounded border border-blue-100">
										<h4 class="text-sm font-medium text-blue-800 mb-1">Plain Language Summary:</h4>
										<p class="text-sm text-gray-700">{summaries.metar.summary || "Generating summary..."}</p>
									</div>
									
									<div class="bg-white p-3 rounded border border-blue-100">
										<h4 class="text-sm font-medium text-blue-800 mb-1">Reasoning:</h4>
										<p class="text-sm text-gray-700">{summaries.metar.reasoning || "Generating reasoning..."}</p>
									</div>
									
									<div class="bg-white p-3 rounded border border-blue-100">
										<h4 class="text-sm font-medium text-blue-800 mb-1">Flight Considerations:</h4>
										<p class="text-sm text-gray-700">{summaries.metar.flight_considerations || "Analyzing flight considerations..."}</p>
									</div>
								</div>
							{:else}
								<div class="bg-gray-100 p-3 rounded">
									<p class="text-sm text-gray-600">{summaries.metar?.error || "Generating METAR summary..."}</p>
								</div>
							{/if}
						</div>
					{:else}
						<div class="text-center py-10">
							<svg class="w-16 h-16 mx-auto text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
							</svg>
							<h3 class="text-lg font-medium text-gray-900 mb-1">No METAR Data</h3>
							<p class="text-gray-500">Enter an airport code and click "Get Weather"</p>
						</div>
					{/if}
				</div>
			</div>
		</div>

		<!-- TAF Section -->
		<div class="col-span-1">
			<div class="bg-white rounded-lg shadow overflow-hidden">
				<div class="bg-gradient-to-r from-indigo-500 to-indigo-600 px-6 py-4">
					<h2 class="text-xl font-bold text-white">TAF Report</h2>
					<p class="text-indigo-100 text-sm">Terminal aerodrome forecast</p>
				</div>

				<div class="p-6">
					{#if error.taf}
						<div class="bg-red-50 border-l-4 border-red-500 p-4">
							<div class="flex">
								<div class="flex-shrink-0">
									<svg class="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
									</svg>
								</div>
								<div class="ml-3">
									<h3 class="text-sm font-medium text-red-800">Error</h3>
									<div class="mt-1 text-sm text-red-700">{error.taf}</div>
								</div>
							</div>
						</div>
					{:else if loading.taf}
						<div class="flex justify-center py-8">
							<svg class="animate-spin h-8 w-8 text-indigo-500" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
						</div>
					{:else if results.taf}
						<!-- Interactive TAF Hover Display -->
						<div class="mb-6">
							<h3 class="font-semibold text-gray-800 mb-3">Interactive TAF Display</h3>
							<TafHoverDisplay tafString={results.taf.raw_text} />
						</div>

						<!-- TAF Summary Card -->
						<div class="mt-6 bg-indigo-50 rounded-lg p-4 border border-indigo-100">
							<h3 class="font-semibold text-indigo-800 mb-2 flex items-center">
								<svg class="w-5 h-5 mr-1 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
								</svg>
								AI-Generated Summary
							</h3>

							{#if summaries.taf && !summaries.taf.error}
								<div class="space-y-3">
									<div class="bg-white p-3 rounded border border-indigo-100">
										<h4 class="text-sm font-medium text-indigo-800 mb-1">Plain Language Summary:</h4>
										<p class="text-sm text-gray-700">{summaries.taf.summary || "Generating summary..."}</p>
									</div>
									
									<div class="bg-white p-3 rounded border border-indigo-100">
										<h4 class="text-sm font-medium text-indigo-800 mb-1">Reasoning:</h4>
										<p class="text-sm text-gray-700">{summaries.taf.reasoning || "Generating reasoning..."}</p>
									</div>
									
									<div class="bg-white p-3 rounded border border-indigo-100">
										<h4 class="text-sm font-medium text-indigo-800 mb-1">Flight Planning:</h4>
										<p class="text-sm text-gray-700">{summaries.taf.flight_planning || "Analyzing flight planning considerations..."}</p>
									</div>
								</div>
							{:else}
								<div class="bg-gray-100 p-3 rounded">
									<p class="text-sm text-gray-600">{summaries.taf?.error || "Generating TAF summary..."}</p>
								</div>
							{/if}
						</div>
					{:else}
						<div class="text-center py-10">
							<svg class="w-16 h-16 mx-auto text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
							</svg>
							<h3 class="text-lg font-medium text-gray-900 mb-1">No TAF Data</h3>
							<p class="text-gray-500">Enter an airport code and click "Get Weather"</p>
						</div>
					{/if}
				</div>
			</div>
		</div>

		<!-- PIREP Section -->
		<div class="col-span-1">
			<div class="bg-white rounded-lg shadow overflow-hidden">
				<div class="bg-gradient-to-r from-emerald-500 to-emerald-600 px-6 py-4">
					<h2 class="text-xl font-bold text-white">PIREP Report</h2>
					<p class="text-emerald-100 text-sm">Pilot weather reports</p>
				</div>

				<div class="p-6">
					{#if error.pirep}
						<div class="bg-red-50 border-l-4 border-red-500 p-4">
							<div class="flex">
								<div class="flex-shrink-0">
									<svg class="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
									</svg>
								</div>
								<div class="ml-3">
									<h3 class="text-sm font-medium text-red-800">Error</h3>
									<div class="mt-1 text-sm text-red-700">{error.pirep}</div>
								</div>
							</div>
						</div>
					{:else if loading.pirep}
						<div class="flex justify-center py-8">
							<svg class="animate-spin h-8 w-8 text-emerald-500" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
						</div>
					{:else if results.pirep && results.pirep.length > 0}
						<!-- Interactive PIREP Hover Display -->
						<div class="mb-6">
							<h3 class="font-semibold text-gray-800 mb-3">Interactive PIREP Display</h3>
							<PirepHoverDisplay 
								pirepString={results.pirep[0].raw_text} 
								locationCode={results.pirep[0].location || station} 
							/>
						</div>

						<!-- PIREP Summary Card -->
						<div class="mt-6 bg-emerald-50 rounded-lg p-4 border border-emerald-100">
							<h3 class="font-semibold text-emerald-800 mb-2 flex items-center">
								<svg class="w-5 h-5 mr-1 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
								</svg>
								AI-Generated Summary
							</h3>

							{#if summaries.pirep && !summaries.pirep.error}
								<div class="space-y-3">
									<div class="bg-white p-3 rounded border border-emerald-100">
										<h4 class="text-sm font-medium text-emerald-800 mb-1">Plain Language Summary:</h4>
										<p class="text-sm text-gray-700">{summaries.pirep.summary || "Generating summary..."}</p>
									</div>
									
									<div class="bg-white p-3 rounded border border-emerald-100">
										<h4 class="text-sm font-medium text-emerald-800 mb-1">Reasoning:</h4>
										<p class="text-sm text-gray-700">{summaries.pirep.reasoning || "Generating reasoning..."}</p>
									</div>
									
									<div class="bg-white p-3 rounded border border-emerald-100">
										<h4 class="text-sm font-medium text-emerald-800 mb-1">Hazard Assessment:</h4>
										<p class="text-sm text-gray-700">{summaries.pirep.hazard_assessment || "Analyzing hazards..."}</p>
									</div>
								</div>
							{:else}
								<div class="bg-gray-100 p-3 rounded">
									<p class="text-sm text-gray-600">{summaries.pirep?.error || "Generating PIREP summary..."}</p>
								</div>
							{/if}
						</div>
					{:else}
						<div class="text-center py-10">
							<svg class="w-16 h-16 mx-auto text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
							</svg>
							<h3 class="text-lg font-medium text-gray-900 mb-1">No PIREP Data</h3>
							<p class="text-gray-500">Enter an airport code and click "Get Weather"</p>
						</div>
					{/if}
				</div>
			</div>
		</div>
	</div>
	
	<!-- Bottom Legend Section -->
	<div class="mt-8 bg-white rounded-lg shadow p-6">
		<h2 class="text-xl font-bold text-gray-800 mb-4">Detailed Information</h2>
		
		<div class="grid grid-cols-1 md:grid-cols-3 gap-6">
			<div>
				<h3 class="font-semibold text-blue-700 mb-2">About METAR</h3>
				<p class="text-sm text-gray-600 mb-2">
					METAR (Meteorological Terminal Air Report) is a format for reporting weather information used by pilots and meteorologists. It contains data on:
				</p>
				<ul class="text-sm text-gray-600 list-disc pl-5 space-y-1">
					<li>Wind direction and speed</li>
					<li>Visibility</li>
					<li>Precipitation and weather phenomena</li>
					<li>Cloud coverage and ceiling height</li>
					<li>Temperature and dew point</li>
					<li>Barometric pressure (altimeter setting)</li>
				</ul>
			</div>
			
			<div>
				<h3 class="font-semibold text-indigo-700 mb-2">About TAF</h3>
				<p class="text-sm text-gray-600 mb-2">
					TAF (Terminal Aerodrome Forecast) is a format for reporting weather forecast information for aviation. Key features:
				</p>
				<ul class="text-sm text-gray-600 list-disc pl-5 space-y-1">
					<li>Forecasts typically cover 24-30 hours</li>
					<li>Includes expected changes in weather conditions</li>
					<li>Utilizes change indicators (BECMG, TEMPO, PROB)</li>
					<li>Forecasts same elements as METAR</li>
					<li>Critical for flight planning and operations</li>
				</ul>
			</div>
			
			<div>
				<h3 class="font-semibold text-emerald-700 mb-2">About PIREP</h3>
				<p class="text-sm text-gray-600 mb-2">
					PIREP (Pilot Report) provides real-world observations from pilots in flight. Important aspects:
				</p>
				<ul class="text-sm text-gray-600 list-disc pl-5 space-y-1">
					<li>First-hand observations from aircraft</li>
					<li>Reports on turbulence, icing, clouds, and other hazards</li>
					<li>Categorized as routine (UA) or urgent (UUA)</li>
					<li>Includes altitude, location, and aircraft type</li>
					<li>Critical supplement to forecast information</li>
				</ul>
			</div>
		</div>
	</div>

	<div class="mt-8 text-center text-sm text-gray-500">
		<p>Developed for Honeywell Design-a-thon | Weather data powered by aviation APIs | Summaries generated by GPT-O1 mini</p>
	</div>
</div>

<style>
	/* Some additional styling for the summaries section */
	.space-y-3 > :not([hidden]) ~ :not([hidden]) {
		--tw-space-y-reverse: 0;
		margin-top: calc(0.75rem * calc(1 - var(--tw-space-y-reverse)));
		margin-bottom: calc(0.75rem * var(--tw-space-y-reverse));
	}
</style>