<script>
	import { onMount } from 'svelte';
	import MetarHoverDisplay from '$lib/components/MetarHoverDisplay.svelte';

	// State management
	let station = 'KPHX';
	let hours = 1;
	let loading = false;
	let error = null;
	let result = null;
	let showAdvanced = false; // Toggle for advanced details
	let hoveredMetarSection = null; // Track which section is being hovered

	// Get class based on flight category
	function getFlightCategoryClass(category) {
		const classMap = {
			'VFR': 'bg-green-600 text-white',
			'MVFR': 'bg-blue-600 text-white',
			'IFR': 'bg-red-600 text-white', 
			'LIFR': 'bg-purple-800 text-white'
		};
		return classMap[category] || 'bg-gray-600 text-white';
	}
	
	// Get text color based on flight category for colored text
	function getFlightCategoryTextClass(category) {
		const classMap = {
			'VFR': 'text-green-700 font-bold',
			'MVFR': 'text-blue-700 font-bold',
			'IFR': 'text-red-700 font-bold',
			'LIFR': 'text-purple-900 font-bold'
		};
		return classMap[category] || 'text-gray-700';
	}

	// Format date for display
	function formatDate(timestamp) {
		if (!timestamp) return 'N/A';
		const date = new Date(timestamp);
		return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + 
			' on ' + date.toLocaleDateString([], { month: 'short', day: 'numeric' });
	}

	// Format parsed METAR date
	function formatMetarDate(timeInfo) {
		if (!timeInfo || !timeInfo.datetime) return 'N/A';
		return new Date(timeInfo.datetime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + 'Z';
	}
	
	// Safely format numerical values
	function safelyFormatNumber(value, unit = '', decimalPlaces = null) {
		if (value === undefined || value === null) return '-';
		
		let numValue = value;
		if (typeof value === 'string') {
			if (['NA', 'N/A', 'MISSING', '////'].includes(value.toUpperCase())) {
				return 'N/A';
			}
			
			numValue = parseFloat(value);
			if (isNaN(numValue)) return value;
		}
		
		if (decimalPlaces !== null) {
			numValue = numValue.toFixed(decimalPlaces);
		}
		
		return `${numValue}${unit}`;
	}
	
	// Generate wind icon class based on direction
	function getWindIconClass(direction) {
		if (!direction || direction === "VRB" || typeof direction !== 'number') {
			return 'transform rotate-0';
		}
		return `transform rotate-[${direction}deg]`;
	}
	
	// Get wind description with color-coding for intensity
	function getWindDescription(wind) {
		if (!wind) return { text: 'Not reported', class: '' };
		
		if (wind.text && wind.text.toLowerCase() === 'calm') {
			return { text: 'Calm', class: 'text-green-600' };
		}
		
		// Color based on speed
		let colorClass = 'text-green-600'; // Default for light winds
		if (wind.speed > 25 || (wind.gust && wind.gust > 35)) {
			colorClass = 'text-red-600 font-bold';
		} else if (wind.speed > 15 || (wind.gust && wind.gust > 25)) {
			colorClass = 'text-orange-600 font-bold';
		} else if (wind.speed > 8) {
			colorClass = 'text-yellow-600';
		}
		
		return {
			text: wind.text || `From ${wind.direction}° at ${wind.speed} knots`,
			class: colorClass
		};
	}
	
	// Get visibility description with color-coding
	function getVisibilityDescription(visibility) {
		if (!visibility) return { text: 'Not reported', class: '' };
		
		const visText = visibility.text || `${visibility.distance || visibility} SM`;
		let colorClass = 'text-green-600';
		
		// Color based on visibility distance
		const distance = visibility.distance || parseFloat(visibility);
		if (distance < 1) {
			colorClass = 'text-red-600 font-bold';
		} else if (distance < 3) {
			colorClass = 'text-orange-600 font-bold';
		} else if (distance < 5) {
			colorClass = 'text-yellow-600';
		}
		
		return { text: visText, class: colorClass };
	}

	// Generate a simple pilot summary if one doesn't exist
	function generatePilotSummary(metarData) {
		if (!metarData) return "No weather data available";
		
		const parts = [];
		parts.push(`At ${metarData.station}`);
		
		if (metarData.flight_category) {
			parts.push(`conditions are ${metarData.flight_category}`);
		}
		
		if (metarData.wind_direction !== undefined && metarData.wind_speed !== undefined) {
			parts.push(`with winds from ${metarData.wind_direction}° at ${metarData.wind_speed} knots`);
		}
		
		if (metarData.visibility !== undefined) {
			parts.push(`visibility ${metarData.visibility} SM`);
		}
		
		if (metarData.temperature !== undefined && metarData.dewpoint !== undefined) {
			parts.push(`temperature ${metarData.temperature}°C dewpoint ${metarData.dewpoint}°C`);
		}
		
		if (metarData.clouds && metarData.clouds.length > 0) {
			const cloudDesc = metarData.clouds.map(c => 
				`${c.cover} ${c.base || c.altitude || c.height || ''}${c.base ? ' feet' : ''}`
			).join(', ');
			parts.push(`clouds ${cloudDesc}`);
		}
		
		return parts.join('. ') + '.';
	}

	// Parse METAR into interactive sections (simplified version for highlighting)
	function parseMetarStringIntoSections(metarString) {
		if (!metarString) return [];
		
		const sections = [];
		const parts = metarString.split(' ');
		
		// Definitions for METAR parts with simplified patterns
		const metarSectionDefinitions = [
			{ type: 'station', pattern: /^[A-Z]{4}$/ },
			{ type: 'time', pattern: /^\d{6}Z$/ },
			{ type: 'wind', pattern: /^(00000|VRB\d{2}|\d{3})\d{2}(G\d{2})?(KT|MPS)$/ },
			{ type: 'variable_wind', pattern: /^\d{3}V\d{3}$/ },
			{ type: 'visibility', pattern: /^(?:\d{4}|CAVOK|[MP]?\d+(?:\s+\d+\/\d+)?SM)$/ },
			{ type: 'runway_vr', pattern: /^R\d{2}[RCL]?\/.+$/ },
			{ type: 'weather', pattern: /^(?:\+|-|VC|RE)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|FG|BR|SA|DU|HZ|FU|VA|PY|PO|SQ|FC|SS|DS){1,3}$/ },
			{ type: 'cloud', pattern: /^(?:SKC|CLR|NSC|NCD|FEW|SCT|BKN|OVC|VV)(?:\d{3})?(?:CB|TCU)?$/ },
			{ type: 'temp_dewpoint', pattern: /^M?\d{2}\/M?\d{2}$/ },
			{ type: 'altimeter', pattern: /^[AQ]\d{4}$/ },
			{ type: 'remarks_indicator', pattern: /^RMK$/ },
			{ type: 'remarks', pattern: /./ }
		];
		
		let remarksStarted = false;
		
		parts.forEach(part => {
			if (part === 'RMK') {
				remarksStarted = true;
				sections.push({ text: part, type: 'remarks_indicator' });
				return;
			}
			
			if (remarksStarted) {
				sections.push({ text: part, type: 'remarks' });
				return;
			}
			
			// Match the part against our patterns
			let matched = false;
			
			for (const def of metarSectionDefinitions) {
				if (def.pattern.test(part)) {
					sections.push({ text: part, type: def.type });
					matched = true;
					break;
				}
			}
			
			if (!matched) {
				sections.push({ text: part, type: 'unknown' });
			}
		});
		
		return sections;
	}

	// Get weather data mapped to categories
	function getWeatherDataByCategory(metarData) {
		if (!metarData) return {};
		
		const parsedMetar = metarData.parsed_metar || {};
		const categories = {
			station: {
				value: metarData.station || '',
				icon: 'airport',
				title: 'Station',
				rawText: metarData.station,
				description: `ICAO Airport Code`,
				color: 'blue'
			},
			time: {
				value: formatMetarDate(parsedMetar.time) || formatDate(metarData.timestamp),
				icon: 'clock',
				title: 'Time',
				rawText: parsedMetar.time?.text,
				description: 'Observation time',
				color: 'purple'
			},
			flight_category: {
				value: metarData.flight_category || parsedMetar.flight_category || 'UNKNOWN',
				icon: 'flight',
				title: 'Flight Category',
				rawText: null, // Not directly in METAR
				description: 'Visual flight rules category',
				color: metarData.flight_category === 'VFR' ? 'green' : 
					metarData.flight_category === 'MVFR' ? 'blue' : 
					metarData.flight_category === 'IFR' ? 'red' : 
					metarData.flight_category === 'LIFR' ? 'purple' : 'gray'
			},
			wind: {
				value: parsedMetar.wind ? 
					parsedMetar.wind.text || 
					`${parsedMetar.wind.direction !== 'VRB' ? parsedMetar.wind.direction + '°' : 'Variable'} ${parsedMetar.wind.speed} ${parsedMetar.wind.unit || 'kt'}` : 
					(metarData.wind_speed !== undefined ? `From ${metarData.wind_direction}° at ${metarData.wind_speed} knots` : 'Not reported'),
				icon: 'wind',
				title: 'Wind',
				rawText: parsedMetar.wind?.text,
				description: 'Wind direction and speed',
				color: 'cyan'
			},
			visibility: {
				value: parsedMetar.visibility ? 
					parsedMetar.visibility.text || `${parsedMetar.visibility.distance} SM` : 
					(metarData.visibility !== undefined ? `${metarData.visibility} SM` : 'Not reported'),
				icon: 'visibility',
				title: 'Visibility',
				rawText: parsedMetar.visibility?.text,
				description: 'Horizontal visibility',
				color: 'amber'
			},
			weather: {
				value: parsedMetar.weather && parsedMetar.weather.length > 0 ? 
					parsedMetar.weather.map(w => w.text).join(', ') : 'No significant weather',
				icon: 'weather',
				title: 'Weather',
				rawText: parsedMetar.weather?.map(w => w.text).join(' '),
				description: 'Present weather phenomena',
				color: 'emerald'
			},
			clouds: {
				value: parsedMetar.clouds && parsedMetar.clouds.length > 0 ?
					parsedMetar.clouds.map(c => `${c.cover_text || c.cover}${c.base ? ` at ${c.base} ft` : ''}${c.type ? ` (${c.type})` : ''}`).join(', ') :
					metarData.clouds && metarData.clouds.length > 0 ?
					metarData.clouds.map(c => `${c.cover}${c.base ? ` at ${c.base} ft` : ''}`).join(', ') :
					'No cloud information',
				icon: 'cloud',
				title: 'Clouds',
				rawText: parsedMetar.clouds?.map(c => c.text).join(' '),
				description: 'Cloud coverage and height',
				color: 'indigo'
			},
			temperature: {
				value: `${safelyFormatNumber(metarData.temperature || parsedMetar?.temperature, '°C')} / ${safelyFormatNumber(metarData.dewpoint || parsedMetar?.dewpoint, '°C')}`,
				icon: 'thermometer',
				title: 'Temp/Dewpoint',
				rawText: parsedMetar.temperature_dewpoint?.text,
				description: 'Temperature and dewpoint',
				color: 'red'
			},
			altimeter: {
				value: parsedMetar.altimeter ? 
					`${safelyFormatNumber(parsedMetar.altimeter.value, ` ${parsedMetar.altimeter.unit}`)}` : 
					'Not reported',
				icon: 'gauge',
				title: 'Altimeter',
				rawText: parsedMetar.altimeter?.text,
				description: 'Atmospheric pressure',
				color: 'purple'
			},
			remarks: {
				value: parsedMetar.remarks?.raw || 'No remarks',
				icon: 'note',
				title: 'Remarks',
				rawText: parsedMetar.remarks?.raw,
				description: 'Additional information',
				color: 'slate'
			}
		};
		
		return categories;
	}

	// Fetch METAR data from the API
	async function fetchMetar() {
		loading = true;
		error = null;
		result = null;

		try {
			// Fix: Updated API endpoint to use the correct path with v1
			const url = `/api/v1/metar/${station}?hours=${hours}`;
			console.log("Fetching METAR data from:", url);
			
			const response = await fetch(url);
			
			if (!response.ok) {
				throw new Error(`API error: ${response.status} ${response.statusText}`);
			}

			result = await response.json();
			console.log("METAR result:", result);
			
			// If there's no parsed_metar or pilot_summary, create a placeholder object
			if (!result.parsed_metar) {
				result.parsed_metar = {};
			}
			
			// Generate a summary if one doesn't exist
			if (!result.pilot_summary && !result.parsed_metar?.pilot_summary) {
				result.pilot_summary = generatePilotSummary(result);
				}
			
			// Parse the METAR string for highlighting
			if (result.raw_text) {
				result.parsedSections = parseMetarStringIntoSections(result.raw_text);
			}
		} catch (err) {
			error = err.message || 'Failed to fetch METAR data';
			console.error("METAR fetch error:", err);
		} finally {
			loading = false;
		}
	}

	// Highlight a section in the METAR string when hovering over a block
	function highlightMetarSection(type) {
		if (!result || !result.parsedSections) return;
		hoveredMetarSection = type;
	}
	
	function clearHighlight() {
		hoveredMetarSection = null;
	}
	
	// Icon components for weather categories
	const icons = {
		airport: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>`,
		clock: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`,
		flight: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3l14 9-14 9V3z"></path></svg>`,
		wind: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.879 16.121A3 3 0 1012.015 11L11 14H9c0 .768.293 1.536.879 2.121z"></path></svg>`,
		visibility: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path></svg>`,
		weather: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z"></path></svg>`,
		cloud: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z"></path></svg>`,
		thermometer: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>`,
		gauge: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`,
		note: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"></path></svg>`
	};

	// Auto-fetch data on component mount
	onMount(() => {
		fetchMetar();
	});
</script>

<svelte:head>
	<title>Cockpit METAR Display</title>
</svelte:head>

<div class="container mx-auto max-w-4xl">
	<div class="mb-6">
		<h1 class="text-2xl font-bold text-gray-800 mb-2">Interactive METAR Display</h1>
		<p class="text-gray-600">
			Hover over any weather block to see the corresponding section in the original METAR.
		</p>
	</div>

	<!-- Request Form -->
	<div class="bg-white rounded-lg shadow p-6 mb-6">
		<div class="flex flex-col md:flex-row gap-4 items-end">
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
				on:click={fetchMetar}
				class="px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
				disabled={loading}
			>
				{#if loading}
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

	<!-- Error message -->
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
					<div class="mt-1 text-sm text-red-700">{error}</div>
				</div>
			</div>
		</div>
	{/if}
	
	{#if result}
		<!-- Raw METAR with highlighted sections -->
		{#if result.raw_text}
			<div class="bg-white rounded-lg shadow p-6 mb-6">
				<!-- Replace simple highlighting with MetarHoverDisplay component -->
				<MetarHoverDisplay metarString={result.raw_text} />
				
				<p class="text-sm text-gray-600 mt-2 italic">
					Hover over any part of the METAR above to see its meaning, or hover over the weather blocks below to highlight related sections.
				</p>
			</div>
		{/if}
		
		<!-- Weather information blocks -->
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
			{#if result}
				{@const categories = getWeatherDataByCategory(result)}
				{#each Object.entries(categories) as [key, category]}
					<div 
						class="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow"
						on:mouseenter={() => highlightMetarSection(key)}
						on:mouseleave={clearHighlight}
					>
						<div class="bg-{category.color}-100 p-3 border-b border-{category.color}-200 flex items-center">
							{@html icons[category.icon] || icons.note}
							<h3 class="ml-2 font-medium text-{category.color}-800">{category.title}</h3>
						</div>
						<div class="p-4">
							<p class="text-gray-800 font-medium">{category.value}</p>
							<p class="text-sm text-gray-500 mt-1">{category.description}</p>
							{#if category.rawText}
								<div class="mt-2 text-xs font-mono bg-gray-100 p-1 rounded">
									{category.rawText}
								</div>
							{/if}
						</div>
					</div>
				{/each}
			{/if}
		</div>
		
		<!-- Pilot summary -->
		{#if result.pilot_summary || (result.parsed_metar && result.parsed_metar.pilot_summary)}
			<div class="bg-white rounded-lg shadow p-6 mb-6">
				<h2 class="text-lg font-semibold text-gray-700 mb-2">Pilot Summary</h2>
				<div class="bg-blue-50 p-4 rounded border border-blue-200">
					<p class="text-lg text-blue-800 leading-relaxed">
						{result.pilot_summary || result.parsed_metar?.pilot_summary}
					</p>
				</div>
			</div>
		{/if}
		
		<!-- Advanced information toggle -->
		<div class="mb-6">
			<button 
				on:click={() => showAdvanced = !showAdvanced}
				class="flex items-center text-blue-600 hover:text-blue-800"
			>
				<svg class="w-4 h-4 mr-1 transform {showAdvanced ? 'rotate-90' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
				</svg>
				{showAdvanced ? 'Hide' : 'Show'} Advanced Details
			</button>
		</div>
		
		{#if showAdvanced}
			<!-- Full parsed data -->
			<div class="mb-4">
				<h3 class="font-medium mb-2">Full Parsed Data</h3>
				<div class="bg-gray-100 p-3 rounded overflow-x-auto border border-gray-300">
					<pre class="text-xs">{JSON.stringify(result.parsed_metar || result, null, 2)}</pre>
				</div>
			</div>
		{/if}
	{:else if !loading && !error}
		<!-- Empty state when no data is loaded -->
		<div class="bg-blue-50 border-blue-200 border rounded-lg p-6 text-center">
			<svg class="w-16 h-16 mx-auto text-blue-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
			</svg>
			<h3 class="text-lg font-medium text-blue-700 mb-2">No Weather Data</h3>
			<p class="text-blue-600">Enter an airport code and click "Get Weather" to view current conditions</p>
		</div>
	{/if}
</div>

<style>
	/* Add some color styling for the different METAR components */
	[data-type="station"] {
		color: #3b82f6; /* blue-500 */
	}
	[data-type="time"] {
		color: #8b5cf6; /* purple-500 */
	}
	[data-type="wind"], [data-type="variable_wind"] {
		color: #06b6d4; /* cyan-500 */
	}
	[data-type="visibility"] {
		color: #f59e0b; /* amber-500 */
	}
	[data-type="weather"] {
		color: #10b981; /* emerald-500 */
	}
	[data-type="cloud"] {
		color: #6366f1; /* indigo-500 */
	}
	[data-type="temp_dewpoint"] {
		color: #ef4444; /* red-500 */
	}
	[data-type="altimeter"] {
		color: #8b5cf6; /* purple-500 */
	}
	[data-type="remarks_indicator"] {
		color: #64748b; /* slate-500 */
		font-weight: bold;
	}
	[data-type="remarks"] {
		color: #64748b; /* slate-500 */
	}
</style>