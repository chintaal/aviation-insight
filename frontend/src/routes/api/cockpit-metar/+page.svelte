<script>
	import { onMount } from 'svelte';

	// State management
	let station = 'KPHX';
	let hours = 1;
	let loading = false;
	let error = null;
	let result = null;
	let showAdvanced = false; // Toggle for advanced details

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

	// Fetch METAR data from the API
	async function fetchMetar() {
		loading = true;
		error = null;
		result = null;

		try {
			// Fix: Updated API endpoint to use relative path correctly
			const url = `../../../api/metar/${station}?hours=${hours}`;
			// Alternative approach if needed:
			// const url = `/api/metar/${station}?hours=${hours}`;
			
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
		} catch (err) {
			error = err.message || 'Failed to fetch METAR data';
			console.error("METAR fetch error:", err);
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
	<title>Cockpit METAR Display</title>
</svelte:head>

<div class="container mx-auto max-w-4xl">
	<div class="mb-6">
		<h1 class="text-2xl font-bold text-gray-800 mb-2">Cockpit METAR Display</h1>
		<p class="text-gray-600">
			Pilot-friendly METAR display designed for quick comprehension of weather conditions in the cockpit.
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
		<!-- Primary cockpit display designed for at-a-glance information -->
		<div class="bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden mb-6">
			<!-- Header with station and flight category -->
			<div class="px-6 py-4 flex justify-between items-center {getFlightCategoryClass(result.flight_category || result.parsed_metar?.flight_category)}">
				<div>
					<h2 class="text-2xl font-bold">{result.station}</h2>
					<p class="text-sm opacity-90">
						{#if result.parsed_metar?.time}
							{formatMetarDate(result.parsed_metar.time)}
						{:else if result.timestamp}
							{formatDate(result.timestamp)}
						{/if}
					</p>
				</div>
				<div class="text-right">
					<span class="text-3xl font-bold">
						{result.flight_category || result.parsed_metar?.flight_category || 'UNKNOWN'}
					</span>
					<p class="text-xs">Flight Category</p>
				</div>
			</div>
			
			<!-- Main weather display using the pilot summary -->
			{#if result.pilot_summary || (result.parsed_metar && result.parsed_metar.pilot_summary)}
				<div class="px-6 py-6 bg-gray-800 border-b border-gray-700">
					<p class="text-lg font-medium leading-relaxed">
						{result.pilot_summary || result.parsed_metar?.pilot_summary}
					</p>
				</div>
			{/if}
			
			<!-- Critical information grid -->
			<div class="grid grid-cols-1 md:grid-cols-2 gap-6 p-6">
				<!-- Wind information -->
				<div class="bg-gray-800 rounded-lg p-4 flex items-start">
					<div class="mr-3 mt-1">
						<svg class="w-8 h-8 text-blue-400 {result.parsed_metar?.wind?.direction !== 'VRB' && typeof result.parsed_metar?.wind?.direction === 'number' ? getWindIconClass(result.parsed_metar.wind.direction) : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path>
						</svg>
					</div>
					<div>
						<h3 class="text-lg font-semibold text-blue-300">Wind</h3>
						{#if result.parsed_metar?.wind}
							{@const windDesc = getWindDescription(result.parsed_metar.wind)}
							<p class="text-lg {windDesc.class}">{windDesc.text}</p>
							
							{#if result.parsed_metar.wind.variable_direction}
								<p class="text-sm mt-1 text-blue-200">
									Variable between {result.parsed_metar.wind.variable_direction.from}° and {result.parsed_metar.wind.variable_direction.to}°
								</p>
							{/if}
						{:else if result.wind_speed !== undefined}
							<p class="text-lg">From {safelyFormatNumber(result.wind_direction, '°')} at {safelyFormatNumber(result.wind_speed, ' knots')}</p>
						{:else}
							<p class="text-lg">Not reported</p>
						{/if}
					</div>
				</div>
				
				<!-- Visibility information -->
				<div class="bg-gray-800 rounded-lg p-4 flex items-start">
					<div class="mr-3 mt-1">
						<svg class="w-8 h-8 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
						</svg>
					</div>
					<div>
						<h3 class="text-lg font-semibold text-yellow-300">Visibility</h3>
						{#if result.parsed_metar?.visibility}
							{@const visDesc = getVisibilityDescription(result.parsed_metar.visibility)}
							<p class="text-lg {visDesc.class}">{visDesc.text}</p>
						{:else if result.visibility !== undefined}
							<p class="text-lg">{safelyFormatNumber(result.visibility, ' SM')}</p>
						{:else}
							<p class="text-lg">Not reported</p>
						{/if}
					</div>
				</div>
				
				<!-- Weather phenomena -->
				<div class="bg-gray-800 rounded-lg p-4 flex items-start">
					<div class="mr-3 mt-1">
						<svg class="w-8 h-8 text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z"></path>
						</svg>
					</div>
					<div>
						<h3 class="text-lg font-semibold text-teal-300">Weather</h3>
						{#if result.parsed_metar?.weather && result.parsed_metar.weather.length > 0}
							{#each result.parsed_metar.weather as wx, i}
								<p class="text-lg {i > 0 ? 'mt-1' : ''}">
									{wx.text}
								</p>
							{/each}
						{:else}
							<p class="text-lg">No significant weather</p>
						{/if}
					</div>
				</div>
				
				<!-- Cloud information -->
				<div class="bg-gray-800 rounded-lg p-4 flex items-start">
					<div class="mr-3 mt-1">
						<svg class="w-8 h-8 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path>
						</svg>
					</div>
					<div>
						<h3 class="text-lg font-semibold text-indigo-300">Clouds {#if result.ceiling || result.parsed_metar?.ceiling}<span class="font-normal text-base text-indigo-200">(ceiling {safelyFormatNumber(result.ceiling || result.parsed_metar?.ceiling, '′')})</span>{/if}</h3>
						
						{#if result.parsed_metar?.clouds && result.parsed_metar.clouds.length > 0}
							{#if result.parsed_metar.clouds.length === 1 && ['SKC', 'CLR', 'NSC', 'NCD', 'CAVOK'].includes(result.parsed_metar.clouds[0].cover)}
								<p class="text-lg">{result.parsed_metar.clouds[0].cover_text || result.parsed_metar.clouds[0].cover}</p>
							{:else}
								{#each result.parsed_metar.clouds as cloud}
									<p class="text-lg {cloud.ceiling ? 'font-semibold' : ''}">
										{cloud.cover_text || cloud.cover} {cloud.base !== null ? `at ${cloud.base} feet` : ''}
										{#if cloud.type_text}<span class="text-sm ml-1">({cloud.type_text})</span>{/if}
									</p>
								{/each}
							{/if}
						{:else if result.clouds && result.clouds.length > 0}
							{#each result.clouds as cloud}
								<p class="text-lg">
									{cloud.cover} at {safelyFormatNumber(cloud.base || cloud.altitude || cloud.height)}′
								</p>
							{/each}
						{:else}
							<p class="text-lg">No cloud information</p>
						{/if}
					</div>
				</div>
			</div>
			
			<!-- Temperature and pressure footer -->
			<div class="bg-gray-700 flex flex-wrap justify-between items-center p-4 text-lg">
				<div>
					<span class="font-semibold">Temperature:</span> 
					<span class="ml-2">
						{#if result.temperature !== undefined || (result.parsed_metar && result.parsed_metar.temperature !== undefined)}
							{safelyFormatNumber(result.temperature || result.parsed_metar?.temperature, '°C')}
						{:else}
							Not reported
						{/if}
					</span>
				</div>
				
				<div>
					<span class="font-semibold">Dew Point:</span>
					<span class="ml-2">
						{#if result.dewpoint !== undefined || (result.parsed_metar && result.parsed_metar.dewpoint !== undefined)}
							{safelyFormatNumber(result.dewpoint || result.parsed_metar?.dewpoint, '°C')}
						{:else}
							Not reported
						{/if}
					</span>
				</div>
				
				<div>
					<span class="font-semibold">Altimeter:</span>
					<span class="ml-2">
						{#if result.parsed_metar?.altimeter}
							{safelyFormatNumber(result.parsed_metar.altimeter.value, ` ${result.parsed_metar.altimeter.unit}`)}
						{:else}
							Not reported
						{/if}
					</span>
				</div>
			</div>
		</div>
		
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
			<!-- Raw METAR text -->
			{#if result.raw_text}
				<div class="mb-4">
					<h3 class="font-medium mb-2">Raw METAR</h3>
					<div class="bg-gray-100 p-3 rounded font-mono text-sm whitespace-pre-wrap border border-gray-300">
						{result.raw_text}
					</div>
				</div>
			{/if}
			
			<!-- Remarks section -->
			{#if result.parsed_metar?.remarks?.raw}
				<div class="mb-4">
					<h3 class="font-medium mb-2">Remarks</h3>
					<div class="bg-gray-100 p-3 rounded text-sm border border-gray-300">
						{result.parsed_metar.remarks.raw}
					</div>
				</div>
			{/if}
			
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