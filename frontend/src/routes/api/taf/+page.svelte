<script>
	import { onMount } from 'svelte';

	// State management
	let station = 'KJFK';
	let hours = 24;
	let loading = false;
	let error = null;
	let result = null;
	let showRawData = false;
	
	// Color codes for flight rules
	const flightRuleColors = {
		'VFR': { bg: 'bg-green-100', text: 'text-green-800', border: 'border-green-200' },
		'MVFR': { bg: 'bg-blue-100', text: 'text-blue-800', border: 'border-blue-200' },
		'IFR': { bg: 'bg-red-100', text: 'text-red-800', border: 'border-red-200' },
		'LIFR': { bg: 'bg-purple-100', text: 'text-purple-800', border: 'border-purple-200' }
	};
	
	// Default to VFR if no rule specified
	const defaultRuleColor = { bg: 'bg-gray-100', text: 'text-gray-800', border: 'border-gray-200' };

	// Format date for display
	function formatDate(timestamp) {
		if (!timestamp) return 'N/A';
		const date = new Date(timestamp);
		return date.toLocaleString(undefined, {
			month: 'short',
			day: 'numeric', 
			hour: '2-digit', 
			minute: '2-digit',
			hour12: false
		});
	}
	
	// Format date for display in UTC
	function formatDateUTC(timestamp) {
		if (!timestamp) return 'N/A';
		const date = new Date(timestamp);
		return date.toUTCString().replace('GMT', 'UTC');
	}

	// Format validity period
	function formatValidityPeriod(from, to) {
		if (!from || !to) return 'N/A';
		
		const fromDate = new Date(from);
		const toDate = new Date(to);
		
		const fromDay = fromDate.getUTCDate();
		const toDay = toDate.getUTCDate();
		
		const fromString = `${fromDate.getUTCDate().toString().padStart(2, '0')}/${(fromDate.getUTCHours()).toString().padStart(2, '0')}Z`;
		const toString = `${toDate.getUTCDate().toString().padStart(2, '0')}/${(toDate.getUTCHours()).toString().padStart(2, '0')}Z`;
		
		return `${fromString} to ${toString}`;
	}

	// Get wind direction arrow
	function getWindDirectionArrow(direction) {
		if (!direction || isNaN(direction)) return '↻';
		
		// Convert direction to one of 8 arrows
		const val = Math.floor((direction / 45) + 0.5);
		const arrows = ['↓', '↙', '←', '↖', '↑', '↗', '→', '↘'];
		return arrows[val % 8];
	}

	// Format wind data
	function formatWind(wind) {
		if (!wind) return 'N/A';
		
		if (wind.direction && wind.direction.value !== undefined) {
			let direction = wind.direction.value === 0 ? 'VRB' : `${wind.direction.value}°`;
			if (wind.direction.spoken) {
				direction += ` (${wind.direction.spoken})`;
			}
			
			let speed = wind.speed?.value !== undefined ? `${wind.speed.value}` : '0';
			let unit = wind.speed?.unit || 'kt';
			
			let gust = '';
			if (wind.gust && wind.gust.value !== undefined) {
				gust = ` gusting ${wind.gust.value}${wind.gust.unit || unit}`;
			}
			
			return `${direction} at ${speed}${unit}${gust}`;
		}
		
		// Fallback to simpler format
		return `${wind.direction || 'VRB'}° at ${wind.speed || '0'} kt${wind.gust ? ` gusting ${wind.gust} kt` : ''}`;
	}
	
	// Format visibility data
	function formatVisibility(vis) {
		if (!vis) return 'N/A';
		
		if (vis.value !== undefined) {
			return `${vis.value} ${vis.unit || 'sm'}${vis.spoken ? ` (${vis.spoken})` : ''}`;
		}
		
		return vis;
	}
	
	// Format cloud data
	function formatClouds(clouds) {
		if (!clouds || clouds.length === 0) return 'No clouds';
		
		return clouds.map(cloud => {
			if (typeof cloud === 'string') return cloud;
			
			const type = cloud.type || '';
			const height = cloud.altitude !== undefined ? cloud.altitude : (cloud.height || '');
			const unit = cloud.unit || 'ft';
			
			return `${type} at ${height}${unit}`;
		}).join(', ');
	}
	
	// Get flight rule styles
	function getFlightRuleStyles(rule) {
		if (!rule || typeof rule !== 'string') return defaultRuleColor;
		return flightRuleColors[rule] || defaultRuleColor;
	}
	
	// Generate a human-readable weather summary
	function generateWeatherSummary(period) {
		if (!period) return '';
		
		let summary = [];
		
		// Time information
		const start = period.start_time?.dt ? new Date(period.start_time.dt) : null;
		const end = period.end_time?.dt ? new Date(period.end_time.dt) : null;
		
		if (start && end) {
			const startStr = `${start.getUTCHours().toString().padStart(2, '0')}:${start.getUTCMinutes().toString().padStart(2, '0')}Z`;
			const endStr = `${end.getUTCHours().toString().padStart(2, '0')}:${end.getUTCMinutes().toString().padStart(2, '0')}Z`;
			summary.push(`From ${startStr} to ${endStr}`);
		}
		
		// Wind information
		if (period.wind_direction?.spoken && period.wind_speed?.value) {
			summary.push(`winds from ${period.wind_direction.spoken} at ${period.wind_speed.value} knots${period.wind_gust?.value ? ` gusting ${period.wind_gust.value}` : ''}`);
		}
		
		// Visibility information
		if (period.visibility?.spoken) {
			summary.push(`visibility ${period.visibility.spoken}`);
		}
		
		// Weather phenomena
		if (period.wx_codes && period.wx_codes.length > 0) {
			summary.push(period.wx_codes.join(', '));
		}
		
		// Clouds information
		if (period.cloud_layers && period.cloud_layers.length > 0) {
			const cloudDesc = period.cloud_layers.map(cloud => 
				`${cloud.type} clouds at ${cloud.altitude} feet`
			).join(', ');
			summary.push(cloudDesc);
		}
		
		// Flight rules
		if (period.flight_rules) {
			summary.push(`Conditions are ${period.flight_rules}`);
		}
		
		return summary.join(', ');
	}

	// Fetch TAF data from the API
	async function fetchTaf() {
		loading = true;
		error = null;
		result = null;

		try {
			// AVWX TAF endpoint
			const url = `/api/v1/taf/${station}?hours=${hours}`;
			
			const response = await fetch(url);
			
			if (!response.ok) {
				throw new Error(`API error: ${response.status}`);
			}

			result = await response.json();
			console.log('TAF data:', result);
		} catch (err) {
			error = err.message || 'Failed to fetch TAF data';
		} finally {
			loading = false;
		}
	}

	// Auto-fetch data on component mount
	onMount(() => {
		fetchTaf();
	});
</script>

<svelte:head>
	<title>TAF API Testing - Aviation Weather API Hub</title>
</svelte:head>

<div class="container mx-auto max-w-4xl">
	<div class="mb-6">
		<h1 class="text-2xl font-bold text-gray-800 mb-2">TAF API Testing</h1>
		<p class="text-gray-600">
			Test the API endpoint for fetching TAF (Terminal Aerodrome Forecast) data from AVWX. TAFs provide forecasts of significant weather 
			phenomena for airports, typically covering 24 to 30 hour periods.
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
					placeholder="e.g. KJFK, KPHX"
					class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
					/>
				<p class="mt-1 text-xs text-gray-500">Enter ICAO airport code</p>
			</div>

			<div>
				<label for="hours" class="block text-sm font-medium text-gray-700 mb-1">Hours</label>
				<input
					type="range"
					id="hours"
					bind:value={hours}
					min="1"
					max="48"
					class="block w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
				/>
				<div class="flex justify-between text-xs text-gray-600 mt-1">
					<span>1 hour</span>
					<span>{hours} hours</span>
					<span>48 hours</span>
				</div>
			</div>
		</div>

		<div class="flex justify-end">
			<button
				on:click={fetchTaf}
				class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
			>
				{#if loading}
					<svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
						<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
					</svg>
					Fetching...
				{:else}
					Fetch TAF
				{/if}
			</button>
		</div>
	</div>

	<!-- API Documentation -->
	<div class="bg-gray-50 rounded-lg border border-gray-200 p-6 mb-6">
		<h2 class="text-lg font-semibold text-gray-700 mb-2">API Endpoint Reference</h2>
		
		<div class="mb-4">
			<h3 class="text-md font-medium text-gray-700">AVWX TAF Endpoint</h3>
			<div class="bg-gray-800 rounded p-3 overflow-x-auto mb-2">
				<code class="text-xs text-green-400">GET /api/v1/taf/{station}</code>
			</div>
		</div>
		
		<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
			<div>
				<h3 class="text-md font-medium text-gray-700 mb-1">Parameters</h3>
				<table class="min-w-full text-sm">
					<tbody>
						<tr>
							<td class="py-1 font-mono text-xs font-medium">station</td>
							<td class="py-1 pl-4">ICAO airport code (e.g., KJFK)</td>
						</tr>
					</tbody>
				</table>
			</div>
			
			<div>
				<h3 class="text-md font-medium text-gray-700 mb-1">Response Format</h3>
				<div class="bg-gray-800 rounded p-3 overflow-x-auto">
<pre class="text-xs text-blue-400"><code>{'{\n  "source": "AVWX",\n  "station": "KJFK",\n  "raw_text": "TAF KJFK 231730Z 2318/2418...",\n  "forecast": [\n    {\n      "start_time": {"dt": "2025-04-23T18:00:00Z"},\n      "end_time": {"dt": "2025-04-24T00:00:00Z"},\n      "wind_direction": {"value": 220, "spoken": "southwest"},\n      "wind_speed": {"value": 12, "unit": "kt"},\n      "visibility": {"value": 6, "unit": "sm", "spoken": "6 miles"},\n      "flight_rules": "VFR",\n      "cloud_layers": [\n        {"type": "FEW", "altitude": 5000},\n        {"type": "SCT", "altitude": 20000}\n      ],\n      "wx_codes": ["VCSH"]\n    }\n  ],\n  "time": {"dt": "2025-04-23T17:30:00Z"},\n  "start_time": "2025-04-23T18:00:00Z",\n  "end_time": "2025-04-24T18:00:00Z"\n}'}</code></pre>
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
		<!-- Raw TAF Data -->
		<div class="mb-6">
			{#if result.raw_text}
				<div class="bg-gray-100 p-4 rounded-lg font-mono text-sm whitespace-pre-wrap">{result.raw_text}</div>
			{/if}
		</div>

		<!-- TAF Summary -->
		<div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
			<div class="flex justify-between items-start flex-wrap">
				<div class="mb-4">
					<h3 class="text-lg font-bold text-gray-800">{result.station}</h3>
					<p class="text-sm text-gray-600">
						{#if result.raw_data?.station?.name}
							{result.raw_data.station.name}
						{/if}
					</p>
				</div>
				
				<div class="text-right">
					<span class="bg-blue-100 text-blue-800 text-xs font-medium px-2 py-0.5 rounded">
						{result.source || 'AVWX'}
					</span>
				</div>
			</div>
			
			<!-- Enhanced Pilot Summary -->
			{#if result.pilot_summary}
				<div class="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
					<h4 class="text-blue-800 font-medium text-sm mb-2 flex items-center">
						<svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
						</svg>
						Pilot Briefing Summary
					</h4>
					<p class="text-blue-900 text-sm">{result.pilot_summary}</p>
				</div>
			{/if}
			
			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
				<!-- Time Information -->
				<div>
					<table class="w-full text-sm">
						<tbody>
							{#if result.issue_time || (result.raw_data && result.raw_data.time && result.raw_data.time.dt)}
								<tr>
									<td class="py-1 text-gray-600">Issued</td>
									<td class="py-1 font-medium">
										{formatDateUTC(result.issue_time || (result.raw_data && result.raw_data.time && result.raw_data.time.dt))}
									</td>
								</tr>
							{/if}
							{#if result.valid_from || result.raw_data?.start_time}
								<tr>
									<td class="py-1 text-gray-600">Valid From</td>
									<td class="py-1">{formatDateUTC(result.valid_from || result.raw_data?.start_time)}</td>
								</tr>
							{/if}
							{#if result.valid_to || result.raw_data?.end_time}
								<tr>
									<td class="py-1 text-gray-600">Valid To</td>
									<td class="py-1">{formatDateUTC(result.valid_to || result.raw_data?.end_time)}</td>
								</tr>
							{/if}
						</tbody>
					</table>
				</div>
				
				<!-- Station Information -->
				<div>
					<table class="w-full text-sm">
						<tbody>
							{#if result.raw_data?.station?.elevation}
								<tr>
									<td class="py-1 text-gray-600">Elevation</td>
									<td class="py-1">
										{result.raw_data.station.elevation} {result.raw_data.station.elevation_ft ? `(${result.raw_data.station.elevation_ft} ft)` : ''}
									</td>
								</tr>
							{/if}
							{#if result.raw_data?.units?.wind_speed || result.raw_data?.units?.altimeter || result.raw_data?.units?.temperature}
								<tr>
									<td class="py-1 text-gray-600">Units</td>
									<td class="py-1">
										Wind: {result.raw_data?.units?.wind_speed || 'kt'}, 
										Alt: {result.raw_data?.units?.altimeter || 'hPa'}, 
										Temp: {result.raw_data?.units?.temperature || 'C'}
									</td>
								</tr>
							{/if}
						</tbody>
					</table>
				</div>
			</div>
		</div>

		<!-- Timeline Visualization -->
		<div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
			<h3 class="text-md font-medium text-gray-800 mb-4">Forecast Timeline</h3>
			
			{#if result.raw_data?.forecast && result.raw_data.forecast.length > 0}
				<div class="relative pb-4 overflow-x-auto">
					<!-- Timeline track -->
					<div class="h-2 bg-gray-200 rounded absolute w-full top-6"></div>
					
					<div class="flex relative">
						{#each result.raw_data.forecast as period, i}
							{#if period.start_time && period.end_time}
								<!-- Calculate position based on start time relative to overall forecast period -->
								{#if period.start_time.dt && period.end_time.dt}
									<div class="mr-1 flex-grow">
										<div class={`p-3 mb-1 rounded border ${period.flight_rules ? getFlightRuleStyles(period.flight_rules).border : 'border-gray-200'}`}>
											<div class={`text-xs font-medium rounded-t px-2 py-1 -mt-3 -mx-3 mb-2 ${period.flight_rules ? getFlightRuleStyles(period.flight_rules).bg : 'bg-gray-100'} ${period.flight_rules ? getFlightRuleStyles(period.flight_rules).text : 'text-gray-800'}`}>
												{period.flight_rules || 'Period'} {i+1}
											</div>
											
											<div class="text-xs">
												<div><span class="font-medium">From:</span> {formatDateUTC(period.start_time.dt)}</div>
												<div><span class="font-medium">To:</span> {formatDateUTC(period.end_time.dt)}</div>
												
												{#if period.type}
													<div class="mt-1 bg-yellow-50 text-yellow-800 text-xs px-2 py-0.5 rounded">
														{period.type}
													</div>
												{/if}
											</div>
										</div>
										
										<div class="absolute w-3 h-3 bg-blue-500 rounded-full top-5 left-0 -ml-1.5"></div>
									</div>
								{/if}
							{/if}
						{/each}
					</div>
				</div>
			{/if}
		</div>
		
		<!-- Forecast Periods -->
		{#if result.raw_data?.forecast && result.raw_data.forecast.length > 0}
			<h3 class="text-lg font-medium text-gray-800 mb-3">Detailed Forecast Periods</h3>
			<div class="space-y-4 mb-4">
				{#each result.raw_data.forecast as period, i}
					<div class={`rounded-lg border ${period.flight_rules ? getFlightRuleStyles(period.flight_rules).border : 'border-gray-200'} overflow-hidden`}>
						<!-- Period header -->
						<div class={`${period.flight_rules ? getFlightRuleStyles(period.flight_rules).bg : 'bg-gray-100'} ${period.flight_rules ? getFlightRuleStyles(period.flight_rules).text : 'text-gray-800'} p-3 font-medium flex justify-between items-center`}>
							<div>
								{#if period.type}
									<span class="mr-2 bg-white bg-opacity-20 rounded-full px-2 py-0.5 text-xs">
										{period.type}
									</span>
								{/if}
								Period {i+1} 
								{#if period.flight_rules}
									<span class="ml-1 text-xs">({period.flight_rules})</span>
								{/if}
							</div>
							
							<div class="text-sm font-normal">
								{#if period.start_time?.dt && period.end_time?.dt}
									{formatValidityPeriod(period.start_time.dt, period.end_time.dt)}
								{/if}
							</div>
						</div>
						
						<!-- Period body -->
						<div class="p-3">
							<!-- Human-readable summary -->
							{#if period.pilot_insights}
								<div class="bg-blue-50 p-3 rounded mb-3 text-sm border border-blue-200">
									<div class="flex items-start">
										<svg class="h-5 w-5 text-blue-500 mt-0.5 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
										</svg>
										<div>
											<span class="font-medium text-blue-800">Pilot Insights:</span>
											<p class="text-blue-900">{period.pilot_insights}</p>
											
											{#if period.planning_considerations && period.planning_considerations.length > 0}
												<div class="mt-2">
													<span class="font-medium text-blue-800">Planning Considerations:</span>
													<ul class="list-disc pl-5 mt-1 text-blue-900">
														{#each period.planning_considerations as item}
															<li>{item}</li>
														{/each}
													</ul>
												</div>
											{/if}
										</div>
									</div>
								</div>
							{:else}
								<div class="bg-gray-50 p-2 rounded mb-3 text-sm">
									{generateWeatherSummary(period)}
								</div>
							{/if}
							
							<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
								<div>
									<table class="w-full text-sm">
										<tbody>
											<!-- Wind information -->
											<tr>
												<td class="py-1 text-gray-600 font-medium">Wind</td>
												<td class="py-1">
													{#if period.wind_direction || period.wind_speed}
														<div class="flex items-center">
															{#if period.wind_direction?.value}
																<span class="mr-2 text-lg" title="Wind direction: {period.wind_direction?.spoken || period.wind_direction.value + '°'}">
																	{getWindDirectionArrow(period.wind_direction.value)}
																</span>
															{/if}
															{formatWind({
																direction: period.wind_direction,
																speed: period.wind_speed,
																gust: period.wind_gust
															})}
														</div>
													{:else}
														N/A
													{/if}
												</td>
											</tr>
											
											<!-- Visibility -->
											<tr>
												<td class="py-1 text-gray-600 font-medium">Visibility</td>
												<td class="py-1">
													{formatVisibility(period.visibility)}
												</td>
											</tr>
										</tbody>
									</table>
								</div>
								
								<div>
									<table class="w-full text-sm">
										<tbody>
											<!-- Clouds -->
											<tr>
												<td class="py-1 text-gray-600 font-medium">Clouds</td>
												<td class="py-1">
													{#if period.cloud_layers && period.cloud_layers.length > 0}
														{#each period.cloud_layers as cloud, idx}
															<div class={idx > 0 ? 'mt-1' : ''}>
																{cloud.type || ''} at {cloud.altitude || cloud.height || 0} {cloud.unit || 'ft'}
															</div>
														{/each}
													{:else}
														N/A
													{/if}
												</td>
											</tr>
											
											<!-- Weather codes -->
											<tr>
												<td class="py-1 text-gray-600 font-medium">Weather</td>
												<td class="py-1">
													{#if period.wx_codes && period.wx_codes.length > 0}
														{#each period.wx_codes as code, idx}
															<span class="bg-blue-50 text-blue-700 text-xs px-2 py-0.5 rounded mr-1 mb-1 inline-block">
																{code}
															</span>
														{/each}
													{:else}
														None reported
													{/if}
												</td>
											</tr>
										</tbody>
									</table>
								</div>
							</div>
							
							{#if period.raw || period.sanitized}
								<div class="mt-3 pt-3 border-t border-gray-200">
									<div class="text-xs font-medium text-gray-500 mb-1">Raw Segment</div>
									<pre class="text-xs font-mono bg-gray-50 p-2 rounded overflow-x-auto">
										{period.raw || period.sanitized}
									</pre>
								</div>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{:else if result.forecast && result.forecast.length > 0}
			<!-- Fallback for old response format -->
			<h3 class="text-lg font-medium text-gray-800 mb-3">Forecast Periods</h3>
			<div class="space-y-4 mb-4">
				{#each result.forecast as period, i}
					<div class="bg-gray-50 p-4 rounded-lg border border-gray-200">
						{#if period.time || period.timeFrom || period.from || period.time_from}
							<h5 class="text-sm font-medium text-gray-700 mb-2">
								Timeframe: {formatDate(period.time || period.timeFrom || period.from || period.time_from)} 
								{#if period.timeTo || period.to || period.time_to}
									to {formatDate(period.timeTo || period.to || period.time_to)}
								{/if}
							</h5>
						{/if}
						
						<table class="w-full text-sm">
							<tbody>
								{#if period.wind || period.wind_speed}
									<tr>
										<td class="py-1 text-gray-600">Wind</td>
										<td class="py-1">
											{period.wind || 
											`${period.wind_direction?.value || '-'}° at ${period.wind_speed?.value || '-'} ${period.wind_speed?.unit || 'kt'}${period.wind_gust?.value ? ` gusting ${period.wind_gust.value} ${period.wind_gust.unit || 'kt'}` : ''}`}
										</td>
									</tr>
								{/if}
								{#if period.visibility || period.visibility?.value}
									<tr>
										<td class="py-1 text-gray-600">Visibility</td>
										<td class="py-1">
											{period.visibility?.value 
												? `${period.visibility.value} ${period.visibility.unit || 'sm'}`
												: period.visibility || '-'}
										</td>
									</tr>
								{/if}
								{#if period.clouds || (period.cloud_layers && period.cloud_layers.length > 0)}
									<tr>
										<td class="py-1 text-gray-600">Clouds</td>
										<td class="py-1">
											{#if period.clouds}
												{period.clouds}
											{:else if period.cloud_layers}
												{#each period.cloud_layers as cloud, idx}
													{#if idx > 0}, {/if}
													{cloud.type} at {cloud.altitude || cloud.height} {cloud.unit || 'ft'}
												{/each}
											{:else}
												-
											{/if}
										</td>
									</tr>
								{/if}
								{#if period.weather || period.wx_codes}
									<tr>
										<td class="py-1 text-gray-600">Weather</td>
										<td class="py-1">
											{period.weather || period.wx_codes?.join(', ') || '-'}
										</td>
									</tr>
								{/if}
							</tbody>
						</table>
					</div>
				{/each}
			</div>
		{:else}
			<div class="bg-yellow-50 border-l-4 border-yellow-500 p-4">
				<div class="flex">
					<div class="flex-shrink-0">
						<svg class="h-5 w-5 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
						</svg>
					</div>
					<div class="ml-3">
						<p class="text-sm text-yellow-700">
							No forecast periods available. This could be because the station doesn't have TAF service or there was an issue processing the data.
						</p>
					</div>
				</div>
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
	{/if}
</div>
