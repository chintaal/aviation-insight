<script>
	import { onMount } from 'svelte';
	import * as Icons from 'lucide-svelte';
	
	let station = $state('KPHX');
	let loading = $state(false);
	let error = $state(null);
	let summaryData = $state(null);
	
	async function fetchSummary() {
		if (!station || station.trim().length === 0) {
			error = 'Please enter an airport code';
			return;
		}
		
		loading = true;
		error = null;
		summaryData = null;
		
		try {
			const response = await fetch(`/api/v1/airport-summary/${station.toUpperCase()}`);
			
			if (!response.ok) {
				const errorText = await response.text();
				throw new Error(`Error ${response.status}: ${errorText}`);
			}
			
			const data = await response.json();
			summaryData = data;
		} catch (err) {
			error = err.message || 'Failed to fetch airport summary';
			console.error("Summary fetch error:", err);
		} finally {
			loading = false;
		}
	}
	
	onMount(() => {
		// Optionally auto-fetch on mount
		// fetchSummary();
	});
	
	function getFlightCategoryColor(category) {
		if (!category) return 'gray';
		const cat = category.toUpperCase();
		if (cat === 'VFR') return 'green';
		if (cat === 'MVFR') return 'blue';
		if (cat === 'IFR') return 'red';
		if (cat === 'LIFR') return 'purple';
		return 'gray';
	}
	
	function formatTimestamp(timestamp) {
		if (!timestamp) return 'N/A';
		try {
			const date = new Date(timestamp * 1000);
			return date.toLocaleString();
		} catch {
			return timestamp;
		}
	}
</script>

<svelte:head>
	<title>Airport Weather Summary - {station}</title>
</svelte:head>

<div class="container mx-auto max-w-7xl px-4 py-8">
	<!-- Header Section -->
	<div class="mb-8">
		<div class="flex items-center gap-3 mb-4">
			<Icons.Plane class="w-10 h-10 text-blue-600" />
			<h1 class="text-4xl font-bold text-gray-800">Comprehensive Airport Weather Summary</h1>
		</div>
		<p class="text-gray-600 text-lg">
			Get a complete AI-powered analysis of all weather reports for any airport
		</p>
	</div>
	
	<!-- Search Section -->
	<div class="bg-white rounded-xl shadow-lg p-6 mb-8">
		<div class="flex flex-wrap gap-4 items-end">
			<div class="flex-1 min-w-[200px]">
				<label for="station" class="block text-sm font-medium text-gray-700 mb-2">
					<Icons.MapPin class="w-4 h-4 inline mr-1" />
					Airport Code (ICAO)
				</label>
				<input
					type="text"
					id="station"
					bind:value={station}
					placeholder="e.g. KPHX, KJFK, KATL"
					class="block w-full rounded-lg border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-lg px-4 py-3"
					onkeydown={(e) => e.key === 'Enter' && fetchSummary()}
					maxlength="4"
				/>
			</div>
			<button
				onclick={fetchSummary}
				disabled={loading}
				class="px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-semibold rounded-lg shadow-lg hover:from-blue-700 hover:to-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
			>
				{#if loading}
					<Icons.Loader2 class="w-5 h-5 animate-spin" />
					Generating Summary...
				{:else}
					<Icons.Search class="w-5 h-5" />
					Generate Summary
				{/if}
			</button>
		</div>
	</div>
	
	<!-- Error Display -->
	{#if error}
		<div class="bg-red-50 border-l-4 border-red-500 p-6 rounded-lg mb-8">
			<div class="flex items-start">
				<Icons.AlertCircle class="w-6 h-6 text-red-500 mr-3 flex-shrink-0 mt-0.5" />
				<div>
					<h3 class="text-lg font-semibold text-red-800 mb-1">Error</h3>
					<p class="text-red-700">{error}</p>
				</div>
			</div>
		</div>
	{/if}
	
	<!-- Loading State -->
	{#if loading}
		<div class="bg-white rounded-xl shadow-lg p-12 text-center">
			<Icons.Loader2 class="w-16 h-16 text-blue-600 animate-spin mx-auto mb-4" />
			<p class="text-xl text-gray-600">Generating comprehensive weather summary...</p>
			<p class="text-sm text-gray-500 mt-2">This may take a few moments</p>
		</div>
	{/if}
	
	<!-- Summary Display -->
	{#if summaryData && !loading}
		<!-- Executive Overview -->
		<div class="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl shadow-lg p-8 mb-8 border border-blue-100">
			<div class="flex items-center gap-3 mb-4">
				<Icons.FileText class="w-8 h-8 text-blue-600" />
				<h2 class="text-3xl font-bold text-gray-800">Executive Overview</h2>
			</div>
			<div class="bg-white rounded-lg p-6 shadow-sm">
				<p class="text-lg text-gray-700 leading-relaxed">
					{summaryData.summary?.overview || 'Comprehensive weather summary available. Review detailed sections below.'}
				</p>
			</div>
		</div>
		
		<!-- Current Conditions -->
		<div class="bg-white rounded-xl shadow-lg p-8 mb-8">
			<div class="flex items-center gap-3 mb-6">
				<Icons.Cloud class="w-8 h-8 text-blue-600" />
				<h2 class="text-3xl font-bold text-gray-800">Current Conditions</h2>
			</div>
			
			{#if summaryData.reports?.metar}
				{@const metar = summaryData.reports.metar}
				{@const conditions = summaryData.summary?.current_conditions}
				{@const categoryColor = getFlightCategoryColor(metar.flight_category)}
				
				<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
					<!-- Flight Category -->
					<div class="rounded-lg p-6 border-2 {
						categoryColor === 'green' ? 'bg-gradient-to-br from-green-50 to-green-100 border-green-300' :
						categoryColor === 'blue' ? 'bg-gradient-to-br from-blue-50 to-blue-100 border-blue-300' :
						categoryColor === 'red' ? 'bg-gradient-to-br from-red-50 to-red-100 border-red-300' :
						categoryColor === 'purple' ? 'bg-gradient-to-br from-purple-50 to-purple-100 border-purple-300' :
						'bg-gradient-to-br from-gray-50 to-gray-100 border-gray-300'
					}">
						<div class="flex items-center justify-between mb-2">
							<Icons.Eye class="w-6 h-6 {
								categoryColor === 'green' ? 'text-green-700' :
								categoryColor === 'blue' ? 'text-blue-700' :
								categoryColor === 'red' ? 'text-red-700' :
								categoryColor === 'purple' ? 'text-purple-700' :
								'text-gray-700'
							}" />
							<span class="px-3 py-1 text-white text-sm font-bold rounded-full {
								categoryColor === 'green' ? 'bg-green-600' :
								categoryColor === 'blue' ? 'bg-blue-600' :
								categoryColor === 'red' ? 'bg-red-600' :
								categoryColor === 'purple' ? 'bg-purple-600' :
								'bg-gray-600'
							}">
								{metar.flight_category || 'N/A'}
							</span>
						</div>
						<p class="text-sm text-gray-600 mb-1">Flight Category</p>
						<p class="text-xs text-gray-500">{conditions?.summary || 'Current flight rules'}</p>
					</div>
					
					<!-- Visibility -->
					<div class="bg-gray-50 rounded-lg p-6 border border-gray-200">
						<div class="flex items-center gap-2 mb-2">
							<Icons.Eye class="w-6 h-6 text-gray-700" />
							<span class="text-2xl font-bold text-gray-800">{metar.visibility || 'N/A'}</span>
							<span class="text-sm text-gray-600">SM</span>
						</div>
						<p class="text-sm text-gray-600">Visibility</p>
					</div>
					
					<!-- Ceiling -->
					<div class="bg-gray-50 rounded-lg p-6 border border-gray-200">
						<div class="flex items-center gap-2 mb-2">
							<Icons.Cloud class="w-6 h-6 text-gray-700" />
							<span class="text-2xl font-bold text-gray-800">{metar.ceiling || 'N/A'}</span>
							<span class="text-sm text-gray-600">ft</span>
						</div>
						<p class="text-sm text-gray-600">Ceiling</p>
					</div>
					
					<!-- Wind -->
					<div class="bg-gray-50 rounded-lg p-6 border border-gray-200">
						<div class="flex items-center gap-2 mb-2">
							<Icons.Wind class="w-6 h-6 text-gray-700" />
							<span class="text-2xl font-bold text-gray-800">
								{metar.wind_direction || 'VRB'}° / {metar.wind_speed || '0'} kts
							</span>
						</div>
						<p class="text-sm text-gray-600">Wind</p>
					</div>
				</div>
				
				<!-- Detailed METAR Info -->
				<div class="bg-gray-50 rounded-lg p-6 mb-4">
					<h3 class="font-semibold text-gray-800 mb-3 flex items-center gap-2">
						<Icons.Info class="w-5 h-5 text-blue-600" />
						Detailed Analysis
					</h3>
					<div class="space-y-2 text-sm text-gray-700">
						{#if conditions?.winds}
							<p><strong>Winds:</strong> {conditions.winds}</p>
						{/if}
						{#if conditions?.weather}
							<p><strong>Weather:</strong> {conditions.weather}</p>
						{/if}
						{#if conditions?.temperature}
							<p><strong>Temperature:</strong> {conditions.temperature}°C</p>
						{/if}
						{#if conditions?.summary}
							<p class="mt-3 pt-3 border-t border-gray-300">{conditions.summary}</p>
						{/if}
					</div>
				</div>
				
				<!-- Raw METAR -->
				<div class="bg-blue-50 rounded-lg p-4 border border-blue-200">
					<code class="text-sm text-blue-900 font-mono">{metar.raw_text || 'No raw METAR available'}</code>
				</div>
			{:else}
				<div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
					<p class="text-yellow-800">No METAR data available for this airport</p>
				</div>
			{/if}
		</div>
		
		<!-- Forecast Outlook -->
		<div class="bg-white rounded-xl shadow-lg p-8 mb-8">
			<div class="flex items-center gap-3 mb-6">
				<Icons.Calendar class="w-8 h-8 text-indigo-600" />
				<h2 class="text-3xl font-bold text-gray-800">Forecast Outlook</h2>
			</div>
			
			{#if summaryData.reports?.taf}
				{@const taf = summaryData.reports.taf}
				{@const forecast = summaryData.summary?.forecast_outlook}
				
				<div class="bg-indigo-50 rounded-lg p-6 mb-4 border border-indigo-200">
					<h3 class="font-semibold text-indigo-800 mb-3 flex items-center gap-2">
						<Icons.Clock class="w-5 h-5" />
						Forecast Summary
					</h3>
					<p class="text-gray-700 mb-4">{forecast?.summary || 'TAF forecast available. Review detailed timeline below.'}</p>
					
					{#if forecast?.timeline}
						<div class="bg-white rounded p-4 mb-4">
							<p class="text-sm text-gray-600 mb-2"><strong>Timeline:</strong></p>
							<p class="text-gray-700">{forecast.timeline}</p>
						</div>
					{/if}
					
					{#if forecast?.ifr_periods}
						<div class="bg-red-50 rounded p-4 mb-4 border border-red-200">
							<p class="text-sm text-red-800 mb-2"><strong>IFR Periods:</strong></p>
							<p class="text-red-700">{forecast.ifr_periods}</p>
						</div>
					{/if}
					
					{#if forecast?.significant_changes}
						<div class="bg-yellow-50 rounded p-4 border border-yellow-200">
							<p class="text-sm text-yellow-800 mb-2"><strong>Significant Changes:</strong></p>
							<p class="text-yellow-700">{forecast.significant_changes}</p>
						</div>
					{/if}
				</div>
				
				<!-- Raw TAF -->
				<div class="bg-indigo-50 rounded-lg p-4 border border-indigo-200">
					<code class="text-sm text-indigo-900 font-mono whitespace-pre-wrap">{taf.raw_text || 'No raw TAF available'}</code>
				</div>
			{:else}
				<div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
					<p class="text-yellow-800">No TAF data available for this airport</p>
				</div>
			{/if}
		</div>
		
		<!-- Hazards Assessment -->
		<div class="bg-white rounded-xl shadow-lg p-8 mb-8">
			<div class="flex items-center gap-3 mb-6">
				<Icons.AlertTriangle class="w-8 h-8 text-red-600" />
				<h2 class="text-3xl font-bold text-gray-800">Hazard Assessment</h2>
			</div>
			
			{#if summaryData}
				{@const hazards = summaryData.summary?.hazards}
				{@const pireps = summaryData.reports?.pireps || []}
				{@const sigmets = summaryData.reports?.sigmets || []}
				
				<div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
				<!-- Turbulence -->
				<div class="bg-orange-50 rounded-lg p-6 border border-orange-200">
					<div class="flex items-center gap-2 mb-3">
						<Icons.Wind class="w-6 h-6 text-orange-700" />
						<h3 class="font-semibold text-orange-800">Turbulence</h3>
					</div>
					<p class="text-sm text-gray-700 mb-2">{hazards?.turbulence || 'No turbulence reports'}</p>
					{#if hazards?.turbulence && hazards.turbulence !== 'No turbulence reports'}
						<p class="text-xs text-gray-600 mt-2">{hazards.summary || 'Review PIREPs for details'}</p>
					{/if}
				</div>
				
				<!-- Icing -->
				<div class="bg-blue-50 rounded-lg p-6 border border-blue-200">
					<div class="flex items-center gap-2 mb-3">
						<Icons.Snowflake class="w-6 h-6 text-blue-700" />
						<h3 class="font-semibold text-blue-800">Icing</h3>
					</div>
					<p class="text-sm text-gray-700 mb-2">{hazards?.icing || 'No icing reports'}</p>
					{#if hazards?.icing && hazards.icing !== 'No icing reports'}
						<p class="text-xs text-gray-600 mt-2">{hazards.summary || 'Review PIREPs for details'}</p>
					{/if}
				</div>
				
				<!-- Thunderstorms -->
				<div class="bg-purple-50 rounded-lg p-6 border border-purple-200">
					<div class="flex items-center gap-2 mb-3">
						<Icons.Zap class="w-6 h-6 text-purple-700" />
						<h3 class="font-semibold text-purple-800">Thunderstorms</h3>
					</div>
					<p class="text-sm text-gray-700 mb-2">{hazards?.thunderstorms || 'No thunderstorm reports'}</p>
					{#if hazards?.thunderstorms && hazards.thunderstorms !== 'No thunderstorm reports'}
						<p class="text-xs text-gray-600 mt-2">{hazards.summary || 'Review SIGMETs for details'}</p>
					{/if}
				</div>
				
				<!-- Other Hazards -->
				<div class="bg-gray-50 rounded-lg p-6 border border-gray-200">
					<div class="flex items-center gap-2 mb-3">
						<Icons.AlertCircle class="w-6 h-6 text-gray-700" />
						<h3 class="font-semibold text-gray-800">Other Hazards</h3>
					</div>
					<p class="text-sm text-gray-700 mb-2">{hazards?.other || 'No other hazards reported'}</p>
					{#if hazards?.other && hazards.other !== 'No other hazards reported'}
						<p class="text-xs text-gray-600 mt-2">{hazards.summary || 'Review all reports for details'}</p>
					{/if}
				</div>
			</div>
			
			{#if hazards?.summary}
				<div class="bg-red-50 rounded-lg p-4 border-l-4 border-red-500">
					<p class="text-sm text-red-800"><strong>Hazard Summary:</strong> {hazards.summary}</p>
				</div>
			{/if}
			
			<!-- PIREPs List -->
			{#if pireps.length > 0}
				<div class="mt-6">
					<h3 class="font-semibold text-gray-800 mb-4 flex items-center gap-2">
						<Icons.Plane class="w-5 h-5 text-blue-600" />
						Pilot Reports ({pireps.length})
					</h3>
					<div class="space-y-3">
						{#each pireps.slice(0, 5) as pirep}
							<div class="bg-gray-50 rounded-lg p-4 border border-gray-200">
								<div class="flex items-start justify-between mb-2">
									<div>
										<p class="font-medium text-gray-800">{pirep.location || 'Unknown'}</p>
										<p class="text-xs text-gray-600">Alt: {pirep.altitude || 'N/A'} | Type: {pirep.aircraft_type || 'N/A'}</p>
									</div>
									{#if pirep.report_type === 'UUA'}
										<span class="px-2 py-1 bg-red-100 text-red-800 text-xs font-semibold rounded">URGENT</span>
									{/if}
								</div>
								<code class="text-xs text-gray-600 font-mono">{pirep.raw_text?.substring(0, 150) || 'No details'}</code>
							</div>
						{/each}
					</div>
				</div>
			{/if}
			
			<!-- SIGMETs List -->
			{#if sigmets.length > 0}
				<div class="mt-6">
					<h3 class="font-semibold text-gray-800 mb-4 flex items-center gap-2">
						<Icons.AlertTriangle class="w-5 h-5 text-red-600" />
						Weather Advisories ({sigmets.length})
					</h3>
					<div class="space-y-3">
						{#each sigmets.slice(0, 5) as sigmet}
							<div class="bg-red-50 rounded-lg p-4 border border-red-200">
								<div class="flex items-start justify-between mb-2">
									<div>
										<p class="font-medium text-red-800">{sigmet.phenomenon || 'Weather Advisory'}</p>
										<p class="text-xs text-red-600">Valid until: {sigmet.valid_to || 'N/A'}</p>
									</div>
								</div>
								<code class="text-xs text-red-700 font-mono">{sigmet.raw_text?.substring(0, 150) || 'No details'}</code>
							</div>
						{/each}
					</div>
				</div>
			{/if}
			{/if}
		</div>
		
		<!-- Operational Recommendations -->
		<div class="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl shadow-lg p-8 mb-8 border border-green-200">
			<div class="flex items-center gap-3 mb-6">
				<Icons.CheckCircle class="w-8 h-8 text-green-600" />
				<h2 class="text-3xl font-bold text-gray-800">Operational Recommendations</h2>
			</div>
			
			{#if summaryData}
				{@const recommendations = summaryData.summary?.recommendations}
				
				<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
				{#if recommendations?.flight_planning}
					<div class="bg-white rounded-lg p-6 shadow-sm">
						<div class="flex items-center gap-2 mb-3">
							<Icons.Map class="w-6 h-6 text-green-600" />
							<h3 class="font-semibold text-gray-800">Flight Planning</h3>
						</div>
						<p class="text-sm text-gray-700">{recommendations.flight_planning}</p>
					</div>
				{/if}
				
				{#if recommendations?.timing}
					<div class="bg-white rounded-lg p-6 shadow-sm">
						<div class="flex items-center gap-2 mb-3">
							<Icons.Clock class="w-6 h-6 text-green-600" />
							<h3 class="font-semibold text-gray-800">Timing</h3>
						</div>
						<p class="text-sm text-gray-700">{recommendations.timing}</p>
					</div>
				{/if}
				
				{#if recommendations?.altitude}
					<div class="bg-white rounded-lg p-6 shadow-sm">
						<div class="flex items-center gap-2 mb-3">
							<Icons.TrendingUp class="w-6 h-6 text-green-600" />
							<h3 class="font-semibold text-gray-800">Altitude</h3>
						</div>
						<p class="text-sm text-gray-700">{recommendations.altitude}</p>
					</div>
				{/if}
				
				{#if recommendations?.equipment}
					<div class="bg-white rounded-lg p-6 shadow-sm">
						<div class="flex items-center gap-2 mb-3">
							<Icons.Settings class="w-6 h-6 text-green-600" />
							<h3 class="font-semibold text-gray-800">Equipment</h3>
						</div>
						<p class="text-sm text-gray-700">{recommendations.equipment}</p>
					</div>
				{/if}
			</div>
			
			{#if recommendations?.risk_assessment}
				<div class="mt-6 bg-yellow-50 rounded-lg p-6 border-l-4 border-yellow-500">
					<div class="flex items-center gap-2 mb-2">
						<Icons.Shield class="w-6 h-6 text-yellow-700" />
						<h3 class="font-semibold text-yellow-800">Risk Assessment</h3>
					</div>
					<p class="text-sm text-yellow-800">{recommendations.risk_assessment}</p>
				</div>
			{/if}
			{/if}
		</div>
		
		<!-- Metadata -->
		<div class="bg-gray-50 rounded-lg p-4 text-center text-sm text-gray-600">
			<p>Summary generated at {formatTimestamp(summaryData.timestamp)}</p>
			<p class="mt-1">Station: {summaryData.station} | Distance: {summaryData.metadata?.distance || 'N/A'} nm | Age: {summaryData.metadata?.age || 'N/A'} hrs</p>
		</div>
	{/if}
	
	<!-- Empty State -->
	{#if !summaryData && !loading && !error}
		<div class="bg-white rounded-xl shadow-lg p-12 text-center">
			<Icons.Plane class="w-20 h-20 text-gray-400 mx-auto mb-4" />
			<h3 class="text-2xl font-semibold text-gray-800 mb-2">Ready to Generate Summary</h3>
			<p class="text-gray-600 mb-6">Enter an airport code above to get a comprehensive AI-powered weather analysis</p>
			<div class="flex flex-wrap justify-center gap-2 text-sm text-gray-500">
				<span class="px-3 py-1 bg-gray-100 rounded-full">METAR</span>
				<span class="px-3 py-1 bg-gray-100 rounded-full">TAF</span>
				<span class="px-3 py-1 bg-gray-100 rounded-full">PIREP</span>
				<span class="px-3 py-1 bg-gray-100 rounded-full">SIGMET</span>
				<span class="px-3 py-1 bg-gray-100 rounded-full">AI Analysis</span>
			</div>
		</div>
	{/if}
</div>

<style>
	/* Custom styles for better visual appearance */
	:global(body) {
		background: linear-gradient(to bottom, #f9fafb, #f3f4f6);
	}
</style>
