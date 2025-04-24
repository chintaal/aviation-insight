<script>
	import { onMount } from 'svelte';
	import TafHoverDisplay from '$lib/components/TafHoverDisplay.svelte';

	// State management
	let station = 'KPHX';
	let loading = false;
	let error = null;
	let result = null;
	let showAdvanced = false; // Toggle for advanced details
	let hoveredTafSection = null; // Track which section is being hovered
	let selectedPeriod = null; // Track which forecast period is selected

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

	// Format time for TAF periods
	function formatTafTime(timestamp) {
		if (!timestamp) return 'N/A';
		const date = new Date(timestamp);
		return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + 'Z ' + 
			date.toLocaleDateString([], { day: 'numeric', month: 'short' });
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

	// Parse TAF into interactive sections for highlighting
	function parseTafStringIntoSections(tafString) {
		if (!tafString) return [];
		
		const sections = [];
		const parts = tafString.split(' ');
		
		// Definitions for TAF parts with simplified patterns
		const tafSectionDefinitions = [
			{ type: 'reportType', pattern: /^TAF$/ },
			{ type: 'station', pattern: /^[A-Z]{4}$/ },
			{ type: 'time', pattern: /^\d{6}Z$/ },
			{ type: 'validity', pattern: /^\d{4}\/\d{4}$/ },
			{ type: 'wind', pattern: /^(00000|VRB\d{2}|\d{3})\d{2}(G\d{2})?(KT|MPS)$/ },
			{ type: 'variable_wind', pattern: /^\d{3}V\d{3}$/ },
			{ type: 'visibility', pattern: /^(?:\d{4}|CAVOK|[MP]?\d+(?:\s+\d+\/\d+)?SM)$/ },
			{ type: 'weather', pattern: /^(?:\+|-|VC|RE)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|FG|BR|SA|DU|HZ|FU|VA|PY|PO|SQ|FC|SS|DS){1,3}$/ },
			{ type: 'cloud', pattern: /^(?:SKC|CLR|NSC|NCD|FEW|SCT|BKN|OVC|VV)(?:\d{3})?(?:CB|TCU)?$/ },
			{ type: 'temp', pattern: /^T[M]?\d{2}\/\d{2}Z$/ },
			{ type: 'changeIndicator', pattern: /^(?:BECMG|TEMPO|FM\d{6}|PROB\d{2})$/ },
			{ type: 'changeTime', pattern: /^\d{4}\/\d{4}$/ },
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
			
			for (const def of tafSectionDefinitions) {
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

	// Format TAF forecast period for display
	function formatForecastPeriod(period) {
		if (!period) return "Not available";
		
		// Check if we have valid date times
		let startTime, endTime;
		
		// Handle parsed_taf format
		if (period.valid_from && period.valid_to) {
			startTime = new Date(period.valid_from);
			endTime = new Date(period.valid_to);
		} 
		// Handle AVWX format
		else if (period.start_time && period.end_time) {
			if (typeof period.start_time === 'string') {
				startTime = new Date(period.start_time);
			} else if (period.start_time.dt) {
				startTime = new Date(period.start_time.dt);
			}
			
			if (typeof period.end_time === 'string') {
				endTime = new Date(period.end_time);
			} else if (period.end_time.dt) {
				endTime = new Date(period.end_time.dt);
			}
		}
		
		if (startTime && endTime) {
			return `${startTime.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}Z to ${endTime.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}Z (${startTime.toLocaleDateString([], {day: 'numeric', month: 'short'})})`;
		}
		
		return "Time period not specified";
	}

	// Get forecast data mapped to categories for display
	function getForecastPeriodData(period) {
		if (!period) return {};
		
		console.log("Processing period:", period);
		
		// Determine flight rules from either direct property or derived from ceiling/visibility
		const flightRules = period.flight_rules || period.flight_rules_details;
		
		// Get wind data - handle multiple possible formats
		let windData = null;
		
		// First try wind as complete object
		if (period.wind) {
			if (typeof period.wind === 'object') {
				windData = period.wind;
			} else if (typeof period.wind === 'string') {
				windData = { text: period.wind };
			}
		} 
		
		// Try individual wind components if not found as object
		if (!windData) {
			// Look for direct wind properties
			if (period.wind_dir_degrees !== undefined || period.wind_speed_kt !== undefined || 
				period.wind_direction !== undefined || period.wind_speed !== undefined) {
				windData = {
					direction: period.wind_dir_degrees !== undefined ? period.wind_dir_degrees : period.wind_direction,
					speed: period.wind_speed_kt !== undefined ? period.wind_speed_kt : period.wind_speed,
					gust: period.wind_gust_kt || period.wind_gust || null,
					unit: 'kt'
				};
			}
		}
		
		// Get visibility data - handle multiple possible formats
		let visibilityData = null;
		
		// First check if visibility exists as an object
		if (period.visibility) {
			if (typeof period.visibility === 'object') {
				visibilityData = period.visibility;
			} else if (typeof period.visibility === 'string') {
				visibilityData = { text: period.visibility };
			} else if (typeof period.visibility === 'number') {
				visibilityData = { 
					distance: period.visibility, 
					unit: 'SM' 
				};
			}
		}
		
		// Try individual visibility properties if not found as object
		if (!visibilityData && period.visibility_statute_mi !== undefined) {
			visibilityData = {
				distance: period.visibility_statute_mi,
				unit: 'SM'
			};
		}
		
		// Get cloud data - handle multiple possible formats
		let cloudData = [];
		if (period.clouds && Array.isArray(period.clouds)) {
			cloudData = period.clouds;
		} else if (period.sky_condition && Array.isArray(period.sky_condition)) {
			cloudData = period.sky_condition.map(c => ({
				cover: c.sky_cover,
				base: c.cloud_base_ft_agl,
				type: c.cloud_type || null
			}));
		} else if (period.sky && typeof period.sky === 'string') {
			cloudData = [{ cover: period.sky }];
		}
		
		// Weather phenomena - handle multiple possible formats
		let weatherData = [];
		if (period.weather && Array.isArray(period.weather)) {
			weatherData = period.weather;
		} else if (period.wx_string) {
			if (Array.isArray(period.wx_string)) {
				weatherData = period.wx_string;
			} else if (typeof period.wx_string === 'string') {
				weatherData = [period.wx_string];
			}
		} else if (period.wx && typeof period.wx === 'string') {
			weatherData = [period.wx];
		}
		
		// Determine change type
		const changeType = period.type || "Base Forecast";
		
			// Format wind text with special handling for missing or variable data
		const formatWind = (data) => {
			if (!data) return 'Not specified';
			
			if (data.text) return data.text;
			
			if (data.speed === 0 || (typeof data.speed === 'string' && data.speed.toUpperCase() === 'CALM')) {
				return 'Wind calm';
			}
			
			// Handle case where direction or speed are objects instead of primitives
			const direction = data.direction;
			const directionText = direction === 'VRB' ? 'Variable' : 
				(direction !== undefined ? 
					(typeof direction === 'object' ? 
						(direction.toString ? direction.toString() : 'Variable') : 
						direction + '°') : 
					'unknown direction');
			
			const speed = data.speed;
			const speedText = speed !== undefined ? 
				(typeof speed === 'object' ? 
					(speed.toString ? speed.toString() : 'unknown') : 
					speed) : 
				'unknown speed';
			
			const unit = data.unit || 'kt';
			
			const gust = data.gust;
			const gustText = gust ? 
				`, gusting ${typeof gust === 'object' ? 
					(gust.toString ? gust.toString() : gust) : 
					gust} ${unit}` : 
				'';
			
			return `From ${directionText} at ${speedText} ${unit}${gustText}`;
		};
		
		// Format visibility text with appropriate handling
		const formatVisibility = (data) => {
			if (!data) return 'Not specified';
			
			if (data.text) return data.text;
			
			const distance = data.distance !== undefined ? 
				(typeof data.distance === 'object' ? 
					(data.distance.toString ? data.distance.toString() : 'unknown') : 
					data.distance) : 
				'unknown';
			
			const unit = data.unit || 'SM';
			
			return `${distance} ${unit}`;
		};
		
		// Compile the data object
		const periodData = {
			time: {
				value: formatForecastPeriod(period),
				icon: 'clock',
				title: 'Time Period',
				description: period.type ? `${period.type} period` : 'Valid time range',
				color: 'purple'
			},
			flight_category: {
				value: flightRules || 'Unknown',
				icon: 'flight',
				title: 'Flight Category',
				description: 'Visual flight rules category',
				color: flightRules === 'VFR' ? 'green' : 
					flightRules === 'MVFR' ? 'blue' : 
					flightRules === 'IFR' ? 'red' : 
					flightRules === 'LIFR' ? 'purple' : 'gray'
			},
			wind: {
				value: formatWind(windData),
				icon: 'wind',
				title: 'Wind',
				description: 'Wind direction and speed',
				color: 'cyan'
			},
			visibility: {
				value: formatVisibility(visibilityData),
				icon: 'visibility',
				title: 'Visibility',
				description: 'Horizontal visibility',
				color: 'amber'
			},
			weather: {
				value: Array.isArray(weatherData) && weatherData.length > 0 ? 
					(weatherData.map(w => w.text || w).join(', ')) : 
					(typeof weatherData === 'string' ? weatherData : 'No significant weather'),
				icon: 'weather',
				title: 'Weather',
				description: 'Weather phenomena',
				color: 'emerald'
			},
			clouds: {
				value: Array.isArray(cloudData) && cloudData.length > 0 ?
					cloudData.map(c => {
						const cover = c.cover_text || c.cover || c.sky_cover || '';
						const height = c.base || c.cloud_base_ft_agl || c.height || null;
						const type = c.type || c.cloud_type || null;
						
						return `${cover}${height ? ` at ${height} ft` : ''}${type ? ` (${type})` : ''}`;
					}).join(', ') :
					'No cloud information',
				icon: 'cloud',
				title: 'Clouds',
				description: 'Cloud coverage and height',
				color: 'indigo'
			}
		};
		
		// Add pilot insights if available
		if (period.pilot_insights || period.pilot_summary) {
			periodData.pilot_insights = {
				value: period.pilot_insights || period.pilot_summary || '',
				icon: 'note',
				title: 'Pilot Insights',
				description: 'Key information for pilots',
				color: 'blue'
			};
		}
		
		// Add planning considerations if available
		if (period.planning_considerations && period.planning_considerations.length > 0) {
			periodData.planning = {
				value: period.planning_considerations.join('; '),
				icon: 'plan',
				title: 'Planning',
				description: 'Flight planning considerations',
				color: 'orange'
			};
		}
		
		return periodData;
	}

	// Fetch TAF data from the API
	async function fetchTaf() {
		loading = true;
		error = null;
		result = null;

		try {
			const url = `/api/v1/taf/${station}`;
			console.log("Fetching TAF data from:", url);
			
			const response = await fetch(url);
			
			if (!response.ok) {
				throw new Error(`API error: ${response.status} ${response.statusText}`);
			}

			result = await response.json();
			console.log("TAF result:", result);
			
			// Parse the TAF string for highlighting
			if (result.raw_text) {
				result.parsedSections = parseTafStringIntoSections(result.raw_text);
			}
			
			// Set the first forecast period as selected by default
			if (result.forecast && result.forecast.length > 0) {
				selectedPeriod = 0;
			} else if (result.parsed_taf && result.parsed_taf.forecast_periods && result.parsed_taf.forecast_periods.length > 0) {
				selectedPeriod = 0;
			}
			
		} catch (err) {
			error = err.message || 'Failed to fetch TAF data';
			console.error("TAF fetch error:", err);
		} finally {
			loading = false;
		}
	}

	// Get the forecast periods from either source
	function getForecastPeriods() {
		if (!result) return [];
		
		// First try using the parsed_taf structure
		if (result.parsed_taf && result.parsed_taf.forecast_periods && result.parsed_taf.forecast_periods.length > 0) {
			return result.parsed_taf.forecast_periods;
		}
		
		// Fall back to the regular forecast property
		return result.forecast || [];
	}
	
	// Highlight a section in the TAF string when hovering
	function highlightTafSection(type) {
		if (!result || !result.parsedSections) return;
		hoveredTafSection = type;
	}
	
	function clearHighlight() {
		hoveredTafSection = null;
	}
	
	// Select a forecast period
	function selectPeriod(index) {
		selectedPeriod = index;
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
		note: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"></path></svg>`,
		plan: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"></path></svg>`,
		calendar: `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>`
	};

	// Auto-fetch data on component mount
	onMount(() => {
		fetchTaf();
	});
</script>

<svelte:head>
	<title>Enhanced TAF Display</title>
</svelte:head>

<div class="container mx-auto max-w-4xl">
	<div class="mb-6">
		<h1 class="text-2xl font-bold text-gray-800 mb-2">Enhanced TAF Display</h1>
		<p class="text-gray-600">
			Interactive Terminal Aerodrome Forecast visualization with detailed pilot insights.
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
				on:click={fetchTaf}
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
					Get Forecast
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
		<!-- TAF Header Information -->
		<div class="bg-white rounded-lg shadow p-6 mb-6">
			<div class="flex flex-wrap items-center justify-between gap-4">
				<div>
					<h2 class="text-xl font-bold text-gray-800">{result.station || 'Unknown Station'}</h2>
					<p class="text-sm text-gray-600">
						{#if result.valid_from && result.valid_to}
							Valid from {formatTafTime(result.valid_from)} to {formatTafTime(result.valid_to)}
						{:else}
							Validity period not available
						{/if}
					</p>
					{#if result.issue_time}
						<p class="text-xs text-gray-500">Issued: {formatTafTime(result.issue_time)}</p>
					{/if}
				</div>
				
				{#if result.pilot_summary}
					<div class="flex-1 min-w-[60%] bg-blue-50 p-3 rounded-md border border-blue-200">
						<p class="text-blue-800 font-medium leading-relaxed">{result.pilot_summary}</p>
					</div>
				{/if}
			</div>
		</div>
		
		<!-- Raw TAF with hover explanations -->
		{#if result.raw_text}
			<div class="bg-white rounded-lg shadow p-6 mb-6">
				<TafHoverDisplay tafString={result.raw_text} />
			</div>
		{/if}
		
		<!-- Forecast Periods -->
		{@const forecastPeriods = getForecastPeriods()}
		{#if forecastPeriods.length > 0}
			<div class="bg-white rounded-lg shadow overflow-hidden mb-6">
				<div class="p-4 bg-gray-50 border-b border-gray-200">
					<h2 class="text-lg font-semibold text-gray-700">Forecast Periods</h2>
					<p class="text-sm text-gray-600">Select a time period to view detailed forecast information</p>
				</div>
				
				<!-- Period tabs -->
				<div class="flex overflow-x-auto border-b border-gray-200 bg-gray-50 gap-1 p-1">
					{#each forecastPeriods as period, i}
						{@const type = period.type || (i === 0 ? "Base" : "Change")}
						{@const flightRules = period.flight_rules || period.flight_rules_details || ""}
						<button 
							class="px-4 py-2 text-sm font-medium rounded-md transition-colors whitespace-nowrap flex items-center gap-2"
							class:bg-indigo-600={selectedPeriod === i}
							class:text-white={selectedPeriod === i}
							class:bg-white={selectedPeriod !== i}
							class:text-gray-700={selectedPeriod !== i}
							on:click={() => selectPeriod(i)}
						>
							{#if type.includes("Base")}
								<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">
									Base
								</span>
							{:else if type.includes("From")}
								<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800">
									FM
								</span>
							{:else if type.includes("Becoming")}
								<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-800">
									BECMG
								</span>
							{:else if type.includes("Temporarily")}
								<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-orange-100 text-orange-800">
									TEMPO
								</span>
							{:else if type.includes("Probability")}
								<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800">
									PROB
								</span>
							{/if}
							
							{formatForecastPeriod(period).split(' to ')[0]}
							
							{#if flightRules}
								<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
									class:bg-green-100={flightRules === 'VFR'}
									class:text-green-800={flightRules === 'VFR'}
									class:bg-blue-100={flightRules === 'MVFR'}
									class:text-blue-800={flightRules === 'MVFR'}
									class:bg-red-100={flightRules === 'IFR'}
									class:text-red-800={flightRules === 'IFR'}
									class:bg-purple-100={flightRules === 'LIFR'}
									class:text-purple-800={flightRules === 'LIFR'}
								>
									{flightRules}
								</span>
							{/if}
						</button>
					{/each}
				</div>
				
				<!-- Selected period details -->
				{#if selectedPeriod !== null && forecastPeriods[selectedPeriod]}
					{@const period = forecastPeriods[selectedPeriod]}
					{@const periodData = getForecastPeriodData(period)}
					
					<div class="p-6">
						<h3 class="text-lg font-medium text-gray-900 mb-4">
							{period.type || (selectedPeriod === 0 ? "Base Forecast" : "Change Group")}
							<span class="text-base font-normal text-gray-600 ml-2">
								{formatForecastPeriod(period)}
							</span>
						</h3>
						
						<!-- Weather information blocks -->
						<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
							{#each Object.entries(periodData) as [key, category]}
								<div 
									class="bg-white rounded-lg shadow-md overflow-hidden border border-gray-200 hover:shadow-lg transition-shadow"
									on:mouseenter={() => highlightTafSection(key)}
									on:mouseleave={clearHighlight}
								>
									<div class="bg-{category.color}-100 p-3 border-b border-{category.color}-200 flex items-center">
										{@html icons[category.icon] || icons.note}
										<h3 class="ml-2 font-medium text-{category.color}-800">{category.title}</h3>
									</div>
									<div class="p-4">
										<p class="text-gray-800 font-medium">{category.value}</p>
										<p class="text-sm text-gray-500 mt-1">{category.description}</p>
									</div>
								</div>
							{/each}
						</div>
					</div>
				{/if}
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
					<pre class="text-xs">{JSON.stringify(result, null, 2)}</pre>
				</div>
			</div>
		{/if}
	{:else if !loading && !error}
		<!-- Empty state when no data is loaded -->
		<div class="bg-blue-50 border-blue-200 border rounded-lg p-6 text-center">
			<svg class="w-16 h-16 mx-auto text-blue-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
			</svg>
			<h3 class="text-lg font-medium text-blue-700 mb-2">No TAF Data</h3>
			<p class="text-blue-600">Enter an airport code and click "Get Forecast" to view terminal aerodrome forecast</p>
		</div>
	{/if}
</div>

<style>
	/* Add some color styling for the different TAF components */
	[data-type="reportType"] {
		font-weight: bold;
	}
	[data-type="station"] {
		color: #3b82f6; /* blue-500 */
	}
	[data-type="time"] {
		color: #8b5cf6; /* purple-500 */
	}
	[data-type="validity"] {
		color: #8b5cf6; /* purple-500 */
		font-weight: bold;
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
	[data-type="temp"] {
		color: #ef4444; /* red-500 */
	}
	[data-type="changeIndicator"] {
		color: #f97316; /* orange-500 */
		font-weight: bold;
	}
	[data-type="changeTime"] {
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