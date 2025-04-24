<script>
	// Props for the component
	export let tafString = '';

	// State management
	let hoveredSection = null;
	let parsedTafSections = [];

	// Process the TAF string whenever it changes
	$: {
		if (tafString) {
			parsedTafSections = parseTafStringIntoSections(tafString);
		} else {
			parsedTafSections = [];
		}
	}

	// Parse TAF into interactive sections
	function parseTafStringIntoSections(tafString) {
		if (!tafString) return [];
		
		const sections = [];
		const parts = tafString.split(' ');
		
		// Definitions for TAF parts
		const tafSectionDefinitions = [
			{
				type: 'reportType',
				pattern: /^TAF$|^TAF\s+AMD$/,
				description: 'Terminal Aerodrome Forecast report type'
			},
			{
				type: 'station',
				pattern: /^[A-Z]{4}$/,
				description: 'ICAO Airport Code'
			},
			{
				type: 'time',
				pattern: /^\d{6}Z$/,
				description: 'Issue time (DDHHMM in UTC) followed by Z'
			},
			{
				type: 'validity',
				pattern: /^\d{4}\/\d{4}$/,
				description: 'Validity period (from DDHH to DDHH in UTC)'
			},
			{
				type: 'wind',
				pattern: /^(00000|VRB\d{2}|\d{3})\d{2}(G\d{2})?(KT|MPS)$/,
				description: 'Wind direction (degrees), speed, gusts (if present), and unit (KT or MPS)'
			},
			{
				type: 'variable_wind',
				pattern: /^\d{3}V\d{3}$/,
				description: 'Variable wind direction range (degrees)'
			},
			{
				type: 'visibility',
				pattern: /^(?:\d{4}|CAVOK|[MP]?\d+(?:\s+\d+\/\d+)?SM)$/,
				description: 'Visibility in statute miles (SM) or meters'
			},
			{
				type: 'weather',
				pattern: /^(?:\+|-|VC|RE)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|FG|BR|SA|DU|HZ|FU|VA|PY|PO|SQ|FC|SS|DS){1,3}$/,
				description: 'Weather phenomena (intensity, descriptor, precipitation, obscuration, other)'
			},
			{
				type: 'cloud',
				pattern: /^(?:SKC|CLR|NSC|NCD|FEW|SCT|BKN|OVC|VV)(?:\d{3})?(?:CB|TCU)?$/,
				description: 'Cloud coverage, height (hundreds of feet), and type'
			},
			{
				type: 'changeIndicator',
				pattern: /^(?:BECMG|TEMPO|PROB\d{2}|PROB\d{2}\s+TEMPO|FM\d{6})$/,
				description: 'Change indicator showing when forecast conditions will change'
			},
			{
				type: 'changeTime',
				pattern: /^\d{4}\/\d{4}$/,
				description: 'Time period for forecast change (from DDHH to DDHH in UTC)'
			},
			{
				type: 'temp',
				pattern: /^T[M]?\d{2}\/[M]?\d{2}$/,
				description: 'Maximum/minimum temperature forecast'
			},
			{
				type: 'wind_shear',
				pattern: /^WS\d{3}\/\d{3}KT$/,
				description: 'Wind shear at specified height'
			},
			{
				type: 'remarks_indicator',
				pattern: /^RMK$/,
				description: 'Remarks section indicator'
			},
			{
				type: 'remarks',
				pattern: /./,
				description: 'Additional remarks and coded information'
			}
		];
		
		let currentIndex = 0;
		let remarksStarted = false;
		let changeIndicator = false;
		
		parts.forEach(part => {
			if (part === 'RMK') {
				remarksStarted = true;
				sections.push({
					text: part,
					type: 'remarks_indicator',
					description: 'Remarks section indicator'
				});
				return;
			}
			
			if (remarksStarted) {
				sections.push({
					text: part,
					type: 'remarks',
					description: 'Additional remarks and coded information'
				});
				return;
			}
			
			// Special handling for change indicators
			if (part === 'BECMG' || part === 'TEMPO' || part.startsWith('PROB') || part.startsWith('FM')) {
				changeIndicator = true;
				sections.push({
					text: part,
					type: 'changeIndicator',
					description: `Change indicator: ${getChangeDescription(part)}`
				});
				return;
			}
			
			// Match the part against our patterns
			let matched = false;
			
			for (const def of tafSectionDefinitions) {
				if (def.pattern.test(part)) {
					sections.push({
						text: part,
						type: def.type,
						description: def.description
					});
					matched = true;
					break;
				}
			}
			
			if (!matched) {
				// Special handling for NSW (No Significant Weather)
				if (part === 'NSW') {
					sections.push({
						text: part,
						type: 'weather',
						description: 'No Significant Weather'
					});
				} else {
					// If no match, use a generic type
					sections.push({
						text: part,
						type: 'unknown',
						description: 'Unrecognized TAF component'
					});
				}
			}
		});
		
		return sections;
	}
	
	// Get description for change indicators
	function getChangeDescription(indicator) {
		if (indicator === 'BECMG') {
			return 'Becoming: Permanent change expected during the indicated period';
		} else if (indicator === 'TEMPO') {
			return 'Temporarily: Temporary fluctuations expected during the indicated period';
		} else if (indicator.startsWith('PROB')) {
			const prob = indicator.substring(4);
			return `Probability ${prob}%: Conditions have ${prob}% chance of occurring`;
		} else if (indicator.startsWith('FM')) {
			const day = indicator.substring(2, 4);
			const hour = indicator.substring(4, 6);
			const minute = indicator.substring(6, 8);
			return `From: Rapid change expected at ${day}/${hour}:${minute}Z`;
		}
		return 'Change indicator';
	}

	// Get more detailed description for a TAF section based on content and type
	function getDetailedDescription(section) {
		const { text, type } = section;
		
		switch (type) {
			case 'reportType':
				if (text.includes('AMD')) {
					return `Report Type: TAF AMD - Terminal Aerodrome Forecast, Amended`;
				}
				return `Report Type: TAF - Terminal Aerodrome Forecast`;
				
			case 'station':
				return `Station Identifier: ${text} - ICAO airport code.`;
				
			case 'time':
				const day = text.substring(0, 2);
				const hour = text.substring(2, 4);
				const minute = text.substring(4, 6);
				return `Issue Time: Day ${day}, ${hour}:${minute} UTC`;
				
			case 'validity':
				const [fromStr, toStr] = text.split('/');
				const fromDay = fromStr.substring(0, 2);
				const fromHour = fromStr.substring(2, 4);
				const toDay = toStr.substring(0, 2);
				const toHour = toStr.substring(2, 4);
				return `Validity Period: From day ${fromDay} at ${fromHour}:00 UTC to day ${toDay} at ${toHour}:00 UTC`;
				
			case 'wind':
				if (text === '00000KT') {
					return 'Wind: Calm (0 knots)';
				} else if (text.startsWith('VRB')) {
					const speed = parseInt(text.match(/VRB(\d{2})/)[1], 10);
					const gust = text.includes('G') ? parseInt(text.match(/G(\d{2})/)[1], 10) : null;
					return `Wind: Variable direction at ${speed} knots${gust ? `, gusting to ${gust} knots` : ''}`;
				} else {
					const dir = parseInt(text.substring(0, 3), 10);
					const speed = parseInt(text.substring(3, 5), 10);
					const gustMatch = text.match(/G(\d{2})/);
					const gust = gustMatch ? parseInt(gustMatch[1], 10) : null;
					const unit = text.endsWith('KT') ? 'knots' : 'meters per second';
					return `Wind: From ${dir}° at ${speed} ${unit}${gust ? `, gusting to ${gust} ${unit}` : ''}`;
				}
				
			case 'variable_wind':
				const [from, to] = text.split('V').map(v => parseInt(v, 10));
				return `Variable Wind Direction: Varying between ${from}° and ${to}°`;
				
			case 'visibility':
				if (text === 'CAVOK') {
					return 'Ceiling And Visibility OK: Visibility 10km or more, no significant weather, no cloud below 5000ft';
				} else if (text.endsWith('SM')) {
					const value = text.replace('SM', '');
					return `Visibility: ${value} statute miles`;
				} else {
					return `Visibility: ${parseInt(text, 10)} meters`;
				}
				
			case 'weather':
				if (text === 'NSW') {
					return 'No Significant Weather forecast during this period';
				}
				
				const weatherCodes = {
					'+': 'Heavy', '-': 'Light', 'VC': 'Vicinity',
					'MI': 'Shallow', 'PR': 'Partial', 'BC': 'Patches', 
					'DR': 'Low Drifting', 'BL': 'Blowing', 'SH': 'Shower',
					'TS': 'Thunderstorm', 'FZ': 'Freezing',
					'DZ': 'Drizzle', 'RA': 'Rain', 'SN': 'Snow', 'SG': 'Snow Grains',
					'IC': 'Ice Crystals', 'PL': 'Ice Pellets', 'GR': 'Hail',
					'GS': 'Small Hail', 'UP': 'Unknown Precipitation',
					'FG': 'Fog', 'BR': 'Mist', 'HZ': 'Haze', 'VA': 'Volcanic Ash',
					'DU': 'Dust', 'SA': 'Sand', 'PY': 'Spray', 'FU': 'Smoke',
					'SQ': 'Squall', 'PO': 'Dust/Sand Whirls', 'DS': 'Duststorm',
					'SS': 'Sandstorm', 'FC': 'Funnel Cloud/Tornado/Waterspout'
				};
				
				let desc = 'Weather: ';
				let remaining = text;
				
				// Check for intensity prefix
				if (remaining.startsWith('+') || remaining.startsWith('-') || remaining.startsWith('VC')) {
					const prefix = remaining.startsWith('VC') ? 'VC' : remaining[0];
					desc += weatherCodes[prefix] + ' ';
					remaining = remaining.substring(prefix.length);
				}
				
				// Try to match weather codes in 2-character pairs
				while (remaining.length > 0) {
					const code = remaining.substring(0, 2);
					if (weatherCodes[code]) {
						desc += weatherCodes[code] + ' ';
					} else {
						desc += code + ' ';
					}
					remaining = remaining.substring(2);
				}
				
				return desc.trim();
				
			case 'cloud':
				const cloudCodes = {
					'SKC': 'Sky Clear', 'CLR': 'Clear (no clouds below 12,000 ft)',
					'NSC': 'No Significant Clouds', 'NCD': 'No Clouds Detected',
					'FEW': 'Few (1-2 oktas)', 'SCT': 'Scattered (3-4 oktas)',
					'BKN': 'Broken (5-7 oktas)', 'OVC': 'Overcast (8 oktas)',
					'VV': 'Vertical Visibility (sky obscured)',
					'CB': 'Cumulonimbus', 'TCU': 'Towering Cumulus'
				};
				
				const coverMatch = text.match(/^(SKC|CLR|NSC|NCD|FEW|SCT|BKN|OVC|VV)/);
				if (!coverMatch) return `Cloud: ${text} (unrecognized format)`;
				
				const cover = coverMatch[1];
				const coverText = cloudCodes[cover] || cover;
				
				const baseMatch = text.match(/\d{3}/);
				const base = baseMatch ? parseInt(baseMatch[0], 10) * 100 : null;
				
				const typeMatch = text.match(/(CB|TCU)$/);
				const cloudType = typeMatch ? cloudCodes[typeMatch[1]] : null;
				
				if (base && cloudType) {
					return `Cloud: ${coverText} at ${base} feet (${cloudType})`;
				} else if (base) {
					return `Cloud: ${coverText} at ${base} feet`;
				} else {
					return `Cloud: ${coverText}`;
				}
				
			case 'changeIndicator':
				if (text === 'BECMG') {
					return 'Becoming: Gradual change to new conditions, which will then persist';
				} else if (text === 'TEMPO') {
					return 'Temporarily: Temporary fluctuations, each lasting less than one hour';
				} else if (text.startsWith('PROB')) {
					const prob = text.substring(4, 6);
					if (text.includes('TEMPO')) {
						return `Probability ${prob}%: Temporary conditions with ${prob}% chance of occurring`;
					} else {
						return `Probability ${prob}%: Conditions have ${prob}% chance of occurring`;
					}
				} else if (text.startsWith('FM')) {
					const day = text.substring(2, 4);
					const hour = text.substring(4, 6);
					const minute = text.substring(6, 8);
					return `From: Rapid and significant change at day ${day}, ${hour}:${minute} UTC`;
				}
				return `Change Indicator: ${text}`;
				
			case 'changeTime':
				const [changeFrom, changeTo] = text.split('/');
				const changeFromDay = changeFrom.substring(0, 2);
				const changeFromHour = changeFrom.substring(2, 4);
				const changeToDay = changeTo.substring(0, 2);
				const changeToHour = changeTo.substring(2, 4);
				return `Change Period: From day ${changeFromDay} at ${changeFromHour}:00 UTC to day ${changeToDay} at ${changeToHour}:00 UTC`;
				
			case 'temp':
				// Format is TXX/XX where X is temperature, M prefix for minus
				const tempParts = text.substring(1).split('/');
				const maxTemp = tempParts[0].startsWith('M') ? 
					-parseInt(tempParts[0].substring(1), 10) : 
					parseInt(tempParts[0], 10);
				const minTemp = tempParts[1].startsWith('M') ? 
					-parseInt(tempParts[1].substring(1), 10) : 
					parseInt(tempParts[1], 10);
				return `Temperature: Maximum ${maxTemp}°C, Minimum ${minTemp}°C`;
				
			case 'wind_shear':
				// Format is WSfff/dddss where fff=height, ddd=dir, ss=speed
				const height = parseInt(text.substring(2, 5), 10) * 100;
				const wsDir = parseInt(text.substring(6, 9), 10);
				const wsSpeed = parseInt(text.substring(9, 11), 10);
				return `Wind Shear: At ${height} feet, wind from ${wsDir}° at ${wsSpeed} knots`;
				
			case 'remarks_indicator':
				return 'Remarks: Additional coded information follows';
				
			case 'remarks':
				// Try to recognize common remarks
				if (text.startsWith('AO')) {
					return `Automated Station Type: ${text === 'AO1' ? 'without precipitation discriminator' : 'with precipitation discriminator'}`;
				} else {
					return `Remark: ${text}`;
				}
				
			default:
				return `${text}: Unrecognized TAF component`;
		}
	}
</script>

<div class="taf-hover-display">
	<h2 class="text-lg font-semibold text-gray-700 mb-4">Interactive TAF String</h2>
	<p class="text-sm text-gray-600 mb-3">
		Hover over each part of the TAF to see its meaning. This helps pilots understand each component of the raw forecast.
	</p>
	
	<div class="relative">
		<div class="bg-gray-100 p-4 rounded font-mono text-base border border-gray-300 flex flex-wrap gap-1.5">
			{#each parsedTafSections as section, i}
				<span 
					class="cursor-help pb-0.5 px-0.5 rounded hover:bg-blue-100 border-b-2 border-transparent hover:border-blue-500 transition-colors"
					on:mouseenter={() => hoveredSection = section}
					on:mouseleave={() => hoveredSection = null}
					data-type={section.type}
				>
					{section.text}
				</span>
			{/each}
		</div>
		
		{#if hoveredSection}
			<div class="mt-3 bg-blue-50 p-3 rounded-md border border-blue-200 transition-all">
				<div class="flex items-start">
					<div class="flex-shrink-0 mt-1">
						<svg class="h-5 w-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
						</svg>
					</div>
					<div class="ml-3">
						<h3 class="text-sm font-medium text-blue-800">
							{hoveredSection.text} <span class="text-blue-600">({hoveredSection.type})</span>
						</h3>
						<p class="mt-1 text-sm text-blue-700">
							{getDetailedDescription(hoveredSection)}
						</p>
					</div>
				</div>
			</div>
		{:else}
			<div class="mt-3 bg-gray-50 p-3 rounded-md border border-gray-200 text-gray-500 text-sm">
				Hover over any part of the TAF above to see its detailed meaning
			</div>
		{/if}
	</div>
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
	[data-type="wind_shear"] {
		color: #dc2626; /* red-600 */
		font-weight: bold;
	}
	[data-type="remarks_indicator"] {
		color: #64748b; /* slate-500 */
		font-weight: bold;
	}
	[data-type="remarks"] {
		color: #64748b; /* slate-500 */
	}
</style>