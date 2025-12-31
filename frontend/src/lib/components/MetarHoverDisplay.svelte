<script>
	// Props for the component
	export let metarString = '';

	// State management
	let hoveredSection = null;
	let parsedMetarSections = [];

	// Process the METAR string whenever it changes
	$: {
		if (metarString) {
			parsedMetarSections = parseMetarStringIntoSections(metarString);
		} else {
			parsedMetarSections = [];
		}
	}

	// Parse METAR into interactive sections
	function parseMetarStringIntoSections(metarString) {
		if (!metarString) return [];
		
		const sections = [];
		const parts = metarString.split(' ');
		
		// Definitions for METAR parts
		const metarSectionDefinitions = [
			{
				type: 'station',
				pattern: /^[A-Z]{4}$/,
				description: 'ICAO Airport Code'
			},
			{
				type: 'time',
				pattern: /^\d{6}Z$/,
				description: 'Observation time (DDHHMM in UTC) followed by Z'
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
				type: 'runway_vr',
				pattern: /^R\d{2}[RCL]?\/.+$/,
				description: 'Runway visual range'
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
				type: 'temp_dewpoint',
				pattern: /^M?\d{2}\/M?\d{2}$/,
				description: 'Temperature/Dewpoint in Celsius (M prefix indicates negative)'
			},
			{
				type: 'altimeter',
				pattern: /^[AQ]\d{4}$/,
				description: 'Altimeter setting (A: inches of mercury, Q: hectopascals)'
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
			
			// Match the part against our patterns
			let matched = false;
			
			for (const def of metarSectionDefinitions) {
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
				// If no match, use a generic type
				sections.push({
					text: part,
					type: 'unknown',
					description: 'Unrecognized METAR component'
				});
			}
		});
		
		return sections;
	}

	// Get more detailed description for a METAR section based on content and type
	function getDetailedDescription(section) {
		const { text, type } = section;
		
		switch (type) {
			case 'station':
				return {
					title: `Station Identifier: ${text}`,
					description: `ICAO airport code. This is a four-letter code designating the specific airport or weather station reporting this observation.`,
					operationalImpact: `This is your reference point for all weather conditions in this report.`,
					summary: `All weather data in this METAR applies to the ${text} airport and its immediate vicinity.`
				};
				
			case 'time':
				const day = text.substring(0, 2);
				const hour = text.substring(2, 4);
				const minute = text.substring(4, 6);
				return {
					title: `Observation Time: Day ${day}, ${hour}:${minute} UTC`,
					description: `This is when the weather observation was made, expressed in Coordinated Universal Time (UTC).`,
					operationalImpact: `Weather conditions can change rapidly, so consider how much time has passed since this observation.`,
					summary: `This METAR was issued on day ${day} at ${hour}:${minute} UTC. The more recent the report, the more reliable it is for flight planning.`
				};
				
			case 'wind':
				if (text === '00000KT') {
					return {
						title: 'Wind: Calm',
						description: 'No measurable wind (0 knots) at the reporting station.',
						operationalImpact: 'Ideal wind conditions. No crosswind component to consider for takeoff and landing.',
						summary: 'Wind is calm, which generally presents favorable conditions for all phases of flight.'
					};
				} else if (text.startsWith('VRB')) {
					const speed = parseInt(text.match(/VRB(\d{2})/)[1], 10);
					const gust = text.includes('G') ? parseInt(text.match(/G(\d{2})/)[1], 10) : null;
					
					return {
						title: `Wind: Variable direction at ${speed} knots${gust ? `, gusting to ${gust} knots` : ''}`,
						description: `Wind direction is inconsistent/variable with a speed of ${speed} knots${gust ? ` and gusts up to ${gust} knots` : ''}.`,
						operationalImpact: `Anticipate changing headwind/crosswind components during takeoff and landing. ${speed < 10 ? 'Light winds generally present minimal challenges.' : 'Moderate winds may require rudder inputs to maintain centerline alignment.'}${gust ? ' Gusts may necessitate additional control inputs.' : ''}`,
						summary: `Wind is varying in direction at ${speed} knots${gust ? ` with gusts to ${gust} knots` : ''}. Be prepared for wind shifts during critical phases of flight.`
					};
				} else {
					const dir = parseInt(text.substring(0, 3), 10);
					const speed = parseInt(text.substring(3, 5), 10);
					const gustMatch = text.match(/G(\d{2})/);
					const gust = gustMatch ? parseInt(gustMatch[1], 10) : null;
					const unit = text.endsWith('KT') ? 'knots' : 'meters per second';
					
					let windIntensity = "light";
					if (speed > 20) windIntensity = "strong";
					else if (speed > 10) windIntensity = "moderate";
					
					return {
						title: `Wind: From ${dir}° at ${speed} ${unit}${gust ? `, gusting to ${gust} ${unit}` : ''}`,
						description: `Wind is coming from a direction of ${dir}° true at ${speed} ${unit}${gust ? ` with gusts reaching ${gust} ${unit}` : ''}.`,
						operationalImpact: `This is a ${windIntensity} wind condition. Calculate crosswind component for runways in use. ${gust ? `Gusts introduce an additional ${gust - speed} ${unit} to consider for approach and landing.` : ''} Adjust approach speed accordingly.`,
						summary: `Wind from ${dir}° at ${speed} ${unit}${gust ? ` gusting to ${gust} ${unit}` : ''}. Calculate crosswind component and adjust approach speed as needed.`
					};
				}
				
			case 'variable_wind':
				const [from, to] = text.split('V').map(v => parseInt(v, 10));
				return {
					title: `Variable Wind Direction: ${from}° to ${to}°`,
					description: `Wind direction is varying between ${from}° and ${to}° true.`,
					operationalImpact: `Plan for varying crosswind conditions during takeoff and landing. This variability may require additional rudder inputs to maintain centerline alignment.`,
					summary: `Wind direction is fluctuating within a ${Math.abs(to - from)}° range, from ${from}° to ${to}°. Be prepared for wind shifts during critical phases of flight.`
				};
				
			case 'visibility':
				if (text === 'CAVOK') {
					return {
						title: 'Ceiling And Visibility OK (CAVOK)',
						description: 'Visibility is 10km (6 miles) or more, no significant weather, no cloud below 5000ft, and no cumulonimbus.',
						operationalImpact: 'Excellent visual flight conditions. No visibility restrictions for VFR operations.',
						summary: 'Ideal visibility and cloud conditions for all flight operations. No significant weather present.'
					};
				} else if (text.endsWith('SM')) {
					const value = text.replace('SM', '');
					const visNum = eval(value); // Safely evaluates fractions like "1 1/2"
					
					let visCategory = "excellent";
					let vfrStatus = "well above VFR minimums";
					if (visNum < 1) {
						visCategory = "poor";
						vfrStatus = "below VFR minimums";
					} else if (visNum < 3) {
						visCategory = "reduced";
						vfrStatus = "marginal for VFR";
					} else if (visNum < 5) {
						visCategory = "moderate";
						vfrStatus = "sufficient for VFR";
					}
					
					return {
						title: `Visibility: ${value} statute miles`,
						description: `Horizontal flight visibility is ${value} statute miles (${(visNum * 1.609).toFixed(1)} kilometers).`,
						operationalImpact: `This is ${visCategory} visibility, ${vfrStatus}. ${visNum < 3 ? 'Consider IFR operation or delay departure.' : 'VFR flight is possible but maintain vigilance.'}`,
						summary: `${value} SM visibility (${visCategory}). ${visNum < 3 ? 'Challenging for VFR operations.' : 'Acceptable for most flight operations.'}`
					};
				} else {
					const meters = parseInt(text, 10);
					const miles = (meters / 1609).toFixed(1);
					
					let visCategory = "excellent";
					let vfrStatus = "well above VFR minimums";
					if (meters < 1600) {
						visCategory = "poor";
						vfrStatus = "below VFR minimums";
					} else if (meters < 5000) {
						visCategory = "reduced";
						vfrStatus = "marginal for VFR";
					} else if (meters < 8001) {
						visCategory = "moderate";
						vfrStatus = "sufficient for VFR";
					}
					
					return {
						title: `Visibility: ${meters} meters`,
						description: `Horizontal flight visibility is ${meters} meters (${miles} statute miles).`,
						operationalImpact: `This is ${visCategory} visibility, ${vfrStatus}. ${meters < 5000 ? 'Consider IFR operation or delay departure.' : 'VFR flight is possible but maintain vigilance.'}`,
						summary: `${meters} meters visibility (${visCategory}). ${meters < 5000 ? 'Challenging for VFR operations.' : 'Acceptable for most flight operations.'}`
					};
				}
				
			case 'weather':
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
				
				let desc = '';
				let remaining = text;
				let intensity = '';
				let weatherType = '';
				let severity = '';
				
				// Check for intensity prefix
				if (remaining.startsWith('+') || remaining.startsWith('-') || remaining.startsWith('VC')) {
					const prefix = remaining.startsWith('VC') ? 'VC' : remaining[0];
					intensity = weatherCodes[prefix];
					remaining = remaining.substring(prefix.length);
				}
				
				// Extract weather phenomenon codes
				let codeList = [];
				// Try to match weather codes in 2-character pairs
				while (remaining.length > 0) {
					const code = remaining.substring(0, 2);
					if (weatherCodes[code]) {
						codeList.push(weatherCodes[code]);
					} else {
						codeList.push(code);
					}
					remaining = remaining.substring(2);
				}
				
				// Categorize the weather severity
				if (text.includes('TS') || text.includes('+') || text.includes('GR')) {
					severity = "severe";
				} else if (text.includes('SN') || text.includes('FG') || text.includes('FZ')) {
					severity = "moderate to significant";
				} else {
					severity = "mild";
				}
				
				// Identify primary weather type
				if (codeList.includes('Thunderstorm')) weatherType = "convective";
				else if (codeList.includes('Rain') || codeList.includes('Drizzle')) weatherType = "precipitation";
				else if (codeList.includes('Snow') || codeList.includes('Ice')) weatherType = "winter";
				else if (codeList.includes('Fog') || codeList.includes('Mist')) weatherType = "visibility restriction";
				else weatherType = "other";
				
				const weatherDesc = `${intensity ? intensity + ' ' : ''}${codeList.join(' ')}`;
				
				return {
					title: `Weather: ${weatherDesc}`,
					description: `Current weather phenomenon: ${weatherDesc}. This indicates ${severity} weather conditions.`,
					operationalImpact: `This ${weatherType} weather presents ${severity} operational challenges. ${text.includes('TS') ? 'Thunderstorms may require course deviations and create turbulence.' : ''} ${text.includes('FG') ? 'Fog will restrict visibility and may require instrument approaches.' : ''} ${text.includes('FZ') ? 'Freezing conditions create icing risk on aircraft surfaces.' : ''} ${text.includes('SN') ? 'Snow may affect braking action and require runway treatment.' : ''}`,
					summary: `${weatherDesc}. ${severity.charAt(0).toUpperCase() + severity.slice(1)} ${weatherType} weather that requires ${severity === "severe" ? "immediate attention and potential rerouting" : "awareness and monitoring"}.`
				};
				
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
				if (!coverMatch) return {
					title: `Cloud: ${text}`,
					description: `Unrecognized cloud format.`,
					operationalImpact: `Unable to determine cloud impact on operations.`,
					summary: `Consult raw METAR or weather briefing service for clarification on cloud conditions.`
				};
				
				const cover = coverMatch[1];
				const coverText = cloudCodes[cover] || cover;
				
				const baseMatch = text.match(/\d{3}/);
				const base = baseMatch ? parseInt(baseMatch[0], 10) * 100 : null;
				
				const typeMatch = text.match(/(CB|TCU)$/);
				const cloudType = typeMatch ? cloudCodes[typeMatch[1]] : null;
				
				let flightCategory = "";
				if (cover === 'BKN' || cover === 'OVC' || cover === 'VV') {
					if (base !== null) {
						if (base < 500) flightCategory = "LIFR (Low Instrument Flight Rules)";
						else if (base < 1000) flightCategory = "IFR (Instrument Flight Rules)";
						else if (base < 3000) flightCategory = "MVFR (Marginal Visual Flight Rules)";
						else flightCategory = "VFR (Visual Flight Rules)";
					}
				}
				
				const baseDesc = base ? `at ${base} feet AGL` : "";
				const typeDesc = cloudType ? ` (${cloudType})` : "";
				
				let operationalImpact = "";
				if (cloudType === 'Cumulonimbus') {
					operationalImpact = `${cloudType} clouds indicate severe thunderstorm activity. Avoid these clouds by at least 20 NM due to severe turbulence, lightning, and potential hail.`;
				} else if (cloudType === 'Towering Cumulus') {
					operationalImpact = `${cloudType} clouds indicate developing thunderstorms. Exercise caution and monitor for further development.`;
				} else if (cover === 'BKN' || cover === 'OVC') {
					if (base !== null && base < 1000) {
						operationalImpact = `Ceiling below 1,000 feet requires instrument approaches. ${flightCategory} conditions in effect.`;
					} else {
						operationalImpact = `Cloud layer may restrict VFR operations. ${flightCategory} conditions in effect.`;
					}
				} else if (cover === 'VV') {
					operationalImpact = `Sky obscured. Vertical visibility limited to ${base} feet. Instrument approaches required.`;
				} else {
					operationalImpact = `Minimal impact on flight operations. Ceiling adequate for VFR flight.`;
				}
				
				return {
					title: `Cloud: ${coverText} ${baseDesc}${typeDesc}`,
					description: `Cloud coverage is ${coverText} (${cover})${base ? ` with base at ${base} feet above ground level` : ''}${cloudType ? `. Cloud type is ${cloudType}` : ''}.`,
					operationalImpact: operationalImpact,
					summary: `${coverText} clouds ${baseDesc}${typeDesc}. ${flightCategory ? `This is considered ${flightCategory}.` : ''}`
				};
				
			case 'temp_dewpoint':
				const [tempStr, dewStr] = text.split('/');
				const temp = tempStr.startsWith('M') ? -parseInt(tempStr.substring(1), 10) : parseInt(tempStr, 10);
				const dew = dewStr.startsWith('M') ? -parseInt(dewStr.substring(1), 10) : parseInt(dewStr, 10);
				const spreadDiff = temp - dew;
				
				let fogRisk = "unlikely";
				if (spreadDiff <= 2) fogRisk = "very likely";
				else if (spreadDiff <= 5) fogRisk = "possible";
				
				let icingRisk = "low";
				if (temp <= 0 && temp >= -12) icingRisk = "moderate to high";
				else if (temp < -12) icingRisk = "reduced but still possible";
				
				let carb_icing = "unlikely";
				if (temp <= 15 && temp > 0 && spreadDiff <= 8) {
					carb_icing = temp <= 5 ? "serious risk" : "moderate risk";
				}
				
				return {
					title: `Temperature: ${temp}°C, Dewpoint: ${dew}°C (Spread: ${spreadDiff}°C)`,
					description: `Outside air temperature is ${temp}°C with dewpoint at ${dew}°C, giving a spread of ${spreadDiff}°C.`,
					operationalImpact: `Fog formation is ${fogRisk} with this ${spreadDiff}°C spread. Structural icing risk is ${icingRisk} at this temperature. Carburetor icing is a ${carb_icing} in these conditions.`,
					summary: `${temp}°C with ${spreadDiff}°C spread from dewpoint. ${spreadDiff <= 3 ? "Watch for fog development." : ""} ${temp <= 0 ? "Be alert for airframe icing conditions." : ""} ${temp <= 15 && temp > 0 ? "Consider use of carburetor heat." : ""}`
				};
				
			case 'altimeter':
				let setting, unit;
				if (text.startsWith('A')) {
					setting = parseInt(text.substring(1), 10) / 100;
					unit = "inHg";
					
					const stdSetting = 29.92;
					const diff = setting - stdSetting;
					const approxAltError = Math.round(diff * 1000);
					
					return {
						title: `Altimeter: ${setting} inHg`,
						description: `Altimeter setting is ${setting} inches of mercury. This is ${diff > 0 ? 'higher' : 'lower'} than standard pressure (29.92 inHg).`,
						operationalImpact: `With this setting, uncorrected altimeters would read approximately ${Math.abs(approxAltError)} feet ${diff > 0 ? 'higher' : 'lower'} than true altitude ("${diff > 0 ? 'High to low, look out below' : 'Low to high, clear the sky'}").`,
						summary: `Set altimeter to ${setting} inHg. This is ${Math.abs(diff).toFixed(2)} inHg ${diff > 0 ? 'above' : 'below'} standard pressure.`
					};
				} else { // Q prefix
					setting = parseInt(text.substring(1), 10);
					unit = "hPa";
					
					const stdSetting = 1013.25;
					const diff = setting - stdSetting;
					const approxAltError = Math.round(diff * 27);
					
					return {
						title: `Altimeter: ${setting} hPa (hectopascals)`,
						description: `Altimeter setting is ${setting} hectopascals (millibars). This is ${diff > 0 ? 'higher' : 'lower'} than standard pressure (1013.25 hPa).`,
						operationalImpact: `With this setting, uncorrected altimeters would read approximately ${Math.abs(approxAltError)} feet ${diff > 0 ? 'higher' : 'lower'} than true altitude ("${diff > 0 ? 'High to low, look out below' : 'Low to high, clear the sky'}").`,
						summary: `Set altimeter to ${setting} hPa. This is ${Math.abs(diff)} hPa ${diff > 0 ? 'above' : 'below'} standard pressure.`
					};
				}
				
			case 'remarks_indicator':
				return {
					title: 'Remarks Section',
					description: 'The RMK indicator shows that additional coded information follows in the remarks section.',
					operationalImpact: 'Remarks may contain important weather trend information, runway conditions, or other operationally significant details.',
					summary: 'Check the remarks section for supplemental information that could affect your flight.'
				};
				
			case 'remarks':
				// Try to recognize common remarks
				if (text.startsWith('AO')) {
					const aoType = text === 'AO1' ? 'without precipitation discriminator' : 'with precipitation discriminator';
					return {
						title: `Automated Station Type: ${text}`,
						description: `This is an automated weather reporting station ${aoType}.`,
						operationalImpact: `${text === 'AO1' ? 'This station cannot distinguish between different types of precipitation.' : 'This station can distinguish between different types of precipitation.'}`,
						summary: `Automated observation ${aoType}.`
					};
				} else if (text.startsWith('T')) {
					return {
						title: 'Exact Temperature/Dewpoint Values',
						description: 'Provides temperature and dewpoint to a tenth of a degree Celsius.',
						operationalImpact: 'More precise temperature/dewpoint information for detailed calculations.',
						summary: 'Exact temperature and dewpoint measurements for precision calculations.'
					};
				} else if (text.startsWith('SLP')) {
					const pressure = parseInt(text.substring(3), 10);
					const slp = pressure < 500 ? `10${pressure}` : `9${pressure}`;
					const formatted = `${slp.substring(0, 4)}.${slp.substring(4)}`;
					return {
						title: `Sea Level Pressure: ${formatted} hPa`,
						description: `The sea level pressure is ${formatted} hectopascals (millibars).`,
						operationalImpact: 'Useful for tracking pressure trends and weather system movement.',
						summary: `Sea level pressure is ${formatted} hPa. ${pressure < 980 ? 'This is relatively low pressure which may indicate poor weather.' : pressure > 1020 ? 'This is relatively high pressure which typically indicates fair weather.' : 'This is near average pressure.'}`
					};
				} else if (text.startsWith('P')) {
					const precip = parseFloat(text.substring(1)) / 100;
					return {
						title: `Precipitation Amount: ${precip} inches`,
						description: `${precip} inches of precipitation has fallen in the last hour.`,
						operationalImpact: `${precip > 0.5 ? 'Heavy' : precip > 0.1 ? 'Moderate' : 'Light'} precipitation may impact visibility and runway conditions.`,
						summary: `${precip} inches of precipitation in the last hour. ${precip > 0.3 ? 'Watch for standing water on runways.' : ''}`
					};
				} else if (text.startsWith('RMK')) {
					return {
						title: 'Remarks Indicator',
						description: 'Indicates the start of the remarks section.',
						operationalImpact: 'Marks where additional important information begins.',
						summary: 'Additional weather information follows in the remarks section.'
					};
				} else {
					return {
						title: `Remark: ${text}`,
						description: `Additional weather information: ${text}`,
						operationalImpact: 'Refer to standard METAR decode guides for specific operational impacts.',
						summary: `Additional data: ${text}. Consult weather briefing services if clarification is needed.`
					};
				}
				
			default:
				return {
					title: `${text}: Unrecognized Component`,
					description: `This is not a standard METAR component or could not be automatically decoded.`,
					operationalImpact: 'Unknown impact. Consider consulting a weather briefing service for clarification.',
					summary: `Unrecognized code: ${text}. Please refer to standard METAR decode guides.`
				};
		}
	}
</script>

<div class="metar-hover-display">
	<h2 class="text-lg font-semibold text-gray-700 mb-4">Interactive METAR String</h2>
	<p class="text-sm text-gray-600 mb-3">
		Hover over each part of the METAR to see its detailed meaning and interpretation. This helps pilots understand each component of the raw METAR.
	</p>
	
	<div class="relative">
		<div class="bg-gray-100 p-4 rounded font-mono text-base border border-gray-300 flex flex-wrap gap-1.5">
			{#each parsedMetarSections as section, i}
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
				{#if typeof getDetailedDescription(hoveredSection) === 'object'}
					{@const details = getDetailedDescription(hoveredSection)}
					<div class="space-y-3">
						<div class="flex items-start">
							<div class="flex-shrink-0 mt-1">
								<svg class="h-5 w-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
								</svg>
							</div>
							<div class="ml-3">
								<h3 class="text-sm font-medium text-blue-800">
									{details.title} <span class="text-blue-600">({hoveredSection.type})</span>
								</h3>
								<p class="mt-1 text-sm text-blue-700">
									{details.description}
								</p>
							</div>
						</div>
						
						<div class="pt-2 border-t border-blue-200">
							<h4 class="text-sm font-medium text-blue-800">Operational Impact:</h4>
							<p class="mt-1 text-sm text-blue-700">{details.operationalImpact}</p>
						</div>
						
						<div class="pt-2 border-t border-blue-200">
							<h4 class="text-sm font-medium text-blue-800">Summary:</h4>
							<p class="mt-1 text-sm text-blue-700">{details.summary}</p>
						</div>
					</div>
				{:else}
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
				{/if}
			</div>
		{:else}
			<div class="mt-3 bg-gray-50 p-3 rounded-md border border-gray-200 text-gray-500 text-sm">
				Hover over any part of the METAR above to see its detailed interpretation and operational significance
			</div>
		{/if}
	</div>
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