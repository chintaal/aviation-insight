<script>
	// Props for the component
	export let pirepString = '';
	export let locationCode = '';

	// State management
	let hoveredSection = null;
	let parsedPirepSections = [];

	// Process the PIREP string whenever it changes
	$: {
		if (pirepString) {
			parsedPirepSections = parsePirepStringIntoSections(pirepString, locationCode);
		} else {
			parsedPirepSections = [];
		}
	}

	// Parse PIREP into interactive sections
	function parsePirepStringIntoSections(pirepString, locationCode) {
		if (!pirepString) return [];
		
		const sections = [];
		// First add the location code as the first section
		if (locationCode) {
			sections.push({
				text: locationCode,
				type: 'location',
				description: 'NAVAID or Airport identification code'
			});
		}
		
		// Split the report into its components (by forward slashes with spaces around them)
		const parts = pirepString.split(/\s*\/\s*/).filter(part => part.trim().length > 0);
		
		// Parse the report type (UA or UUA)
		if (parts.length > 0 && (parts[0] === 'UA' || parts[0] === 'UUA')) {
			sections.push({
				text: parts[0],
				type: 'report_type',
				description: parts[0] === 'UA' ? 'Routine pilot report' : 'Urgent pilot report'
			});
			parts.shift(); // Remove the first element
		}
		
		// Parse the remaining elements
		parts.forEach(part => {
			if (part.startsWith('OV ')) {
				sections.push({
					text: '/' + part,
					type: 'location',
					description: 'Location: Position relative to a NAVAID'
				});
			} 
			else if (part.startsWith('TM ')) {
				sections.push({
					text: '/' + part,
					type: 'time',
					description: 'Time: Observation time in UTC (HHMM)'
				});
			}
			else if (part.startsWith('FL')) {
				const flValue = part.substring(2).trim();
				let description = 'Flight Level';
				
				if (flValue === 'DURGD' || flValue === 'DURG' || flValue === 'DURD' || flValue === 'DURC') {
					description = 'Flight Level: During descent/climb';
				} else if (!isNaN(flValue)) {
					const altitude = parseInt(flValue) * 100;
					description = `Flight Level: ${altitude} feet`;
				}
				
				sections.push({
					text: '/' + part,
					type: 'altitude',
					description
				});
			}
			else if (part.startsWith('TP ')) {
				sections.push({
					text: '/' + part,
					type: 'aircraft',
					description: 'Aircraft Type: Type of aircraft making the report'
				});
			}
			else if (part.startsWith('SK ')) {
				sections.push({
					text: '/' + part,
					type: 'sky',
					description: 'Sky conditions: Cloud coverage and bases'
				});
			}
			else if (part.startsWith('TA ')) {
				sections.push({
					text: '/' + part,
					type: 'temperature',
					description: 'Temperature: Air temperature in Celsius'
				});
			}
			else if (part.startsWith('TB ')) {
				let tbDescription = 'Turbulence: ';
				const tbContent = part.substring(3).trim().toUpperCase();
				
				if (tbContent === 'NEG') {
					tbDescription += 'No turbulence reported';
				} else if (tbContent.includes('LGT') && tbContent.includes('MOD')) {
					tbDescription += 'Light to moderate turbulence';
				} else if (tbContent.includes('MOD') && tbContent.includes('SEV')) {
					tbDescription += 'Moderate to severe turbulence';
				} else if (tbContent.includes('SEV')) {
					tbDescription += 'Severe turbulence';
				} else if (tbContent.includes('MOD')) {
					tbDescription += 'Moderate turbulence';
				} else if (tbContent.includes('LGT')) {
					tbDescription += 'Light turbulence';
				} else if (tbContent.includes('SMOOTH')) {
					tbDescription += 'Smooth conditions (no turbulence)';
				}
				
				if (tbContent.includes('CONS')) {
					tbDescription += ', continuous';
				} else if (tbContent.includes('OCNL')) {
					tbDescription += ', occasional';
				} else if (tbContent.includes('INTMT')) {
					tbDescription += ', intermittent';
				}
				
				// Extract any altitude information
				const altMatch = tbContent.match(/\d{3}/);
				if (altMatch) {
					const altitude = parseInt(altMatch[0]) * 100;
					tbDescription += ` at ${altitude} feet`;
				}
				
				sections.push({
					text: '/' + part,
					type: 'turbulence',
					description: tbDescription
				});
			}
			else if (part.startsWith('IC ')) {
				let icDescription = 'Icing: ';
				const icContent = part.substring(3).trim().toUpperCase();
				
				if (icContent === 'NEG') {
					icDescription += 'No icing reported';
				} else if (icContent.includes('LGT') && icContent.includes('MOD')) {
					icDescription += 'Light to moderate icing';
				} else if (icContent.includes('MOD') && icContent.includes('SEV')) {
					icDescription += 'Moderate to severe icing';
				} else if (icContent.includes('SEV')) {
					icDescription += 'Severe icing';
				} else if (icContent.includes('MOD')) {
					icDescription += 'Moderate icing';
				} else if (icContent.includes('LGT')) {
					icDescription += 'Light icing';
				} else if (icContent.includes('TRC')) {
					icDescription += 'Trace icing';
				}
				
				if (icContent.includes('RIME')) {
					icDescription += ', rime ice';
				} else if (icContent.includes('CLEAR')) {
					icDescription += ', clear ice';
				} else if (icContent.includes('MIXED')) {
					icDescription += ', mixed ice';
				}
				
				// Extract any altitude information
				const altMatch = icContent.match(/\d{3}/);
				if (altMatch) {
					const altitude = parseInt(altMatch[0]) * 100;
					icDescription += ` at ${altitude} feet`;
				}
				
				sections.push({
					text: '/' + part,
					type: 'icing',
					description: icDescription
				});
			}
			else if (part.startsWith('WX ')) {
				sections.push({
					text: '/' + part,
					type: 'weather',
					description: 'Weather: Observed weather conditions'
				});
			}
			else if (part.startsWith('RM ')) {
				sections.push({
					text: '/' + part,
					type: 'remarks',
					description: 'Remarks: Additional information'
				});
			}
			else {
				// If we can't categorize it, add it as unknown
				sections.push({
					text: '/' + part,
					type: 'unknown',
					description: 'Additional PIREP information'
				});
			}
		});
		
		return sections;
	}

	// Get more detailed description for a section based on its content
	function getDetailedDescription(section) {
		// Basic description from the parsed section
		return section.description;
	}
</script>

<div class="pirep-hover-display">
	<h2 class="text-lg font-semibold text-gray-700 mb-4">Interactive PIREP String</h2>
	<p class="text-sm text-gray-600 mb-3">
		Hover over each part of the PIREP to see its meaning. This helps pilots understand each component of the raw pilot report.
	</p>
	
	<div class="relative">
		<div class="bg-gray-100 p-4 rounded font-mono text-base border border-gray-300 flex flex-wrap gap-1.5">
			{#each parsedPirepSections as section, i}
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
				Hover over any part of the PIREP above to see its detailed meaning
			</div>
		{/if}
	</div>
</div>

<style>
	/* Add color styling for the different PIREP components */
	[data-type="location"] {
		color: #3b82f6; /* blue-500 */
	}
	[data-type="report_type"] {
		color: #8b5cf6; /* purple-500 */
	}
	[data-type="time"] {
		color: #8b5cf6; /* purple-500 */
	}
	[data-type="altitude"] {
		color: #f59e0b; /* amber-500 */
	}
	[data-type="aircraft"] {
		color: #059669; /* emerald-600 */
	}
	[data-type="turbulence"] {
		color: #ef4444; /* red-500 */
	}
	[data-type="icing"] {
		color: #3b82f6; /* blue-500 */
	}
	[data-type="weather"], [data-type="sky"] {
		color: #10b981; /* emerald-500 */
	}
	[data-type="temperature"] {
		color: #f59e0b; /* amber-500 */
	}
	[data-type="remarks"] {
		color: #64748b; /* slate-500 */
	}
	[data-type="unknown"] {
		color: #6b7280; /* gray-500 */
	}
</style>