<script>
	import { onMount } from 'svelte';
	import PirepHoverDisplay from '$lib/components/PirepHoverDisplay.svelte';

	// State management
	let station = 'KATL';
	let distance = 200;
	let age = 1.5;
	let loading = false;
	let error = null;
	let result = null;
	let selectedPirep = null;
	let sidebarOpen = true;
	let mapVisible = true;
	let map;
	let markers = [];
	
	// Get color class based on turbulence intensity
	function getTurbulenceClass(intensity, type = 'bg') {
		if (!intensity) return `${type}-gray-100 text-gray-800`;
		
		const level = intensity.toLowerCase();
		if (level.includes('sev') || level === 'severe') {
			return `${type}-red-100 text-red-800`;
		} else if (level.includes('mod') || level === 'moderate') {
			return `${type}-orange-100 text-orange-800`;
		} else if (level.includes('lgt') || level === 'light') {
			return `${type}-yellow-100 text-yellow-800`;
		} else {
			return `${type}-gray-100 text-gray-800`;
		}
	}

	// Get color class based on icing intensity
	function getIcingClass(intensity, type = 'bg') {
		if (!intensity) return `${type}-gray-100 text-gray-800`;
		
		const level = intensity.toLowerCase();
		if (level.includes('sev') || level === 'severe') {
			return `${type}-purple-100 text-purple-800`;
		} else if (level.includes('mod') || level === 'moderate') {
			return `${type}-blue-100 text-blue-800`;
		} else if (level.includes('lgt') || level === 'light' || level.includes('trc') || level === 'trace') {
			return `${type}-green-100 text-green-800`;
		} else {
			return `${type}-gray-100 text-gray-800`;
		}
	}

	// Format date for display
	function formatDate(timestamp) {
		if (!timestamp) return 'N/A';
		const date = new Date(timestamp);
		return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + 
			' on ' + date.toLocaleDateString([], { month: 'short', day: 'numeric' });
	}
	
	// Get altitude text formatted for display
	function getAltitudeText(altitude) {
		if (!altitude) return 'Unknown';
		if (typeof altitude === 'string') return altitude;
		return `${altitude.toLocaleString()} ft`;
	}
	
	// Calculate time ago for display
	function timeAgo(dateString) {
		if (!dateString) return '';
		
		const date = new Date(dateString);
		const now = new Date();
		const diffMs = now - date;
		const diffMins = Math.floor(diffMs / 60000);
		
		if (diffMins < 60) {
			return `${diffMins} min${diffMins !== 1 ? 's' : ''} ago`;
		} else {
			const diffHours = Math.floor(diffMins / 60);
			return `${diffHours} hr${diffHours !== 1 ? 's' : ''} ago`;
		}
	}
	
	// Format coordinates for map URL
	function getMapUrl(pirep) {
		if (!pirep || !pirep.location) return null;
		
		// This is a simple approach - for an actual implementation,
		// you would use lat/lon coordinates if available
		return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(pirep.location)}`;
	}
	
	// Get icon based on report type
	function getReportTypeIcon(reportType) {
		if (!reportType) return 'clipboard';
		
		if (reportType === 'UUA') {
			return 'alert-triangle'; // Urgent PIREP
		}
		return 'clipboard-check'; // Regular PIREP (UA)
	}

	// Generate a summary of the PIREP for quick reference
	function generatePirepSummary(pirep) {
		if (!pirep) return "No PIREP data available";
		
		const parts = [];
		
		// Add report type
		const reportTypeText = pirep.report_type === 'UUA' ? 'Urgent PIREP' : 'PIREP';
		parts.push(reportTypeText);
		
		// Add location
		if (pirep.location) {
			parts.push(`near ${pirep.location}`);
		}
		
		// Add altitude
		if (pirep.altitude) {
			parts.push(`at ${getAltitudeText(pirep.altitude)}`);
		}
		
		// Add aircraft type
		if (pirep.aircraft_type) {
			parts.push(`from a ${pirep.aircraft_type}`);
		}
		
		// Add weather hazards
		const hazards = [];
		if (pirep.turbulence && pirep.turbulence.intensity) {
			hazards.push(`${pirep.turbulence.intensity} turbulence`);
		}
		if (pirep.icing && pirep.icing.intensity) {
			hazards.push(`${pirep.icing.intensity} ${pirep.icing.type || ''} icing`.trim());
		}
		
		if (hazards.length > 0) {
			parts.push(`reporting ${hazards.join(' and ')}`);
		}
		
		return parts.join(' ');
	}
	
	// Get a simple direction description from location code
	function getDirectionFromLocation(locationCode) {
		if (!locationCode || typeof locationCode !== 'string') return '';
		
		// Extract directional info if in format like "10N BWI" (10nm North of BWI)
		const directionMatch = locationCode.match(/(\d+)([NESW]{1,2})\s+([A-Z]{3,4})/);
		if (directionMatch) {
			const [, distance, direction, airport] = directionMatch;
			return `${distance}nm ${getFullDirectionName(direction)} of ${airport}`;
		}
		
		return locationCode;
	}
	
	// Convert abbreviation to full direction name
	function getFullDirectionName(dir) {
		const directions = {
			'N': 'North',
			'NE': 'Northeast',
			'E': 'East',
			'SE': 'Southeast',
			'S': 'South', 
			'SW': 'Southwest',
			'W': 'West',
			'NW': 'Northwest'
		};
		
		return directions[dir] || dir;
	}

	// Extract coordinates from PIREP location
	function extractCoordinates(location) {
		// Default to Atlanta if we can't parse coordinates
		let defaultCoords = { lat: 33.6367, lng: -84.4281 }; // Atlanta coordinates
		
		if (!location) return defaultCoords;
		
		// Check if location has coordinates in it (e.g. "3634N/08425W")
		const coordPattern = /(\d{2,4})([NS])\/(\d{4,5})([EW])/;
		const match = location.match(coordPattern);
		
		if (match) {
			const [_, latNum, latDir, lngNum, lngDir] = match;
			
			// Parse latitude
			let lat = parseInt(latNum.substring(0, 2)) + (latNum.length > 2 ? parseFloat(latNum.substring(2)) / 60 : 0);
			if (latDir === 'S') lat = -lat;
			
			// Parse longitude
			let lng = parseInt(lngNum.substring(0, 3)) + (lngNum.length > 3 ? parseFloat(lngNum.substring(3)) / 60 : 0);
			if (lngDir === 'W') lng = -lng;
			
			return { lat, lng };
		}
		
		// If the location is an airport code, use Google Maps Geocoding API
		if (/^[A-Z]{3,4}$/.test(location)) {
			// First check our existing database
			const airportCode = location.toUpperCase();
			if (airportCode in airportCoordinates) {
				return airportCoordinates[airportCode];
			}
			
			// If not found in our database and we have a valid Geocoder, look it up
			if (window.google && window.google.maps && geocoder) {
				// We can't directly return here since geocoding is asynchronous
				// Instead, we'll request geocoding and update the map later when it's ready
				geocodeAirport(location);
				
				// Return Atlanta coordinates initially - the map will update when geocoding completes
				return defaultCoords;
			}
		}
		
		// Parse directional format like "15 N CAE" (15nm North of Columbia)
		const dirPattern1 = /(\d+)\s+([NESW]{1,2})\s+([A-Z]{3,4})/i;
		let dirMatch = location.match(dirPattern1);
		if (dirMatch) {
			const [, distance, direction, airport] = dirMatch;
			const airportCode = airport.toUpperCase();
			
			if (airportCode in airportCoordinates) {
				const baseCoords = airportCoordinates[airportCode];
				return calculateOffsetCoordinates(baseCoords, direction, parseFloat(distance));
			}
		}
		
		// Handle format like "ATL180037" (Atlanta 180° at 37nm)
		const dirPattern2 = /([A-Z]{3})(\d{3})(\d{3})/i;
		dirMatch = location.match(dirPattern2);
		if (dirMatch) {
			const [, navaid, bearing, distance] = dirMatch;
			const navaidCode = navaid.toUpperCase();
			
			if (navaidCode in airportCoordinates) {
				const baseCoords = airportCoordinates[navaidCode];
				const distanceNm = parseInt(distance) / 100; // Convert 037 to 37nm
				return calculateRadialDistance(baseCoords, parseInt(bearing), distanceNm);
			}
		}
		
		// If the location is directly an airport code
		const airportCode = location.toUpperCase();
		if (airportCode in airportCoordinates) {
			return airportCoordinates[airportCode];
		}
		
		// If we couldn't parse it, return the default coordinates without random offset
		// This prevents points from being scattered randomly
		return defaultCoords;
	}
	
	// Cache for geocoded airport coordinates
	let geocodedAirports = {};
	let geocoder;
	
	// Geocode airport code using Google Maps Geocoding API
	function geocodeAirport(airportCode) {
		if (!airportCode) return;
		
		// If already geocoded, don't repeat
		if (geocodedAirports[airportCode]) {
			return;
		}
		
		// Create geocoder if it doesn't exist
		if (!geocoder && window.google && window.google.maps) {
			geocoder = new google.maps.Geocoder();
		}
		
		if (geocoder) {
			// Add 'Airport' to improve accuracy of geocoding
			const address = `${airportCode} Airport`;
			
			geocoder.geocode({ 'address': address }, (results, status) => {
				if (status === google.maps.GeocoderStatus.OK && results[0]) {
					const position = results[0].geometry.location;
					const coords = {
						lat: position.lat(),
						lng: position.lng()
					};
					
					// Save to cache
					geocodedAirports[airportCode] = coords;
					airportCoordinates[airportCode] = coords;
					
					console.log(`Geocoded ${airportCode} to:`, coords);
					
					// If this is the current station, update the map
					if (airportCode === station && map) {
						// Re-center the map
						map.setCenter(coords);
						
						// Update station marker if it exists
						const stationMarker = markers.find(m => m.getTitle && m.getTitle() === station);
						if (stationMarker) {
							stationMarker.setPosition(coords);
						}
						
						// Update search circle if it exists
						markers.forEach(marker => {
							if (marker instanceof google.maps.Circle) {
								marker.setCenter(coords);
							}
						});
					}
				} else {
					console.warn(`Failed to geocode ${airportCode}: ${status}`);
				}
			});
		}
	}
	
	// Comprehensive airport coordinates database
	const airportCoordinates = {
		// Major airports in the region
		'ATL': { lat: 33.6367, lng: -84.4281 }, // Atlanta Hartsfield-Jackson
		'BHM': { lat: 33.5629, lng: -86.7535 }, // Birmingham
		'CAE': { lat: 33.9388, lng: -81.1195 }, // Columbia Metropolitan
		'CHA': { lat: 35.0353, lng: -85.2036 }, // Chattanooga
		'GAD': { lat: 33.9726, lng: -86.0889 }, // Gadsden
		'GRD': { lat: 34.2487, lng: -82.1595 }, // Greenwood County
		'GZH': { lat: 31.4170, lng: -87.0437 }, // Evergreen
		'HSV': { lat: 34.6372, lng: -86.7752 }, // Huntsville
		'MGM': { lat: 32.3006, lng: -86.3939 }, // Montgomery
		'PDK': { lat: 33.8756, lng: -84.3020 }, // DeKalb-Peachtree Atlanta
		'SAV': { lat: 32.1276, lng: -81.2020 }, // Savannah
		'SPA': { lat: 34.9156, lng: -81.9566 }, // Spartanburg
		'SSI': { lat: 31.1513, lng: -81.3913 }, // St Simons Island
		'TYS': { lat: 35.8108, lng: -83.9940 }, // Knoxville
		'WRB': { lat: 32.6400, lng: -83.5919 }, // Warner Robins
		'ABY': { lat: 31.5356, lng: -84.1944 }, // Albany
		'AMG': { lat: 31.5347, lng: -82.5077 }, // Alma
		'CSV': { lat: 35.9513, lng: -85.0852 }, // Crossville
		'MDQ': { lat: 32.8339, lng: -83.5622 }, // Macon Downtown
		'TCL': { lat: 33.2206, lng: -87.6114 }, // Tuscaloosa
		// Additional navigation points
		'LDK': { lat: 34.5675, lng: -83.5334 }, // Lakemont VOR
		'HCH': { lat: 36.4047, lng: -85.4394 }, // Hinch Mountain VOR
		'VNA': { lat: 32.2717, lng: -83.9380 }, // Vienna VOR
		'VXV': { lat: 35.8825, lng: -83.9450 }, // Knoxville VOR
		'MCN': { lat: 32.6918, lng: -83.6491 }, // Macon VOR
		'RQZ': { lat: 33.3069, lng: -82.3215 }, // Colliers VOR
		'VUZ': { lat: 33.6717, lng: -86.8997 }, // Vulcan VOR
		'GQO': { lat: 35.2147, lng: -85.0719 }, // Chattanooga VOR
		'MVC': { lat: 30.9377, lng: -86.5847 }  // Crestview VOR
	};
	
	// Calculate coordinates based on direction and distance from a point
	function calculateOffsetCoordinates(baseCoords, direction, distanceNm) {
		// Convert nautical miles to kilometers (1 nm = 1.852 km)
		const distanceKm = distanceNm * 1.852;
		
		// Earth's radius in km
		const R = 6371;
		
		// Convert distance to radians
		const d = distanceKm / R;
		
		// Convert base coordinates from degrees to radians
		const lat1 = baseCoords.lat * Math.PI / 180;
		const lon1 = baseCoords.lng * Math.PI / 180;
		
		// Calculate bearing in radians based on direction
		let bearing;
		switch(direction.toUpperCase()) {
			case 'N': bearing = 0; break;
			case 'NE': bearing = Math.PI / 4; break;
			case 'E': bearing = Math.PI / 2; break;
			case 'SE': bearing = 3 * Math.PI / 4; break;
			case 'S': bearing = Math.PI; break;
			case 'SW': bearing = 5 * Math.PI / 4; break;
			case 'W': bearing = 3 * Math.PI / 2; break;
			case 'NW': bearing = 7 * Math.PI / 4; break;
			default: bearing = 0;
		}
		
		// Calculate new position
		const lat2 = Math.asin(Math.sin(lat1) * Math.cos(d) + Math.cos(lat1) * Math.sin(d) * Math.cos(bearing));
		const lon2 = lon1 + Math.atan2(
			Math.sin(bearing) * Math.sin(d) * Math.cos(lat1),
			Math.cos(d) - Math.sin(lat1) * Math.sin(lat2)
		);
		
		// Convert back to degrees
		return {
			lat: lat2 * 180 / Math.PI,
			lng: lon2 * 180 / Math.PI
		};
	}
	
	// Calculate coordinates based on radial (bearing) and distance
	function calculateRadialDistance(baseCoords, bearing, distanceNm) {
		// Convert nautical miles to kilometers (1 nm = 1.852 km)
		const distanceKm = distanceNm * 1.852;
		
		// Earth's radius in km
		const R = 6371;
		
		// Convert distance to radians
		const d = distanceKm / R;
		
		// Convert base coordinates from degrees to radians
		const lat1 = baseCoords.lat * Math.PI / 180;
		const lon1 = baseCoords.lng * Math.PI / 180;
		
		// Convert bearing to radians
		const bearingRad = bearing * Math.PI / 180;
		
		// Calculate new position
		const lat2 = Math.asin(Math.sin(lat1) * Math.cos(d) + Math.cos(lat1) * Math.sin(d) * Math.cos(bearingRad));
		const lon2 = lon1 + Math.atan2(
			Math.sin(bearingRad) * Math.sin(d) * Math.cos(lat1),
			Math.cos(d) - Math.sin(lat1) * Math.sin(lat2)
		);
		
		// Convert back to degrees
		return {
			lat: lat2 * 180 / Math.PI,
			lng: lon2 * 180 / Math.PI
		};
	}
	
	// Determine marker icon and color based on PIREP content
	function getPirepMarkerOptions(pirep) {
		let color = '#3b82f6'; // Default blue
		let size = 14; // Default size
		let icon = 'PIREP';
		let strokeWeight = 2;
		
		// Make urgent PIREPs larger
		if (pirep.report_type === 'UUA') {
			color = '#ef4444'; // Red
			size = 18;
			icon = 'UUA';
			strokeWeight = 3;
		} 
		
		// Turbulence colors (priority over default)
		if (pirep.turbulence && pirep.turbulence.intensity) {
			const intensity = pirep.turbulence.intensity.toLowerCase();
			if (intensity.includes('sev')) {
				color = '#b91c1c'; // Red-700
				size = 18;
				icon = 'T-SEV';
			} else if (intensity.includes('mod')) {
				color = '#ea580c'; // Orange-600
				size = 16;
				icon = 'T-MOD';
			} else if (intensity.includes('lgt')) {
				color = '#ca8a04'; // Yellow-600
				size = 15;
				icon = 'T-LGT';
			}
		} 
		
		// Icing colors (only apply if no turbulence)
		if ((!pirep.turbulence || !pirep.turbulence.intensity) && pirep.icing && pirep.icing.intensity) {
			const intensity = pirep.icing.intensity.toLowerCase();
			if (intensity.includes('sev')) {
				color = '#7e22ce'; // Purple-700
				size = 17;
				icon = 'I-SEV';
			} else if (intensity.includes('mod')) {
				color = '#2563eb'; // Blue-600
				size = 15;
				icon = 'I-MOD';
			} else if (intensity.includes('lgt') || intensity.includes('trc')) {
				color = '#16a34a'; // Green-600
				size = 14;
				icon = 'I-LGT';
			}
		}
		
		return { color, size, icon, strokeWeight };
	}
	
	// Initialize the map
	function initMap() {
		if (!mapVisible) return;
		
		// Load Google Maps API if not already loaded
		if (!window.google || !window.google.maps) {
			const script = document.createElement('script');
			script.src = `https://maps.googleapis.com/maps/api/js?key=  API_KEY&libraries=visualization&callback=initPirepMap`;
			script.defer = true;
			script.async = true;
			
			// Define callback function in global scope
			window.initPirepMap = () => {
				createMap();
				addPirepsToMap();
			};
			
			document.head.appendChild(script);
		} else {
			createMap();
			addPirepsToMap();
		}
	}
	
	// Create the map instance
	function createMap() {
		// Find coordinates for center (use station or first PIREP)
		let center = { lat: 33.6367, lng: -84.4281 }; // Default to Atlanta
		
		if (Array.isArray(result) && result.length > 0) {
			center = extractCoordinates(result[0].location);
		} else {
			center = extractCoordinates(station);
		}
		
		// Create map if element exists
		const mapElement = document.getElementById('pirep-map');
		if (mapElement) {
			// If map already exists, destroy it to prevent conflicts
			if (map) {
				// Clear any existing markers
				markers.forEach(marker => {
					if (marker.setMap) {
						marker.setMap(null);
					}
				});
				markers = [];
				
				// Reset the map instance
				map = null;
			}
			
			// Custom map styles
			const mapStyles = [
				{
					featureType: "administrative",
					elementType: "geometry",
					stylers: [{ visibility: "simplified" }]
				},
				{
					featureType: "poi",
					stylers: [{ visibility: "off" }]
				},
				{
					featureType: "road",
					elementType: "labels.icon",
					stylers: [{ visibility: "off" }]
				},
				{
					featureType: "transit",
					stylers: [{ visibility: "off" }]
				},
				{
					featureType: "water",
					elementType: "geometry",
					stylers: [{ color: "#b3d1ff" }]
				},
				{
					featureType: "landscape",
					elementType: "geometry",
					stylers: [{ color: "#e6f2ff" }]
				}
			];
			
			map = new google.maps.Map(mapElement, {
				center,
				zoom: 8,
				mapTypeId: 'terrain',
				mapTypeControlOptions: {
					style: google.maps.MapTypeControlStyle.DROPDOWN_MENU,
					position: google.maps.ControlPosition.TOP_RIGHT,
					mapTypeIds: ['terrain', 'satellite', 'hybrid', 'styled_map']
				},
				styles: mapStyles,
				fullscreenControl: true,
				streetViewControl: false,
				zoomControlOptions: {
					position: google.maps.ControlPosition.RIGHT_CENTER
				}
			});
			
			// Create custom styled map type
			const styledMapType = new google.maps.StyledMapType(mapStyles, { name: 'Aviation Map' });
			map.mapTypes.set('styled_map', styledMapType);
			
			// Add custom controls
			addLegendControl(map);
		}
	}

	// Add a legend control to the map
	function addLegendControl(map) {
		const legendControl = document.createElement('div');
		legendControl.className = 'map-legend';
		legendControl.innerHTML = `
			<div class="map-legend-container bg-white p-2 shadow-md rounded-md text-xs">
				<div class="font-bold border-b pb-1 mb-1">PIREP Legend</div>
				<div class="grid grid-cols-2 gap-x-3 gap-y-1">
					<div class="flex items-center">
						<span class="inline-block w-3 h-3 rounded-full mr-1" style="background-color: #ef4444;"></span>
						UUA (Urgent)
					</div>
					<div class="flex items-center">
						<span class="inline-block w-3 h-3 rounded-full mr-1" style="background-color: #3b82f6;"></span>
						UA (Routine)
					</div>
					<div class="flex items-center">
						<span class="inline-block w-3 h-3 rounded-full mr-1" style="background-color: #b91c1c;"></span>
						Severe Turb
					</div>
					<div class="flex items-center">
						<span class="inline-block w-3 h-3 rounded-full mr-1" style="background-color: #ea580c;"></span>
						Moderate Turb
					</div>
					<div class="flex items-center">
						<span class="inline-block w-3 h-3 rounded-full mr-1" style="background-color: #ca8a04;"></span>
						Light Turb
					</div>
					<div class="flex items-center">
						<span class="inline-block w-3 h-3 rounded-full mr-1" style="background-color: #7e22ce;"></span>
						Severe Ice
					</div>
					<div class="flex items-center">
						<span class="inline-block w-3 h-3 rounded-full mr-1" style="background-color: #2563eb;"></span>
						Moderate Ice
					</div>
					<div class="flex items-center">
						<span class="inline-block w-3 h-3 rounded-full mr-1" style="background-color: #16a34a;"></span>
						Light Ice
					</div>
				</div>
			</div>
		`;
		map.controls[google.maps.ControlPosition.LEFT_BOTTOM].push(legendControl);
	}
	
	// Add PIREPs as markers on the map
	function addPirepsToMap() {
		if (!map || !Array.isArray(result)) return;
		
		// Clear existing markers
		markers.forEach(marker => marker.setMap(null));
		markers = [];
		
		// Create bounds object to fit all markers
		const bounds = new google.maps.LatLngBounds();
		
			// Create array to store all points for heatmap (if needed)
		const heatmapPoints = [];
		const pirepPositions = [];
		
		// Add markers for each PIREP
		result.forEach((pirep, index) => {
			const position = extractCoordinates(pirep.location);
			pirepPositions.push(position);
			const markerOptions = getPirepMarkerOptions(pirep);
			
			// Create labels according to hazard or report type
			const label = {
				text: markerOptions.icon,
				color: 'white',
				fontSize: '10px',
				fontWeight: 'bold'
			};
			
			// Create marker
			const marker = new google.maps.Marker({
				position,
				map,
				title: pirep.raw_text,
				label: label,
				icon: {
					path: google.maps.SymbolPath.CIRCLE,
					fillColor: markerOptions.color,
					fillOpacity: 0.8,
					strokeWeight: markerOptions.strokeWeight,
					strokeColor: 'white',
					scale: markerOptions.size
				},
				animation: google.maps.Animation.DROP,
				optimized: false, // Ensures proper z-index for animation
				zIndex: 100 + index
			});
			
			// Add weight for heatmap based on severity
			let weight = 0.5; // Default weight
			if (pirep.report_type === 'UUA') weight = 0.8;
			if (pirep.turbulence && pirep.turbulence.intensity) {
				const intensity = pirep.turbulence.intensity.toLowerCase();
				if (intensity.includes('sev')) weight = 1.0;
				else if (intensity.includes('mod')) weight = 0.7;
			}
			
			heatmapPoints.push({
				location: new google.maps.LatLng(position.lat, position.lng),
				weight: weight
			});
			
			// Create rich info window with formatted content
			const infoContent = `
				<div class="text-sm p-1" style="width: 220px; max-width: 100%;">
					<div class="font-bold text-blue-800 mb-1 flex items-center justify-between">
						<div>${pirep.location}</div>
						<div class="text-xs px-1 py-0.5 rounded ${pirep.report_type === 'UUA' ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800'}">
							${pirep.report_type || 'UA'}
						</div>
					</div>
					<div class="mb-1 text-gray-700">
						<strong>${pirep.aircraft_type || 'Unknown'}</strong> at ${getAltitudeText(pirep.altitude)}
					</div>
					${pirep.timestamp ? `<div class="text-xs text-gray-500 mb-1">${timeAgo(pirep.timestamp)}</div>` : ''}
					${pirep.turbulence ? `
						<div class="flex mt-1">
							<span class="text-xs font-medium px-1 py-0.5 rounded ${getTurbulenceClass(pirep.turbulence.intensity)}">
								${pirep.turbulence.intensity} Turbulence
							</span>
						</div>
					` : ''}
					${pirep.icing ? `
						<div class="flex mt-1">
							<span class="text-xs font-medium px-1 py-0.5 rounded ${getIcingClass(pirep.icing.intensity)}">
								${pirep.icing.intensity} Icing
							</span>
						</div>
					` : ''}
					<div class="mt-2 text-xs text-right">
						<a href="#" class="text-blue-600 hover:text-blue-800">View Details</a>
					</div>
				</div>
			`;
			
			const infoWindow = new google.maps.InfoWindow({
				content: infoContent,
				maxWidth: 300,
				pixelOffset: new google.maps.Size(0, -5)
			});
			
			// Add hover effect
			marker.addListener('mouseover', () => {
				infoWindow.open(map, marker);
			});
			
			marker.addListener('mouseout', () => {
				infoWindow.close();
			});
			
			// Add click handler to select this PIREP
			marker.addListener('click', () => {
				selectPirep(pirep);
				infoWindow.close();
			});
			
			// Store marker and add position to bounds
			markers.push(marker);
			bounds.extend(position);
		});
		
		// If we have PIREPs, draw connecting lines between them
		if (pirepPositions.length > 1) {
			// Sort positions by timestamp if available
			const sortedPositions = [...pirepPositions];
			
			// Create a flight path (line) connecting PIREPs
			const flightPath = new google.maps.Polyline({
				path: sortedPositions,
				geodesic: true,
				strokeColor: '#1e40af', // Blue-800
				strokeOpacity: 0.5,
				strokeWeight: 2,
				map: map
			});
			markers.push(flightPath);
		}
		
		// Create a heatmap layer for visualization intensity
		if (heatmapPoints.length > 3) {
			const heatmap = new google.maps.visualization.HeatmapLayer({
				data: heatmapPoints,
				map: map,
				radius: 30,
				opacity: 0.7,
				gradient: [
					'rgba(0, 255, 255, 0)',
					'rgba(0, 255, 255, 1)',
					'rgba(0, 191, 255, 1)',
					'rgba(0, 127, 255, 1)',
					'rgba(0, 63, 255, 1)',
					'rgba(0, 0, 255, 1)',
					'rgba(0, 0, 223, 1)',
					'rgba(0, 0, 191, 1)',
					'rgba(0, 0, 159, 1)',
					'rgba(0, 0, 127, 1)',
					'rgba(63, 0, 91, 1)',
					'rgba(127, 0, 63, 1)',
					'rgba(191, 0, 31, 1)',
					'rgba(255, 0, 0, 1)'
				]
			});
			
			// Add a toggle control for the heatmap
			const heatmapToggle = document.createElement('div');
			heatmapToggle.className = 'heatmap-toggle bg-white p-2 shadow-md rounded-md text-xs mt-2';
			heatmapToggle.innerHTML = `
				<label class="flex items-center cursor-pointer">
					<input type="checkbox" checked class="form-checkbox h-3 w-3 text-blue-600 rounded">
					<span class="ml-1">Heatmap</span>
				</label>
			`;
			map.controls[google.maps.ControlPosition.LEFT_BOTTOM].push(heatmapToggle);
			
			// Handle toggle click
			const checkbox = heatmapToggle.querySelector('input');
			checkbox.addEventListener('change', () => {
				heatmap.setMap(checkbox.checked ? map : null);
			});
			
			// Store for cleanup
			markers.push(heatmap);
		}
		
		// Center the station location with a custom marker
		const stationPosition = extractCoordinates(station);
		const stationMarker = new google.maps.Marker({
			position: stationPosition,
			map,
			title: station,
			icon: {
				path: google.maps.SymbolPath.CIRCLE,
				fillColor: '#4b5563', // gray-600
				fillOpacity: 0.8,
				strokeWeight: 2,
				strokeColor: 'white',
				scale: 10
			},
			label: {
				text: station,
				color: 'white',
				fontSize: '10px',
				fontWeight: 'bold'
			},
			zIndex: 1000 // Ensure on top
		});
		markers.push(stationMarker);
		bounds.extend(stationPosition);
		
		// Add a circle showing the search radius with gradient effect
		const searchCircle = new google.maps.Circle({
			strokeColor: '#4b5563', // gray-600
			strokeOpacity: 0.5,
			strokeWeight: 2,
			fillColor: '#9ca3af', // gray-400
			fillOpacity: 0.1,
			map,
			center: stationPosition,
			radius: distance * 1852, // Convert nm to meters (1 nm = 1852 m)
			zIndex: 50
		});
		
		// Add range rings every 50nm
		const ringInterval = 50 * 1852; // 50nm in meters
		const numRings = Math.floor(distance / 50);
		for (let i = 1; i <= numRings; i++) {
			const rangeRing = new google.maps.Circle({
				strokeColor: '#6b7280', // gray-500
				strokeOpacity: 0.3,
				strokeWeight: 1,
				fillColor: 'transparent',
				map,
				center: stationPosition,
				radius: i * ringInterval,
				zIndex: 49
			});
			markers.push(rangeRing);
		}
		
		// Fit map to show all markers and the search radius
		map.fitBounds(bounds);
		
		// If bounds are too small, use the search circle bounds
		if (map.getZoom() > 9) {
			map.fitBounds(searchCircle.getBounds());
		}
		
		// Add animated pulse effect to station
		animateStationMarker();
	}
	
	// Add a pulsing animation effect to station marker
	function animateStationMarker() {
		let growing = true;
		let size = 10;
		const animateMarker = () => {
			// Find station marker
			const stationMarker = markers.find(m => m.getTitle && m.getTitle() === station);
			if (!stationMarker || !mapVisible) return;
			
			// Update size
			if (growing) {
				size += 0.1;
				if (size >= 12) growing = false;
			} else {
				size -= 0.1;
				if (size <= 10) growing = true;
			}
			
			// Update icon scale
			const icon = stationMarker.getIcon();
			icon.scale = size;
			stationMarker.setIcon(icon);
			
			// Request next frame
			requestAnimationFrame(animateMarker);
		};
		
		// Start animation
		requestAnimationFrame(animateMarker);
	}

	// Toggle map visibility
	function toggleMap() {
		mapVisible = !mapVisible;
		
		if (mapVisible) {
			setTimeout(() => {
				initMap();
			}, 100);
		}
	}

	// Fetch PIREP data from the API
	async function fetchPirep() {
		loading = true;
		error = null;
		result = null;
		selectedPirep = null;

		try {
			const url = `/api/v1/pirep/${station}?distance=${distance}&age=${age}`;
			
			const response = await fetch(url);
			
			if (!response.ok) {
				throw new Error(`API error: ${response.status}`);
			}

			result = await response.json();
			
			// Set the first PIREP as selected if available
			if (Array.isArray(result) && result.length > 0) {
				selectedPirep = result[0];
				// Initialize the map after data is loaded
				if (mapVisible) {
					setTimeout(() => {
						initMap();
					}, 100);
				}
			}
		} catch (err) {
			error = err.message || 'Failed to fetch PIREP data';
		} finally {
			loading = false;
		}
	}
	
	// Handle sidebar toggle
	function toggleSidebar() {
		sidebarOpen = !sidebarOpen;
	}
	
	// Set the selected PIREP
	function selectPirep(pirep) {
		selectedPirep = pirep;
		
		// On mobile, close sidebar after selection
		if (window.innerWidth < 768) {
			sidebarOpen = false;
		}
	}

	// Auto-fetch data on component mount
	onMount(() => {
		fetchPirep();
	});
</script>

<svelte:head>
	<title>Cockpit PIREP Display</title>
	<!-- Include heroicons for better icons -->
	<link href="https://cdn.jsdelivr.net/npm/heroicons@1.0.1/outline/css/heroicons.min.css" rel="stylesheet">
</svelte:head>

<div class="bg-gray-100 min-h-screen">
	<div class="container mx-auto px-2 py-4">
		<!-- Header -->
		<div class="bg-white rounded-lg shadow-md p-4 mb-4">
			<h1 class="text-2xl font-bold text-gray-800">Cockpit PIREP Display</h1>
			<p class="text-gray-600 mt-1">
				Visualize Pilot Reports (PIREPs) in a format designed for flight planning and situational awareness.
			</p>
		</div>
		
		<!-- Search form -->
		<div class="bg-white rounded-lg shadow-md p-4 mb-4">
			<div class="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
				<div>
					<label for="station" class="block text-sm font-medium text-gray-700 mb-1">Airport Code</label>
					<input
						type="text"
						id="station"
						bind:value={station}
						placeholder="e.g. KATL, KJFK"
						class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
					/>
				</div>

				<div>
					<label for="distance" class="block text-sm font-medium text-gray-700 mb-1">Distance (nm)</label>
					<input
						type="number"
						id="distance"
						bind:value={distance}
						min="1"
						max="500"
						class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
					/>
				</div>

				<div>
					<label for="age" class="block text-sm font-medium text-gray-700 mb-1">Age (hours)</label>
					<input
						type="number"
						id="age"
						bind:value={age}
						min="0.5"
						max="24"
						step="0.5"
						class="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
					/>
				</div>

				<div>
					<button
						on:click={fetchPirep}
						class="w-full px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
						disabled={loading}
					>
						{#if loading}
							<svg class="animate-spin inline-block -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
							Loading...
						{:else}
							Fetch PIREPs
						{/if}
					</button>
				</div>
			</div>
		</div>
		
		<!-- Error display -->
		{#if error}
			<div class="bg-red-50 border-l-4 border-red-500 p-4 mb-4">
				<div class="flex">
					<div class="flex-shrink-0">
						<svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
							<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zm-1 9a1 1 0 01-1-1V8a1 1 0 112 0v6a1 1 0 01-1 1z" clip-rule="evenodd"/>
						</svg>
					</div>
					<div class="ml-3">
						<h3 class="text-sm font-medium text-red-800">Error</h3>
						<div class="mt-1 text-sm text-red-700">{error}</div>
					</div>
				</div>
			</div>
		{/if}
		
		<!-- Main PIREP display area with map -->
		{#if result && Array.isArray(result) && result.length > 0}
			<!-- Toggle buttons for different views -->
			<div class="bg-white rounded-lg shadow-md p-2 mb-4">
				<div class="flex flex-wrap gap-2">
					<!-- Sidebar toggle (mobile only) -->
					<button 
						class="md:hidden flex items-center justify-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
						on:click={toggleSidebar}
					>
						{sidebarOpen ? 'Hide' : 'Show'} List
						<svg class="ml-2 -mr-1 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
							<path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
						</svg>
					</button>
					
					<!-- Map toggle -->
					<button 
						class="flex items-center justify-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
						on:click={toggleMap}
					>
						{mapVisible ? 'Hide' : 'Show'} Map
						<svg class="ml-2 -mr-1 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
							{#if mapVisible}
								<path fill-rule="evenodd" d="M5 10a1 1 0 011-1h8a1 1 0 110 2H6a1 1 0 01-1-1z" clip-rule="evenodd" />
							{:else}
								<path fill-rule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clip-rule="evenodd" />
							{/if}
						</svg>
					</button>
					
					<!-- Count display -->
					<span class="ml-auto px-2 py-1 bg-blue-100 text-blue-800 text-sm rounded-md flex items-center">
						{result.length} PIREP{result.length !== 1 ? 's' : ''} found
					</span>
				</div>
			</div>
			
			<!-- Layout with optional map -->
			<div class="grid grid-cols-1 gap-4 mb-4">
				<!-- Map container (conditionally displayed) -->
				{#if mapVisible}
					<div class="bg-white rounded-lg shadow-md overflow-hidden p-0">
						<div id="pirep-map" class="w-full h-[400px]"></div>
					</div>
				{/if}
			</div>
			
			<!-- Main content area with sidebar and details -->
			<div class="relative">
				<div class="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4">
					<!-- Sidebar listing PIREPs -->
					{#if sidebarOpen}
						<div class="md:w-1/3 lg:w-1/4 bg-white rounded-lg shadow-md overflow-hidden">
							<div class="p-3 bg-blue-600 text-white font-semibold flex justify-between items-center">
								<span>PIREP List</span>
								<span class="text-sm font-normal">
									{station}, {distance}nm
								</span>
							</div>
							<div class="overflow-y-auto max-h-[calc(100vh-300px)]">
								{#each result as pirep, i}
									<button 
										on:click={() => selectPirep(pirep)}
										class="w-full text-left p-3 border-b border-gray-200 hover:bg-blue-50 focus:outline-none focus:bg-blue-50 {selectedPirep === pirep ? 'bg-blue-50 border-l-4 border-l-blue-600' : ''}"
									>
										<!-- PIREP summary card -->
										<div class="flex items-start">
											<!-- Left icon showing report type -->
											<div class="flex-shrink-0 rounded-full p-2 mr-3 {pirep.report_type === 'UUA' ? 'bg-red-100' : 'bg-blue-100'}">
												<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 {pirep.report_type === 'UUA' ? 'text-red-700' : 'text-blue-700'}" fill="none" viewBox="0 0 24 24" stroke="currentColor">
													<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
												</svg>
											</div>
											
											<!-- PIREP summary info -->
											<div class="flex-1 min-w-0">
												<div class="flex justify-between items-center">
													<div class="flex items-center">
														<span class="inline-flex items-center justify-center bg-gray-200 rounded-full h-5 w-5 text-xs font-bold mr-1">
															{i + 1}
														</span>
														<p class="text-sm font-medium text-gray-900 truncate">
															{pirep.location}
														</p>
													</div>
													<p class="text-xs text-gray-500">
														{pirep.timestamp ? timeAgo(pirep.timestamp) : ''}
													</p>
												</div>
												<p class="text-sm text-gray-500 truncate">
													{pirep.aircraft_type || 'Unknown aircraft'} at {getAltitudeText(pirep.altitude)}
												</p>
												
												<!-- Hazard indicators -->
												<div class="mt-1 flex flex-wrap gap-1">
													{#if pirep.turbulence && pirep.turbulence.intensity}
														<span class="inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium {getTurbulenceClass(pirep.turbulence.intensity)}">
															{pirep.turbulence.intensity} Turbulence
														</span>
													{/if}
													
													{#if pirep.icing && pirep.icing.intensity}
														<span class="inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium {getIcingClass(pirep.icing.intensity)}">
															{pirep.icing.intensity} Icing
														</span>
													{/if}
												</div>
											</div>
										</div>
									</button>
								{/each}
							</div>
						</div>
					{/if}
					
					<!-- Main PIREP display area -->
					<div class="flex-1 bg-white rounded-lg shadow-md overflow-hidden">
						{#if selectedPirep}
							<div class="p-4 bg-white text-gray-800">
								<div class="flex items-center justify-between">
									<h2 class="text-xl font-bold">
										<span class={`inline-flex items-center justify-center rounded-full h-6 w-6 mr-2 ${selectedPirep.report_type === 'UUA' ? 'bg-red-500' : 'bg-blue-500'}`}>
											<span class="text-white text-xs font-bold">{selectedPirep.report_type || 'UA'}</span>
										</span>
										{selectedPirep.location} PIREP
									</h2>
									
									{#if selectedPirep.timestamp}
										<span class="text-sm text-gray-600">
											{formatDate(selectedPirep.timestamp)}
										</span>
									{/if}
								</div>
								
								<p class="mt-1 text-gray-600">
									{generatePirepSummary(selectedPirep)}
								</p>
							</div>
							
							<!-- Interactive PIREP hover display -->
							<div class="p-4 bg-white text-gray-800 border-b border-gray-200">
								<div class="font-mono text-sm bg-gray-100 p-3 rounded">
									<PirepHoverDisplay pirepString={selectedPirep.raw_text} locationCode={selectedPirep.location} />
								</div>
							</div>
							
							<!-- PIREP details -->
							<div class="p-4">
								<!-- Aircraft and location details -->
								<div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
									<div class="bg-gray-50 p-4 rounded-lg">
										<h3 class="font-medium text-gray-700 mb-2 flex items-center">
											<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
												<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
											</svg>
											Report Details
										</h3>
										
										<table class="w-full text-sm">
											<tbody>
												<tr>
													<td class="py-1 text-gray-600">Type:</td>
													<td class="py-1 font-medium">
														<span class={`inline-flex items-center px-2.5 py-0.5 rounded-md text-sm font-medium ${selectedPirep.report_type === 'UUA' ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800'}`}>
															{selectedPirep.report_type === 'UUA' ? 'Urgent PIREP' : 'Routine PIREP'}
														</span>
													</td>
												</tr>
												<tr>
													<td class="py-1 text-gray-600">Location:</td>
													<td class="py-1">
														{getDirectionFromLocation(selectedPirep.location)}
													</td>
												</tr>
												{#if selectedPirep.aircraft_type}
													<tr>
														<td class="py-1 text-gray-600">Aircraft:</td>
														<td class="py-1">{selectedPirep.aircraft_type}</td>
													</tr>
												{/if}
												{#if selectedPirep.altitude !== undefined}
													<tr>
														<td class="py-1 text-gray-600">Altitude:</td>
														<td class="py-1">{getAltitudeText(selectedPirep.altitude)}</td>
													</tr>
												{/if}
											</tbody>
										</table>
										
										{#if getMapUrl(selectedPirep)}
											<div class="mt-3">
												<a 
													href={getMapUrl(selectedPirep)} 
													target="_blank" 
													rel="noopener noreferrer"
													class="inline-flex items-center text-sm text-blue-600 hover:text-blue-800"
												>
													<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
														<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
													</svg>
													View on Map
												</a>
											</div>
										{/if}
									</div>
									
									<div class="bg-gray-50 p-4 rounded-lg">
										<h3 class="font-medium text-gray-700 mb-2 flex items-center">
											<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
												<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
											</svg>
											Weather Hazards
										</h3>
										
										<div class="space-y-4">
											{#if selectedPirep.turbulence}
												<div>
													<h4 class="text-sm font-medium text-gray-700 mb-1">Turbulence</h4>
													<div class="flex flex-wrap items-center">
														<span class={`px-2.5 py-1 rounded-md text-sm font-medium ${getTurbulenceClass(selectedPirep.turbulence.intensity)}`}>
															{selectedPirep.turbulence.intensity || 'Reported'}
														</span>
														
														{#if selectedPirep.turbulence.frequency}
															<span class="ml-2 text-sm text-gray-600">{selectedPirep.turbulence.frequency}</span>
														{/if}
														
														{#if selectedPirep.turbulence.altitude}
															<span class="ml-2 text-sm text-gray-600">at {selectedPirep.turbulence.altitude} ft</span>
														{/if}
													</div>
												</div>
											{/if}
											
											{#if selectedPirep.icing}
												<div>
													<h4 class="text-sm font-medium text-gray-700 mb-1">Icing</h4>
													<div class="flex flex-wrap items-center">
														<span class={`px-2.5 py-1 rounded-md text-sm font-medium ${getIcingClass(selectedPirep.icing.intensity)}`}>
															{selectedPirep.icing.intensity || 'Reported'}
														</span>
														
														{#if selectedPirep.icing.type}
															<span class="ml-2 text-sm text-gray-600">{selectedPirep.icing.type}</span>
														{/if}
													</div>
												</div>
											{/if}
											
											{#if !selectedPirep.turbulence && !selectedPirep.icing}
												<p class="text-sm text-gray-500 italic">No specific hazards reported</p>
											{/if}
										</div>
									</div>
								</div>
								
								<!-- Additional details -->
								<div class="grid grid-cols-1 gap-4">
									<!-- Sky conditions and remarks -->
									<div class="bg-gray-50 p-4 rounded-lg">
										<h3 class="font-medium text-gray-700 mb-3 flex items-center">
											<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
												<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
											</svg>
											Additional Information
										</h3>
										
										<div class="space-y-3">
											{#if selectedPirep.sky_conditions}
												<div>
													<h4 class="text-sm font-medium text-gray-700">Sky Conditions</h4>
													<p class="mt-1 text-sm text-gray-800 bg-white p-2 rounded border border-gray-200">
														{selectedPirep.sky_conditions}
													</p>
												</div>
											{/if}
											
											{#if selectedPirep.remarks}
												<div>
													<h4 class="text-sm font-medium text-gray-700">Remarks</h4>
													<p class="mt-1 text-sm text-gray-800 bg-white p-2 rounded border border-gray-200">
														{selectedPirep.remarks}
													</p>
												</div>
											{/if}
											
											{#if !selectedPirep.sky_conditions && !selectedPirep.remarks}
												<p class="text-sm text-gray-500 italic">No additional information reported</p>
											{/if}
										</div>
									</div>
								</div>
							</div>
						{:else}
							<!-- No PIREP selected state -->
							<div class="p-8 text-center">
								<svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 mx-auto text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
								</svg>
								<h3 class="mt-2 text-lg font-medium text-gray-900">No PIREP Selected</h3>
								<p class="mt-1 text-sm text-gray-500">
									Select a PIREP from the sidebar to view detailed information.
								</p>
							</div>
						{/if}
					</div>
				</div>
			</div>
		{:else if result && Array.isArray(result) && result.length === 0}
			<!-- No results state -->
			<div class="bg-white rounded-lg shadow-md p-8 text-center">
				<svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 mx-auto text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
				</svg>
				<h3 class="mt-2 text-lg font-medium text-gray-900">No PIREPs Found</h3>
				<p class="mt-1 text-sm text-gray-500">
					No pilot reports were found for {station} within {distance} nm in the past {age} hours.
				</p>
				<p class="mt-4 text-sm text-gray-500">
					Try expanding your search area or time range, or search for a different airport.
				</p>
			</div>
		{:else if !loading && !error}
			<!-- Empty state when no search performed -->
			<div class="bg-blue-50 border-blue-200 border rounded-lg p-8 text-center">
				<svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 mx-auto text-blue-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
				</svg>
				<h3 class="text-lg font-medium text-blue-700 mb-2">No PIREP Data</h3>
				<p class="text-blue-600">
					Enter an airport code and click "Fetch PIREPs" to view pilot reports in the area.
				</p>
			</div>
		{/if}
		
		<!-- Footer with helpful info -->
		<div class="mt-4 bg-white rounded-lg shadow-md p-4">
			<h3 class="font-medium text-gray-700 mb-2">Understanding PIREPs</h3>
			<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
				<div>
					<h4 class="text-sm font-medium text-gray-700 mb-1">Turbulence Intensity</h4>
					<div class="flex flex-wrap gap-2">
						<span class="inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-medium bg-yellow-100 text-yellow-800">
							LGT (Light)
						</span>
						<span class="inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-medium bg-orange-100 text-orange-800">
							MOD (Moderate)
						</span>
						<span class="inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-medium bg-red-100 text-red-800">
							SEV (Severe)
						</span>
					</div>
				</div>
				
				<div>
					<h4 class="text-sm font-medium text-gray-700 mb-1">Icing Intensity</h4>
					<div class="flex flex-wrap gap-2">
						<span class="inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-medium bg-green-100 text-green-800">
							TRACE
						</span>
						<span class="inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-medium bg-green-100 text-green-800">
							LGT (Light)
						</span>
						<span class="inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-medium bg-blue-100 text-blue-800">
							MOD (Moderate)
						</span>
						<span class="inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-medium bg-purple-100 text-purple-800">
							SEV (Severe)
						</span>
					</div>
				</div>
				
				<div>
					<h4 class="text-sm font-medium text-gray-700 mb-1">Report Types</h4>
					<div class="flex flex-wrap gap-2">
						<span class="inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-medium bg-blue-100 text-blue-800">
							UA (Routine)
						</span>
						<span class="inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-medium bg-red-100 text-red-800">
							UUA (Urgent)
						</span>
					</div>
				</div>
			</div>
			
			
		</div>
	</div>
</div>

<style>
	/* Add some styling for the PIREP hover component elements */
	:global([data-type="location"]) {
		color: #3b82f6; /* blue-500 */
	}
	:global([data-type="report_type"]) {
		color: #ef4444; /* red-500 */
	}
	:global([data-type="time"]) {
		color: #8b5cf6; /* purple-500 */
	}
	:global([data-type="altitude"]) {
		color: #10b981; /* emerald-500 */
	}
	:global([data-type="aircraft"]) {
		color: #f59e0b; /* amber-500 */
	}
	:global([data-type="turbulence"]) {
		color: #ef4444; /* red-500 */
	}
	:global([data-type="icing"]) {
		color: #6366f1; /* indigo-500 */
	}
	:global([data-type="sky"]) {
		color: #0ea5e9; /* sky-500 */
	}
	:global([data-type="remarks"]) {
		color: #64748b; /* slate-500 */
	}
	
	/* Make the map container responsive */
	#pirep-map {
		min-height: 400px;
	}
	
	@media (min-width: 768px) {
		#pirep-map {
			min-height: 500px;
		}
	}
</style>