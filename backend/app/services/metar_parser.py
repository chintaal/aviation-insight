"""
METAR Parser - Parses raw METAR strings and generates pilot-friendly summaries

This module provides comprehensive parsing of METAR (Meteorological Aerodrome Report) strings and
produces structured data and human-readable summaries suitable for pilot briefings.
"""
import re
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Weather phenomena codes and their descriptions
WEATHER_PHENOMENA = {
    # Intensity/proximity prefixes
    '+': 'Heavy',
    '-': 'Light',
    'VC': 'Vicinity',
    
    # Weather phenomena
    'MI': 'Shallow',
    'PR': 'Partial',
    'BC': 'Patches',
    'DR': 'Low Drifting',
    'BL': 'Blowing',
    'SH': 'Shower',
    'TS': 'Thunderstorm',
    'FZ': 'Freezing',
    
    # Precipitation
    'DZ': 'Drizzle',
    'RA': 'Rain',
    'SN': 'Snow',
    'SG': 'Snow Grains',
    'IC': 'Ice Crystals',
    'PL': 'Ice Pellets',
    'GR': 'Hail',
    'GS': 'Small Hail',
    'UP': 'Unknown Precipitation',
    
    # Obscuration
    'FG': 'Fog',
    'BR': 'Mist',
    'HZ': 'Haze',
    'VA': 'Volcanic Ash',
    'DU': 'Widespread Dust',
    'SA': 'Sand',
    'PY': 'Spray',
    'FU': 'Smoke',
    
    # Other
    'SQ': 'Squall',
    'PO': 'Dust/Sand Whirls',
    'DS': 'Duststorm',
    'SS': 'Sandstorm',
    'FC': 'Funnel Cloud/Tornado/Waterspout'
}

# Cloud cover codes and their descriptions
CLOUD_COVER_CODES = {
    'SKC': 'Sky Clear',
    'CLR': 'Clear (no clouds below 12,000 ft)',
    'NSC': 'No Significant Clouds',
    'NCD': 'No Clouds Detected',
    'FEW': 'Few (1-2 oktas)',
    'SCT': 'Scattered (3-4 oktas)',
    'BKN': 'Broken (5-7 oktas)',
    'OVC': 'Overcast (8 oktas)',
    'VV': 'Vertical Visibility (sky obscured)'
}

# Cardinal directions
CARDINAL_DIRECTIONS = [
    'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
]


class MetarParser:
    """Parser for METAR (Meteorological Aerodrome Report) strings"""
    
    def __init__(self, metar_string: str):
        """Initialize the parser with a METAR string"""
        self.raw = metar_string.strip()
        self.parts = self.raw.split()
        self.parsed_metar = {}
        self.pilot_summary = ""
        self.additional_info = {}
    
    def parse(self) -> Dict[str, Any]:
        """Parse the METAR string and return structured data"""
        if not self.raw:
            return {}
            
        try:
            # Initialize parsed data structure
            self.parsed_metar = {
                "raw_text": self.raw,
                "station": None,
                "time": None,
                "auto": False,
                "wind": None,
                "visibility": None,
                "weather": [],
                "clouds": [],
                "temperature": None,
                "dewpoint": None,
                "altimeter": None,
                "remarks": None,
                "ceiling": None,
                "flight_category": None
            }
            
            # Start parsing
            current_part = 0
            
            # Station identifier (first part)
            if len(self.parts) > current_part:
                self.parsed_metar["station"] = self.parts[current_part]
                current_part += 1
            
            # Date/time (second part) in format DDHHMMZ
            if current_part < len(self.parts) and self.parts[current_part].endswith('Z'):
                self._parse_time(self.parts[current_part])
                current_part += 1
            
            # Check for AUTO indicator
            if current_part < len(self.parts) and self.parts[current_part] == 'AUTO':
                self.parsed_metar["auto"] = True
                current_part += 1
            
            # Parse wind
            if current_part < len(self.parts) and (self.parts[current_part].endswith('KT') or 
                                                  self.parts[current_part].endswith('MPS')):
                self._parse_wind(self.parts[current_part])
                current_part += 1
                
                # Check for variable wind direction
                if current_part < len(self.parts) and re.match(r'^\d{3}V\d{3}$', self.parts[current_part]):
                    self._parse_variable_wind(self.parts[current_part])
                    current_part += 1
            
            # Parse visibility
            current_part = self._parse_visibility(current_part)
            
            # Skip runway visual range information
            while current_part < len(self.parts) and self.parts[current_part].startswith('R') and '/' in self.parts[current_part]:
                current_part += 1
            
            # Parse weather phenomena
            current_part = self._parse_weather(current_part)
            
            # Parse cloud information
            current_part = self._parse_clouds(current_part)
            
            # Parse temperature and dewpoint
            if current_part < len(self.parts) and '/' in self.parts[current_part]:
                self._parse_temp_dewpoint(self.parts[current_part])
                current_part += 1
            
            # Parse altimeter
            if current_part < len(self.parts) and (self.parts[current_part].startswith('A') or 
                                                  self.parts[current_part].startswith('Q')):
                self._parse_altimeter(self.parts[current_part])
                current_part += 1
            
            # Collect remarks
            self._parse_remarks(current_part)
            
            # Calculate flight category
            self._calculate_flight_category()
            
            # Generate pilot summary
            self._generate_summary()
            
            return self.parsed_metar
            
        except Exception as e:
            logger.error(f"Error parsing METAR: {str(e)}")
            return {"raw_text": self.raw, "error": str(e)}
    
    def _parse_time(self, time_str: str) -> None:
        """Parse the time section of the METAR"""
        if len(time_str) >= 7:  # Format: DDHHMMZ
            try:
                day = int(time_str[0:2])
                hour = int(time_str[2:4])
                minute = int(time_str[4:6])
                
                # Create a datetime object (use current year/month)
                now = datetime.now(timezone.utc)
                observation_time = datetime(now.year, now.month, day, hour, minute, tzinfo=timezone.utc)
                
                # If the date is in the future, it's likely from the previous month
                if observation_time > now and (observation_time - now).days > 15:
                    if observation_time.month == 1:
                        observation_time = observation_time.replace(year=now.year - 1, month=12)
                    else:
                        observation_time = observation_time.replace(month=observation_time.month - 1)
                
                self.parsed_metar["time"] = {
                    "raw": time_str,
                    "day": day,
                    "hour": hour,
                    "minute": minute,
                    "datetime": observation_time,
                    "iso": observation_time.isoformat()
                }
            except ValueError:
                self.parsed_metar["time"] = {"raw": time_str}
        else:
            self.parsed_metar["time"] = {"raw": time_str}
    
    def _parse_wind(self, wind_str: str) -> None:
        """Parse the wind section of the METAR"""
        # Handle calm winds
        if wind_str == '00000KT':
            self.parsed_metar["wind"] = {
                "direction": 0,
                "speed": 0,
                "gust": None,
                "unit": "KT",
                "cardinal": "N",
                "text": "Calm"
            }
            return
        
        # Handle variable winds
        if wind_str.startswith('VRB'):
            speed_part = re.search(r'VRB(\d+)(G(\d+))?KT', wind_str)
            if speed_part:
                speed = int(speed_part.group(1))
                gust = int(speed_part.group(3)) if speed_part.group(3) else None
                self.parsed_metar["wind"] = {
                    "direction": "VRB",
                    "speed": speed,
                    "gust": gust,
                    "unit": "KT",
                    "variable": True,
                    "text": f"Variable at {speed} knots{' gusting to ' + str(gust) + ' knots' if gust else ''}"
                }
            return
        
        # Regular wind pattern
        wind_match = re.match(r'^(\d{3})(\d{2,3})(G(\d{2,3}))?(?:KT|MPS)$', wind_str)
        if wind_match:
            direction = int(wind_match.group(1))
            speed = int(wind_match.group(2))
            gust = int(wind_match.group(4)) if wind_match.group(4) else None
            unit = "KT" if wind_str.endswith("KT") else "MPS"
            
            # Convert direction to cardinal
            cardinal = self._get_cardinal_direction(direction)
            
            self.parsed_metar["wind"] = {
                "direction": direction,
                "speed": speed,
                "gust": gust,
                "unit": unit,
                "cardinal": cardinal,
                "variable": False,
                "text": f"From {direction}° ({cardinal}) at {speed} {unit}{' gusting to ' + str(gust) + ' ' + unit if gust else ''}"
            }
    
    def _parse_variable_wind(self, var_str: str) -> None:
        """Parse variable wind direction range"""
        match = re.match(r'^(\d{3})V(\d{3})$', var_str)
        if match and "wind" in self.parsed_metar and self.parsed_metar["wind"]:
            from_dir = int(match.group(1))
            to_dir = int(match.group(2))
            
            self.parsed_metar["wind"]["variable_direction"] = {
                "from": from_dir,
                "to": to_dir
            }
            
            # Update the text description
            if "text" in self.parsed_metar["wind"]:
                self.parsed_metar["wind"]["text"] += f" (varying between {from_dir}° and {to_dir}°)"
    
    def _parse_visibility(self, current_part: int) -> int:
        """Parse the visibility section of the METAR"""
        if current_part >= len(self.parts):
            return current_part
        
        vis_str = self.parts[current_part]
        
        # Handle CAVOK (Ceiling and Visibility OK)
        if vis_str == 'CAVOK':
            self.parsed_metar["visibility"] = {
                "distance": 10,
                "unit": "SM", 
                "cavok": True,
                "text": "10+ statute miles"
            }
            # CAVOK also implies no clouds below 5000 feet and no significant weather
            self.parsed_metar["clouds"] = [{
                "cover": "CAVOK",
                "base": None,
                "type": None,
                "ceiling": False
            }]
            return current_part + 1
        
        # Handle fractions (e.g., "1/2SM")
        if "/" in vis_str and vis_str.endswith("SM"):
            vis_value = self._fraction_to_float(vis_str.replace("SM", ""))
            self.parsed_metar["visibility"] = {
                "distance": vis_value,
                "unit": "SM",
                "text": f"{vis_str.replace('SM', '')} statute miles"
            }
            return current_part + 1
        
        # Handle "M1/4SM" format (less than 1/4 mile)
        if vis_str.startswith("M") and vis_str.endswith("SM"):
            vis_value = self._fraction_to_float(vis_str.replace("M", "").replace("SM", ""))
            self.parsed_metar["visibility"] = {
                "distance": vis_value,
                "unit": "SM",
                "less_than": True,
                "text": f"Less than {vis_str[1:].replace('SM', '')} statute miles"
            }
            return current_part + 1
        
        # Handle "X X/XSM" format (e.g., "1 1/2SM")
        if current_part < len(self.parts) - 1 and self.parts[current_part + 1].endswith("SM") and "/" in self.parts[current_part + 1]:
            whole = int(vis_str)
            fraction_part = self.parts[current_part + 1].replace("SM", "")
            vis_value = whole + self._fraction_to_float(fraction_part)
            
            self.parsed_metar["visibility"] = {
                "distance": vis_value,
                "unit": "SM",
                "text": f"{vis_str} {fraction_part} statute miles"
            }
            return current_part + 2
        
        # Handle standard visibility with SM
        if vis_str.endswith("SM"):
            try:
                vis_value = float(vis_str.replace("SM", ""))
                self.parsed_metar["visibility"] = {
                    "distance": vis_value,
                    "unit": "SM",
                    "text": f"{vis_value} statute miles"
                }
                return current_part + 1
            except ValueError:
                pass
        
        # Handle visibility in meters (e.g. "2000" or "2000M")
        if vis_str.isdigit() or (vis_str.endswith("M") and vis_str[:-1].isdigit()):
            vis_meters = int(vis_str[:-1] if vis_str.endswith("M") else vis_str)
            vis_miles = round(vis_meters / 1609.34, 1)  # Convert to statute miles
            
            self.parsed_metar["visibility"] = {
                "distance": vis_meters,
                "distance_sm": vis_miles,
                "unit": "M",
                "text": f"{vis_meters} meters ({vis_miles} statute miles)"
            }
            return current_part + 1
        
        # No visibility found
        return current_part
    
    def _parse_weather(self, current_part: int) -> int:
        """Parse the weather phenomena section of the METAR"""
        while current_part < len(self.parts):
            wx_str = self.parts[current_part]
            
            # Check if this part matches a weather phenomenon
            is_weather = False
            
            # It's weather if it starts with +, -, or VC
            if wx_str.startswith('+') or wx_str.startswith('-') or wx_str.startswith('VC'):
                is_weather = True
            
            # It's weather if it contains recognizable weather codes
            if not is_weather:
                for code in WEATHER_PHENOMENA:
                    if len(code) == 2 and code in wx_str:
                        is_weather = True
                        break
            
            if not is_weather:
                break
            
            # We've identified a weather element, now parse it
            intensity = ""
            description = []
            
            # Check for intensity/proximity prefix
            if wx_str.startswith('+'):
                intensity = "Heavy"
                wx_str = wx_str[1:]
            elif wx_str.startswith('-'):
                intensity = "Light"
                wx_str = wx_str[1:]
            elif wx_str.startswith('VC'):
                intensity = "Vicinity"
                wx_str = wx_str[2:]
            
            # Parse the remaining weather codes in pairs
            i = 0
            descriptions = []
            while i < len(wx_str):
                if i + 2 <= len(wx_str):
                    code = wx_str[i:i+2]
                    if code in WEATHER_PHENOMENA:
                        descriptions.append(WEATHER_PHENOMENA[code])
                i += 2
            
            if descriptions:
                weather_info = {
                    "raw": self.parts[current_part],
                    "intensity": intensity,
                    "descriptions": descriptions,
                    "text": (intensity + " " if intensity else "") + " ".join(descriptions)
                }
                self.parsed_metar["weather"].append(weather_info)
            
            current_part += 1
        
        return current_part
    
    def _parse_clouds(self, current_part: int) -> int:
        """Parse the cloud section of the METAR"""
        while current_part < len(self.parts):
            cloud_str = self.parts[current_part]
            
            # Check for recognized cloud patterns
            cloud_match = re.match(r'^(VV|FEW|SCT|BKN|OVC)(\d{3})(CB|TCU)?$', cloud_str)
            special_condition = cloud_str in ["SKC", "CLR", "NSC", "NCD"]
            
            if not (cloud_match or special_condition):
                break
            
            if special_condition:
                cloud_info = {
                    "cover": cloud_str,
                    "cover_text": CLOUD_COVER_CODES.get(cloud_str, cloud_str),
                    "base": None,
                    "type": None,
                    "ceiling": False
                }
                self.parsed_metar["clouds"].append(cloud_info)
            else:
                cover = cloud_match.group(1)
                height = int(cloud_match.group(2)) * 100  # Convert to feet
                cloud_type = cloud_match.group(3)
                
                cloud_type_full = None
                if cloud_type == "CB":
                    cloud_type_full = "Cumulonimbus"
                elif cloud_type == "TCU":
                    cloud_type_full = "Towering Cumulus"
                
                is_ceiling = cover in ["BKN", "OVC", "VV"]
                
                cloud_info = {
                    "cover": cover,
                    "cover_text": CLOUD_COVER_CODES.get(cover, cover),
                    "base": height,
                    "type": cloud_type,
                    "type_text": cloud_type_full,
                    "ceiling": is_ceiling
                }
                self.parsed_metar["clouds"].append(cloud_info)
                
                # Track ceiling (lowest broken or overcast layer)
                if is_ceiling and (self.parsed_metar["ceiling"] is None or height < self.parsed_metar["ceiling"]):
                    self.parsed_metar["ceiling"] = height
            
            current_part += 1
        
        return current_part
    
    def _parse_temp_dewpoint(self, temp_str: str) -> None:
        """Parse the temperature and dewpoint section of the METAR"""
        if '/' in temp_str:
            parts = temp_str.split('/')
            if len(parts) == 2:
                temp_raw, dew_raw = parts
                
                # Parse temperature
                if temp_raw.startswith('M'):
                    self.parsed_metar["temperature"] = -int(temp_raw[1:])
                else:
                    self.parsed_metar["temperature"] = int(temp_raw)
                
                # Parse dewpoint
                if dew_raw.startswith('M'):
                    self.parsed_metar["dewpoint"] = -int(dew_raw[1:])
                else:
                    self.parsed_metar["dewpoint"] = int(dew_raw)
    
    def _parse_altimeter(self, alt_str: str) -> None:
        """Parse the altimeter section of the METAR"""
        if alt_str.startswith('A'):
            # Inches of mercury (e.g. A2992)
            value = float(alt_str[1:]) / 100
            self.parsed_metar["altimeter"] = {
                "value": value,
                "unit": "inHg"
            }
        elif alt_str.startswith('Q'):
            # Hectopascals (e.g. Q1013)
            value = int(alt_str[1:])
            self.parsed_metar["altimeter"] = {
                "value": value,
                "unit": "hPa"
            }
    
    def _parse_remarks(self, current_part: int) -> None:
        """Parse remarks section (everything after RMK)"""
        rmk_index = -1
        for i in range(current_part, len(self.parts)):
            if self.parts[i] == 'RMK':
                rmk_index = i
                break
        
        if rmk_index >= 0:
            self.parsed_metar["remarks"] = {
                "raw": ' '.join(self.parts[rmk_index + 1:]),
                "parts": self.parts[rmk_index + 1:]
            }
    
    def _calculate_flight_category(self) -> None:
        """Calculate the flight category based on visibility and ceiling"""
        visibility = None
        ceiling = self.parsed_metar["ceiling"]
        
        if self.parsed_metar["visibility"] and "distance" in self.parsed_metar["visibility"]:
            if self.parsed_metar["visibility"].get("unit") == "SM":
                visibility = self.parsed_metar["visibility"]["distance"]
            elif self.parsed_metar["visibility"].get("unit") == "M":
                # Convert meters to statute miles
                visibility = self.parsed_metar["visibility"]["distance_sm"]
        
        # Default to VFR
        category = "VFR"
        
        # Apply flight category rules
        if (ceiling is not None and ceiling < 500) or (visibility is not None and visibility < 1):
            category = "LIFR"  # Low IFR
        elif (ceiling is not None and ceiling < 1000) or (visibility is not None and visibility < 3):
            category = "IFR"   # IFR
        elif (ceiling is not None and ceiling < 3000) or (visibility is not None and visibility < 5):
            category = "MVFR"  # Marginal VFR
        
        self.parsed_metar["flight_category"] = category
    
    def _get_cardinal_direction(self, degrees: int) -> str:
        """Convert wind direction in degrees to cardinal direction"""
        index = round(degrees / 22.5) % 16
        return CARDINAL_DIRECTIONS[index]
    
    def _fraction_to_float(self, fraction_str: str) -> float:
        """Convert a fraction string to a float"""
        if '/' in fraction_str:
            try:
                num, denom = fraction_str.split('/')
                return float(num) / float(denom)
            except (ValueError, ZeroDivisionError):
                return 0
        else:
            try:
                return float(fraction_str)
            except ValueError:
                return 0
    
    def _generate_summary(self) -> None:
        """Generate a pilot-friendly summary of the METAR"""
        station = self.parsed_metar.get("station", "")
        time_info = self.parsed_metar.get("time", {})
        
        # Start with the airport and time
        parts = [f"At {station}"]
        
        if "hour" in time_info and "minute" in time_info:
            parts[0] += f" as of {time_info['hour']}:{time_info['minute']:02d}Z"
        
        # Add flight category with descriptive text
        if self.parsed_metar.get("flight_category"):
            category = self.parsed_metar['flight_category']
            category_descriptions = {
                "VFR": "Visual Flight Rules",
                "MVFR": "Marginal Visual Flight Rules",
                "IFR": "Instrument Flight Rules",
                "LIFR": "Low Instrument Flight Rules"
            }
            description = category_descriptions.get(category, category)
            parts.append(f"Conditions are {category} ({description})")
        
        # Add wind information with operational impact
        if self.parsed_metar.get("wind"):
            wind = self.parsed_metar["wind"]
            wind_text = f"Wind {wind.get('text', '')}"
            
            # Add operational notes about wind
            if wind.get("direction") != "VRB" and isinstance(wind.get("direction"), int):
                runway_crosswind_note = self._get_runway_crosswind_note(wind.get("direction"), wind.get("speed", 0))
                if runway_crosswind_note:
                    wind_text += f". {runway_crosswind_note}"
            
            # Add note about gusty conditions if present
            if wind.get("gust") and wind.get("gust") > 10:
                gust_factor = wind.get("gust") - wind.get("speed", 0)
                if gust_factor > 10:
                    wind_text += f". Significant wind shear possible with {gust_factor} knot gust factor"
                elif gust_factor > 5:
                    wind_text += f". Be prepared for {gust_factor} knot gusts on approach"
            
            parts.append(wind_text)
        
        # Add visibility with operational context
        if self.parsed_metar.get("visibility") and "text" in self.parsed_metar["visibility"]:
            vis_text = f"Visibility {self.parsed_metar['visibility']['text']}"
            
            # Add operational context based on visibility distance
            if "distance" in self.parsed_metar["visibility"]:
                vis_distance = self.parsed_metar["visibility"]["distance"]
                if vis_distance < 1:
                    vis_text += ". Approach and landing will require precision instruments"
                elif vis_distance < 3:
                    vis_text += ". Instrument approach required"
                elif vis_distance < 5:
                    vis_text += ". Visual approach possible but exercise caution"
            
            parts.append(vis_text)
        
        # Add weather phenomena with operational impact
        if self.parsed_metar.get("weather") and len(self.parsed_metar["weather"]) > 0:
            weather_texts = []
            has_thunderstorm = False
            has_freezing = False
            has_rain = False
            has_snow = False
            has_fog = False
            
            for w in self.parsed_metar["weather"]:
                if "text" in w:
                    weather_texts.append(w["text"])
                    
                    # Check for special weather conditions
                    if "thunderstorm" in w["text"].lower() or "ts" in w["raw"].lower():
                        has_thunderstorm = True
                    if "freezing" in w["text"].lower() or "fz" in w["raw"].lower():
                        has_freezing = True
                    if "rain" in w["text"].lower() or "ra" in w["raw"].lower():
                        has_rain = True
                    if "snow" in w["text"].lower() or "sn" in w["raw"].lower():
                        has_snow = True
                    if "fog" in w["text"].lower() or "fg" in w["raw"].lower():
                        has_fog = True
            
            if weather_texts:
                weather_part = f"Weather: {', '.join(weather_texts)}"
                
                # Add operational notes about specific weather
                operational_notes = []
                if has_thunderstorm:
                    operational_notes.append("Expect turbulence and possible wind shear")
                if has_freezing:
                    operational_notes.append("Icing conditions likely")
                if has_rain and self.parsed_metar.get("temperature", 0) < 5:
                    operational_notes.append("Possibility of hydroplaning on wet runway")
                if has_snow:
                    operational_notes.append("Possible runway contamination and reduced braking action")
                if has_fog and self.parsed_metar.get("visibility", {}).get("distance", 10) < 3:
                    operational_notes.append("Reduced visual references on approach")
                
                if operational_notes:
                    weather_part += f". {' and '.join(operational_notes)}"
                
                parts.append(weather_part)
        
        # Add cloud information with operational impact
        if self.parsed_metar.get("clouds") and len(self.parsed_metar["clouds"]) > 0:
            cloud_parts = []
            has_cb = False
            has_tcu = False
            lowest_ceiling = None
            
            # Handle special sky conditions
            if len(self.parsed_metar["clouds"]) == 1 and self.parsed_metar["clouds"][0].get("cover") in ["SKC", "CLR", "NSC", "NCD", "CAVOK"]:
                cover_text = self.parsed_metar["clouds"][0].get("cover_text", self.parsed_metar["clouds"][0].get("cover"))
                cloud_parts.append(cover_text)
            else:
                for cloud in self.parsed_metar["clouds"]:
                    cloud_text = cloud.get("cover_text", cloud.get("cover", ""))
                    
                    if cloud.get("base") is not None:
                        cloud_text += f" at {cloud['base']} feet"
                        if cloud.get("ceiling") and (lowest_ceiling is None or cloud['base'] < lowest_ceiling):
                            lowest_ceiling = cloud['base']
                    
                    if cloud.get("type_text"):
                        cloud_text += f" ({cloud['type_text']})"
                        if cloud.get("type") == "CB":
                            has_cb = True
                        elif cloud.get("type") == "TCU":
                            has_tcu = True
                    
                    cloud_parts.append(cloud_text)
            
            if cloud_parts:
                cloud_part = f"Clouds: {', '.join(cloud_parts)}"
                
                # Add operational notes about clouds
                if has_cb:
                    cloud_part += ". Embedded thunderstorms, severe turbulence possible"
                elif has_tcu:
                    cloud_part += ". Building cumulus, potential for moderate turbulence"
                
                parts.append(cloud_part)
            
            # Add specific ceiling information if it wasn't mentioned in clouds
            if lowest_ceiling is not None and lowest_ceiling < 3000:
                parts.append(f"Ceiling: {lowest_ceiling} feet AGL")
        
        # Add temperature and dewpoint with operational impact
        if self.parsed_metar.get("temperature") is not None and self.parsed_metar.get("dewpoint") is not None:
            temp = self.parsed_metar["temperature"]
            dew = self.parsed_metar["dewpoint"]
            
            temp_part = f"Temperature {temp}°C, dewpoint {dew}°C"
            
            # Calculate temperature-dewpoint spread and add operational context
            spread = temp - dew
            
            if spread <= 2 and temp > 0:
                temp_part += f". Spread of {spread}°C indicates high humidity, fog formation possible"
            elif spread <= 3 and temp <= 0:
                temp_part += f". Narrow spread of {spread}°C with freezing temperatures, icing conditions likely"
            elif temp < 0:
                temp_part += ". Below freezing temperatures, watch for ice accumulation"
            elif temp > 30:
                temp_part += ". High temperature may affect aircraft performance"
            
            parts.append(temp_part)
        
        # Add altimeter with operational note
        if self.parsed_metar.get("altimeter") and "value" in self.parsed_metar["altimeter"]:
            alt = self.parsed_metar["altimeter"]
            alt_part = f"Altimeter {alt['value']} {alt['unit']}"
            
            # Add note about pressure changes if relevant
            if alt['unit'] == "inHg" and (alt['value'] < 29.80 or alt['value'] > 30.20):
                if alt['value'] < 29.80:
                    alt_part += ". Low pressure system, verify altimeter setting frequently"
                else:
                    alt_part += ". High pressure system, be mindful of true altitude"
            
            parts.append(alt_part)
        
        # Put it all together
        self.parsed_metar["pilot_summary"] = ". ".join(parts) + "."
        self.pilot_summary = self.parsed_metar["pilot_summary"]
    
    def _get_runway_crosswind_note(self, wind_direction: int, wind_speed: int) -> str:
        """Generate a note about potential crosswinds based on wind direction"""
        # Only add notes for significant winds
        if wind_speed < 8:
            return ""
            
        # Simplified runway orientation estimate based on wind direction
        # This assumes the runway is oriented close to the wind direction
        # In a real system, you would use actual runway data for the airport
        runway_dir = round(wind_direction / 10) * 10
        opposite_dir = (runway_dir + 180) % 360
        
        runway = f"{runway_dir//10:02d}/{opposite_dir//10:02d}"
        
        if wind_speed > 15:
            return f"Strong winds favoring runway {runway}"
        else:
            return f"Winds favoring runway {runway}"

# Function for external use
def parse_metar(metar_string: str) -> Dict[str, Any]:
    """Parse a METAR string and return structured data with a pilot-friendly summary
    
    Args:
        metar_string: The raw METAR string to parse
        
    Returns:
        Dictionary with parsed METAR data and a pilot-friendly summary
    """
    parser = MetarParser(metar_string)
    return parser.parse()