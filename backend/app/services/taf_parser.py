"""
TAF Parser - Parses raw TAF strings and generates pilot-friendly summaries

This module provides comprehensive parsing of TAF (Terminal Aerodrome Forecast) strings and
produces structured data and human-readable summaries suitable for pilot briefings.
"""
import re
import logging
from datetime import datetime, timedelta, timezone
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

# Change indicators
CHANGE_INDICATORS = {
    'BECMG': 'Becoming',
    'TEMPO': 'Temporarily',
    'PROB': 'Probability',
    'PROB30': '30% Probability',
    'PROB40': '40% Probability',
    'PROB30 TEMPO': '30% Probability Temporarily',
    'PROB40 TEMPO': '40% Probability Temporarily',
    'FM': 'From'
}

# Flight rules criteria
def determine_flight_rules(visibility_sm: Optional[float], ceiling_ft: Optional[int]) -> str:
    """Determine flight rules category based on visibility and ceiling"""
    # Check for Low IFR - either visibility < 1 SM or ceiling < 500 ft
    if ((visibility_sm is not None and visibility_sm < 1) or 
        (ceiling_ft is not None and ceiling_ft < 500)):
        return "LIFR"  # Low IFR
    
    # Check for IFR - either visibility < 3 SM or ceiling < 1000 ft
    elif ((visibility_sm is not None and visibility_sm < 3) or 
          (ceiling_ft is not None and ceiling_ft < 1000)):
        return "IFR"  # IFR
    
    # Check for MVFR - either visibility < 5 SM or ceiling < 3000 ft
    elif ((visibility_sm is not None and visibility_sm < 5) or 
          (ceiling_ft is not None and ceiling_ft < 3000)):
        return "MVFR"  # Marginal VFR
    
    # Otherwise, it's VFR
    else:
        return "VFR"  # VFR


class TafParser:
    """Parser for TAF (Terminal Aerodrome Forecast) strings"""
    
    def __init__(self, taf_string: str):
        """Initialize the parser with a TAF string"""
        self.raw = taf_string.strip()
        self.lines = self._preprocess_taf(self.raw)
        self.parsed_taf = {}
        self.forecast_periods = []
        self.pilot_summary = ""
        
    def _preprocess_taf(self, taf_string: str) -> List[str]:
        """Preprocess TAF string to handle line breaks and create a list of lines"""
        # Replace any sequence of whitespace with a single space
        clean_taf = re.sub(r'\s+', ' ', taf_string)
        
        # Split the TAF into lines based on change indicators
        lines = []
        current_line = ""
        
        for part in clean_taf.split():
            # Check if this part is a change indicator or FM (FROM) group
            if (part in ['BECMG', 'TEMPO'] or part.startswith('PROB') or 
                (part.startswith('FM') and len(part) >= 6 and part[2:].isdigit())):
                if current_line:
                    lines.append(current_line.strip())
                current_line = part + " "
            elif not lines and not current_line:  # First part of the TAF
                current_line = part + " "
            else:
                current_line += part + " "
                
        # Add the last line if it's not empty
        if current_line:
            lines.append(current_line.strip())
            
        # If no lines were created, return the entire TAF as a single line
        if not lines:
            return [clean_taf]
            
        return lines
        
    def parse(self) -> Dict[str, Any]:
        """Parse the TAF string and return structured data"""
        if not self.raw:
            return {}
            
        try:
            # Initialize parsed data structure
            self.parsed_taf = {
                "raw_text": self.raw,
                "station": None,
                "issue_time": None,
                "valid_period": {
                    "from": None,
                    "to": None
                },
                "forecast_periods": [],
                "remarks": None
            }
            
            # Parse the first line (main TAF line)
            self._parse_taf_header(self.lines[0])
            
            # Parse each forecast period
            self._parse_forecast_periods()
            
            # Calculate flight rules for each period
            self._calculate_flight_rules()
            
            # Generate detailed summaries for each period
            self._generate_period_summaries()
            
            # Generate overall TAF summary
            self._generate_taf_summary()
            
            return self.parsed_taf
            
        except Exception as e:
            logger.error(f"Error parsing TAF: {str(e)}")
            return {"raw_text": self.raw, "error": str(e)}
    
    def _parse_taf_header(self, header_line: str) -> None:
        """Parse the main TAF header line"""
        parts = header_line.split()
        
        # Parse station (usually the second element, after 'TAF')
        station_idx = 0
        if parts[0].upper() == 'TAF':
            station_idx = 1
        
        if station_idx < len(parts):
            self.parsed_taf["station"] = parts[station_idx]
        
        # Parse the date/time (format: DDHHMMz)
        for i, part in enumerate(parts):
            if re.match(r'\d{6}Z', part.upper()):
                self._parse_issue_time(part.upper())
                break
        
        # Parse validity period (format: DDDD/DDDD)
        for i, part in enumerate(parts):
            if re.match(r'\d{4}/\d{4}', part):
                self._parse_valid_period(part)
                break
    
    def _parse_issue_time(self, time_str: str) -> None:
        """Parse the issue time of the TAF"""
        if len(time_str) < 7:  # Format: DDHHMMZ
            return
            
        try:
            day = int(time_str[0:2])
            hour = int(time_str[2:4])
            minute = int(time_str[4:6])
            
            # Create a datetime object (use current year/month)
            now = datetime.now(timezone.utc)
            issue_time = datetime(now.year, now.month, day, hour, minute, tzinfo=timezone.utc)
            
            # If the date is in the future, it's likely from the previous month
            if issue_time > now and (issue_time - now).days > 15:
                if issue_time.month == 1:
                    issue_time = issue_time.replace(year=now.year - 1, month=12)
                else:
                    issue_time = issue_time.replace(month=issue_time.month - 1)
            
            self.parsed_taf["issue_time"] = {
                "raw": time_str,
                "day": day,
                "hour": hour,
                "minute": minute,
                "datetime": issue_time,
                "iso": issue_time.isoformat()
            }
        except ValueError:
            self.parsed_taf["issue_time"] = {"raw": time_str}
    
    def _parse_valid_period(self, period_str: str) -> None:
        """Parse the validity period of the TAF (format: DDDD/DDDD)"""
        parts = period_str.split('/')
        if len(parts) != 2:
            return
            
        try:
            from_day = int(parts[0][0:2])
            from_hour = int(parts[0][2:4])
            
            to_day = int(parts[1][0:2])
            to_hour = int(parts[1][2:4])
            
            # Create datetime objects (use current year/month)
            now = datetime.now(timezone.utc)
            valid_from = datetime(now.year, now.month, from_day, from_hour, 0, tzinfo=timezone.utc)
            valid_to = datetime(now.year, now.month, to_day, to_hour, 0, tzinfo=timezone.utc)
            
            # Handle month transitions
            if to_day < from_day:  # Period crosses a month boundary
                if valid_from.month == 12:
                    valid_to = valid_to.replace(year=valid_from.year + 1, month=1)
                else:
                    valid_to = valid_to.replace(month=valid_from.month + 1)
            
            self.parsed_taf["valid_period"] = {
                "from": {
                    "raw": parts[0],
                    "day": from_day,
                    "hour": from_hour,
                    "datetime": valid_from,
                    "iso": valid_from.isoformat()
                },
                "to": {
                    "raw": parts[1],
                    "day": to_day,
                    "hour": to_hour,
                    "datetime": valid_to,
                    "iso": valid_to.isoformat()
                }
            }
        except ValueError:
            self.parsed_taf["valid_period"] = {
                "from": {"raw": parts[0] if len(parts) > 0 else None},
                "to": {"raw": parts[1] if len(parts) > 1 else None}
            }
    
    def _parse_forecast_periods(self) -> None:
        """Parse all forecast periods in the TAF"""
        main_conditions = None
        
        # First, parse the main conditions (first line after the header)
        if len(self.lines) > 1:
            main_conditions = self._parse_weather_conditions(self.lines[1])
            main_conditions["type"] = "Base Forecast"
            
            # Fix: Add null checks when accessing nested properties
            if self.parsed_taf.get("valid_period") and self.parsed_taf["valid_period"].get("from"):
                valid_from = self.parsed_taf["valid_period"]["from"].get("datetime")
                if valid_from:
                    main_conditions["valid_from"] = valid_from
                
            if self.parsed_taf.get("valid_period") and self.parsed_taf["valid_period"].get("to"):
                valid_to = self.parsed_taf["valid_period"]["to"].get("datetime")
                if valid_to:
                    main_conditions["valid_to"] = valid_to
            
            self.parsed_taf["forecast_periods"].append(main_conditions)
        
        # Parse change periods
        last_valid_from = None
        last_valid_to = None
        
        if self.parsed_taf.get("valid_period"):
            if self.parsed_taf["valid_period"].get("from"):
                last_valid_from = self.parsed_taf["valid_period"]["from"].get("datetime")
            if self.parsed_taf["valid_period"].get("to"):
                last_valid_to = self.parsed_taf["valid_period"]["to"].get("datetime")
        
        for i, line in enumerate(self.lines[2:], start=2):
            change_type, period = self._parse_change_indicator(line)
            if not change_type:
                continue
                
            conditions = self._parse_weather_conditions(line)
            conditions["type"] = change_type
            
            # Handle FM (From) groups specifically
            if change_type.startswith("From"):
                # Extract the time from the FM indicator (format: FMDDHHMM)
                fm_match = re.search(r'FM(\d{2})(\d{2})(\d{2})', line)
                if fm_match:
                    day = int(fm_match.group(1))
                    hour = int(fm_match.group(2))
                    minute = int(fm_match.group(3))
                    
                    now = datetime.now(timezone.utc)
                    from_time = datetime(now.year, now.month, day, hour, minute, tzinfo=timezone.utc)
                    
                    # Handle month transitions
                    if last_valid_from and day < last_valid_from.day:
                        if from_time.month == 12:
                            from_time = from_time.replace(year=from_time.year + 1, month=1)
                        else:
                            from_time = from_time.replace(month=from_time.month + 1)
                    
                    conditions["valid_from"] = from_time
                    if last_valid_to:
                        conditions["valid_to"] = last_valid_to
                    
                    last_valid_from = from_time
            
            # Handle other change groups with time periods
            elif period and len(period) == 2:
                from_time, to_time = period
                if from_time and to_time:
                    conditions["valid_from"] = from_time
                    conditions["valid_to"] = to_time
            
            self.parsed_taf["forecast_periods"].append(conditions)
    
    def _parse_change_indicator(self, line: str) -> Tuple[Optional[str], Optional[Tuple[datetime, datetime]]]:
        """Parse change indicator and period from a line"""
        parts = line.split()
        if not parts:
            return None, None
            
        # Check for FROM groups (format: FMDDHHMM)
        if parts[0].startswith('FM') and len(parts[0]) >= 9:
            return "From", None
            
        # Check for other change indicators
        change_type = None
        for indicator in CHANGE_INDICATORS:
            if parts[0] == indicator or (len(parts) > 1 and f"{parts[0]} {parts[1]}" == indicator):
                change_type = CHANGE_INDICATORS[indicator]
                break
                
        if not change_type:
            return None, None
            
        # Look for period indicator (format: DDHH/DDHH) 
        period = None
        for part in parts:
            if re.match(r'\d{4}/\d{4}', part):
                period_parts = part.split('/')
                if len(period_parts) == 2:
                    try:
                        from_day = int(period_parts[0][:2])
                        from_hour = int(period_parts[0][2:])
                        to_day = int(period_parts[1][:2])
                        to_hour = int(period_parts[1][2:])
                        
                        now = datetime.now(timezone.utc)
                        from_time = datetime(now.year, now.month, from_day, from_hour, 0, tzinfo=timezone.utc)
                        to_time = datetime(now.year, now.month, to_day, to_hour, 0, tzinfo=timezone.utc)
                        
                        # Handle month transitions
                        if to_day < from_day:
                            if to_time.month == 12:
                                to_time = to_time.replace(year=to_time.year + 1, month=1)
                            else:
                                to_time = to_time.replace(month=to_time.month + 1)
                        
                        period = (from_time, to_time)
                    except ValueError:
                        pass
                break
                
        return change_type, period
    
    def _parse_weather_conditions(self, line: str) -> Dict[str, Any]:
        """Parse weather conditions from a forecast line"""
        conditions = {
            "raw": line,
            "wind": None,
            "visibility": None,
            "weather": [],
            "clouds": [],
            "ceiling": None,
            "icing": None,
            "turbulence": None,
            "wind_shear": None
        }
        
        parts = line.split()
        current_part = 0
        
        # Skip change indicators if present
        if parts and (parts[0].startswith('FM') or parts[0] in CHANGE_INDICATORS or 
                     (len(parts) > 1 and f"{parts[0]} {parts[1]}" in CHANGE_INDICATORS)):
            current_part = 1
            if len(parts) > 1 and parts[0].startswith('PROB') and parts[1] == 'TEMPO':
                current_part = 2
            
            # Skip time group if present after change indicator
            if current_part < len(parts) and re.match(r'\d{4}/\d{4}', parts[current_part]):
                current_part += 1
        
        while current_part < len(parts):
            # Parse wind
            if (current_part < len(parts) and (parts[current_part].endswith('KT') or 
                                              parts[current_part].endswith('MPS'))):
                conditions["wind"] = self._parse_wind(parts[current_part])
                current_part += 1
                continue
                
            # Parse visibility
            if current_part < len(parts):
                vis_str = parts[current_part]
                
                # Handle different visibility formats
                if vis_str == "CAVOK" or vis_str == "SKC":
                    conditions["visibility"] = {
                        "distance": 10,
                        "unit": "SM",
                        "text": "10+ statute miles",
                        "cavok": True
                    }
                    current_part += 1
                    continue
                    
                if vis_str == "P6SM":
                    conditions["visibility"] = {
                        "distance": 6,
                        "unit": "SM",
                        "text": "Greater than 6 statute miles",
                        "greater_than": True
                    }
                    current_part += 1
                    continue
                
                # Handle vis like "4SM" or "3/4SM"
                if vis_str.endswith("SM"):
                    if "/" in vis_str:
                        # Fractional visibility
                        vis_text = vis_str.replace("SM", "")
                        try:
                            num, denom = vis_text.split('/')
                            vis_value = float(num) / float(denom)
                            conditions["visibility"] = {
                                "distance": vis_value,
                                "unit": "SM",
                                "text": f"{vis_text} statute miles"
                            }
                            current_part += 1
                            continue
                        except (ValueError, ZeroDivisionError):
                            pass
                    else:
                        # Integer or decimal visibility
                        vis_text = vis_str.replace("SM", "")
                        try:
                            vis_value = float(vis_text)
                            conditions["visibility"] = {
                                "distance": vis_value,
                                "unit": "SM",
                                "text": f"{vis_value} statute miles"
                            }
                            current_part += 1
                            continue
                        except ValueError:
                            pass
            
            # Parse weather phenomena
            if current_part < len(parts):
                wx_str = parts[current_part]
                
                # Check for special cases
                if wx_str == "NSW":
                    conditions["weather"].append({
                        "raw": "NSW",
                        "text": "No Significant Weather"
                    })
                    current_part += 1
                    continue
                
                # Check if it's a weather phenomenon
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
                
                if is_weather:
                    # Parse the weather phenomenon
                    intensity = ""
                    descriptions = []
                    
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
                    while i < len(wx_str):
                        if i + 2 <= len(wx_str):
                            code = wx_str[i:i+2]
                            if code in WEATHER_PHENOMENA:
                                descriptions.append(WEATHER_PHENOMENA[code])
                        i += 2
                    
                    if descriptions:
                        weather_info = {
                            "raw": parts[current_part],
                            "intensity": intensity,
                            "descriptions": descriptions,
                            "text": (intensity + " " if intensity else "") + " ".join(descriptions)
                        }
                        conditions["weather"].append(weather_info)
                    
                    current_part += 1
                    continue
            
            # Parse clouds
            if current_part < len(parts):
                cloud_str = parts[current_part]
                
                # Check for recognized cloud patterns
                cloud_match = re.match(r'^(VV|FEW|SCT|BKN|OVC)(\d{3})(CB|TCU)?$', cloud_str)
                special_condition = cloud_str in ["SKC", "CLR", "NSC", "NCD", "CAVOK"]
                
                if cloud_match or special_condition:
                    if special_condition:
                        cloud_info = {
                            "cover": cloud_str,
                            "cover_text": CLOUD_COVER_CODES.get(cloud_str, cloud_str),
                            "base": None,
                            "type": None,
                            "ceiling": False
                        }
                        conditions["clouds"].append(cloud_info)
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
                        conditions["clouds"].append(cloud_info)
                        
                        # Track ceiling (lowest broken or overcast layer)
                        if is_ceiling and (conditions["ceiling"] is None or height < conditions["ceiling"]):
                            conditions["ceiling"] = height
                    
                    current_part += 1
                    continue
            
            # No match, move to next part
            current_part += 1
        
        return conditions
    
    def _parse_wind(self, wind_str: str) -> Dict[str, Any]:
        """Parse a wind string (e.g., 27015G25KT)"""
        # Handle calm winds
        if wind_str == '00000KT':
            return {
                "direction": 0,
                "speed": 0,
                "gust": None,
                "unit": "KT",
                "cardinal": "N",
                "text": "Calm"
            }
        
        # Handle variable winds
        if wind_str.startswith('VRB'):
            speed_part = re.search(r'VRB(\d+)(G(\d+))?KT', wind_str)
            if speed_part:
                speed = int(speed_part.group(1))
                gust = int(speed_part.group(3)) if speed_part.group(3) else None
                return {
                    "direction": "VRB",
                    "speed": speed,
                    "gust": gust,
                    "unit": "KT",
                    "variable": True,
                    "text": f"Variable at {speed} knots{' gusting to ' + str(gust) + ' knots' if gust else ''}"
                }
            return None
        
        # Regular wind pattern
        wind_match = re.match(r'^(\d{3})(\d{2,3})(G(\d{2,3}))?(?:KT|MPS)$', wind_str)
        if wind_match:
            direction = int(wind_match.group(1))
            speed = int(wind_match.group(2))
            gust = int(wind_match.group(4)) if wind_match.group(4) else None
            unit = "KT" if wind_str.endswith("KT") else "MPS"
            
            # Convert direction to cardinal
            cardinal = self._get_cardinal_direction(direction)
            
            return {
                "direction": direction,
                "speed": speed,
                "gust": gust,
                "unit": unit,
                "cardinal": cardinal,
                "variable": False,
                "text": f"From {direction}° ({cardinal}) at {speed} {unit}{' gusting to ' + str(gust) + ' ' + unit if gust else ''}"
            }
        
        return None
    
    def _get_cardinal_direction(self, degrees: int) -> str:
        """Convert wind direction in degrees to cardinal direction"""
        index = round(degrees / 22.5) % 16
        return CARDINAL_DIRECTIONS[index]
    
    def _calculate_flight_rules(self) -> None:
        """Calculate the flight rules for each forecast period"""
        for period in self.parsed_taf["forecast_periods"]:
            # Get visibility in SM
            visibility_sm = None
            if period.get("visibility"):
                if period["visibility"].get("unit") == "SM":
                    visibility_sm = period["visibility"].get("distance")
                elif period["visibility"].get("unit") == "M":
                    # Convert meters to statute miles
                    meters = period["visibility"].get("distance", 0)
                    if meters is not None:  # Add null check
                        visibility_sm = meters / 1609.34
            
            # Get ceiling in feet
            ceiling_ft = period.get("ceiling")
            
            # Determine flight rules
            period["flight_rules"] = determine_flight_rules(visibility_sm, ceiling_ft)
    
    def _generate_period_summaries(self) -> None:
        """Generate detailed summaries for each forecast period"""
        if not self.parsed_taf["forecast_periods"]:
            return
            
        # Add forecast-specific insights for each period
        for i, period in enumerate(self.parsed_taf["forecast_periods"]):
            period_notes = []
            
            # Add time range
            if period.get("valid_from") and period.get("valid_to"):
                from_time = period["valid_from"]
                to_time = period["valid_to"]
                
                # Format times for display
                from_str = from_time.strftime("%d/%H:%MZ")
                to_str = to_time.strftime("%d/%H:%MZ")
                
                period_notes.append(f"Valid from {from_str} until {to_str}")
            
            # Add change type if any
            if period.get("type") and period["type"] != "Base Forecast":
                period_notes.append(f"{period['type']} conditions")
                
            # Add flight category
            if period.get("flight_rules"):
                category = period["flight_rules"]
                category_descriptions = {
                    "VFR": "Visual Flight Rules",
                    "MVFR": "Marginal Visual Flight Rules",
                    "IFR": "Instrument Flight Rules",
                    "LIFR": "Low Instrument Flight Rules"
                }
                description = category_descriptions.get(category, category)
                period_notes.append(f"{category} ({description})")
            
            # Add wind information with operational impact
            if period.get("wind"):
                wind = period["wind"]
                wind_text = f"Wind {wind.get('text', '')}"
                
                # Add note about gusty conditions if present
                if wind.get("gust") and wind.get("gust") > 10:
                    gust_factor = wind.get("gust", 0) - wind.get("speed", 0)
                    if gust_factor > 10:
                        wind_text += f". Significant wind shear possible with {gust_factor} knot gust factor"
                    elif gust_factor > 5:
                        wind_text += f". Prepare for {gust_factor} knot gusts on approach/departure"
                
                # Fix: Add null check for wind direction
                if wind.get("direction") != "VRB" and isinstance(wind.get("direction"), int) and wind.get("speed", 0) > 8:
                    wind_text += ". Possible crosswind considerations for takeoff and landing"
                
                period_notes.append(wind_text)
            
            # Add visibility with operational context
            if period.get("visibility") and "text" in period["visibility"]:
                vis_text = f"Visibility {period['visibility']['text']}"
                
                # Add operational context based on visibility distance
                if "distance" in period["visibility"]:
                    vis_distance = period["visibility"]["distance"]
                    if vis_distance < 1:
                        vis_text += ". Approach and landing will require precision instruments"
                    elif vis_distance < 3:
                        vis_text += ". Plan for instrument approach procedures"
                    elif vis_distance < 5:
                        vis_text += ". Visual approach possible with reduced references"
                
                period_notes.append(vis_text)
            
            # Add weather phenomena with operational impact
            if period.get("weather") and len(period["weather"]) > 0:
                weather_texts = []
                has_thunderstorm = False
                has_freezing = False
                has_rain = False
                has_snow = False
                has_fog = False
                
                for w in period["weather"]:
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
                    if has_rain:
                        operational_notes.append("Wet runway conditions")
                    if has_snow:
                        operational_notes.append("Possible runway contamination and braking issues")
                    if has_fog and period.get("visibility", {}).get("distance", 10) < 3:
                        operational_notes.append("Reduced visual references")
                    
                    if operational_notes:
                        weather_part += f". {' and '.join(operational_notes)}"
                    
                    period_notes.append(weather_part)
            elif any(w.get("raw") == "NSW" for w in period.get("weather", [])):
                period_notes.append("No significant weather expected")
            
            # Add cloud information with operational impact
            if period.get("clouds") and len(period["clouds"]) > 0:
                cloud_parts = []
                has_cb = False
                has_tcu = False
                
                # Handle special sky conditions
                if len(period["clouds"]) == 1 and period["clouds"][0].get("cover") in ["SKC", "CLR", "NSC", "NCD", "CAVOK"]:
                    cover_text = period["clouds"][0].get("cover_text", period["clouds"][0].get("cover"))
                    cloud_parts.append(cover_text)
                else:
                    for cloud in period["clouds"]:
                        cloud_text = cloud.get("cover_text", cloud.get("cover", ""))
                        
                        if cloud.get("base") is not None:
                            cloud_text += f" at {cloud['base']} feet"
                        
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
                    
                    # Add ceiling information if applicable
                    if period.get("ceiling") is not None:
                        if period["ceiling"] < 1000:
                            cloud_part += f". Low {period['ceiling']} foot ceiling may require instrument approach"
                        elif period["ceiling"] < 3000:
                            cloud_part += f". Ceiling of {period['ceiling']} feet - prepare for potential IFR conditions"
                    
                    period_notes.append(cloud_part)
            
            # Create a concise summary of the period's forecast
            period["pilot_summary"] = ". ".join(period_notes) + "."
            
            # Add planning considerations based on the overall period
            flight_rules = period.get("flight_rules", "")
            planning = []
            
            if flight_rules in ["LIFR", "IFR"]:
                planning.append("File IFR flight plan")
                planning.append("Ensure aircraft & pilot are IFR current/qualified")
                planning.append("Verify approach minimums for destination")
                planning.append("Consider alternate requirements")
            
            if any(w.get("text", "").lower().find("thunderstorm") >= 0 for w in period.get("weather", [])):
                planning.append("Check radar frequently")
                planning.append("Plan routes to avoid convective activity")
            
            if any(w.get("text", "").lower().find("freezing") >= 0 for w in period.get("weather", [])) or any(w.get("text", "").lower().find("ice") >= 0 for w in period.get("weather", [])):
                planning.append("Verify aircraft has appropriate ice protection equipment")
                planning.append("Plan altitudes to minimize icing exposure")
            
            wind = period.get("wind", {})
            if wind.get("speed", 0) > 20 or wind.get("gust", 0) > 25:
                planning.append("Be prepared for challenging takeoff/landing conditions")
                planning.append("Review aircraft's crosswind limitations")
            
            if planning:
                period["planning_considerations"] = planning
                
                # Add a brief note of planning considerations to the summary
                if len(planning) > 0:
                    period["pilot_summary"] += f" Planning considerations: {planning[0]}"
                    if len(planning) > 1:
                        period["pilot_summary"] += f" and {len(planning)-1} other item(s)"
    
    def _generate_taf_summary(self) -> None:
        """Generate an overall summary of the TAF"""
        summary_parts = []
        
        # Add station and validity period
        station = self.parsed_taf.get("station", "")
        valid_period = self.parsed_taf.get("valid_period", {})
        
        if station:
            opening = f"TAF for {station}"
            
            if valid_period.get("from", {}).get("datetime") and valid_period.get("to", {}).get("datetime"):
                from_time = valid_period["from"]["datetime"]
                to_time = valid_period["to"]["datetime"]
                
                # Format the times
                from_str = from_time.strftime("%d%HZ")
                to_str = to_time.strftime("%d%HZ")
                
                opening += f" valid from {from_str} to {to_str}"
            
            summary_parts.append(opening)
        
        # Analyze the entire forecast for key weather changes
        if self.parsed_taf["forecast_periods"]:
            # Check for significant condition changes
            min_vis = float('inf')
            min_ceiling = float('inf')
            worst_category = "VFR"
            categories_order = ["VFR", "MVFR", "IFR", "LIFR"]
            has_thunderstorm = False
            has_freezing_precip = False
            has_snow = False
            has_strong_winds = False
            max_wind_speed = 0
            
            for period in self.parsed_taf["forecast_periods"]:
                # Track worst visibility
                if period.get("visibility", {}).get("distance") is not None:
                    min_vis = min(min_vis, period["visibility"]["distance"])
                
                # Track worst ceiling
                if period.get("ceiling") is not None:
                    min_ceiling = min(min_ceiling, period["ceiling"])
                
                # Track worst flight category
                if period.get("flight_rules") in categories_order:
                    current_index = categories_order.index(period["flight_rules"])
                    worst_index = categories_order.index(worst_category)
                    if current_index > worst_index:
                        worst_category = period["flight_rules"]
                
                # Check for thunderstorms
                for wx in period.get("weather", []):
                    wx_text = wx.get("text", "").lower()
                    if "thunderstorm" in wx_text or "ts" in wx.get("raw", "").lower():
                        has_thunderstorm = True
                    if "freezing" in wx_text or "fz" in wx.get("raw", "").lower():
                        has_freezing_precip = True
                    if "snow" in wx_text or "sn" in wx.get("raw", "").lower():
                        has_snow = True
                
                # Check for strong winds
                if period.get("wind", {}).get("speed", 0) > max_wind_speed:
                    max_wind_speed = period["wind"]["speed"]
                if period.get("wind", {}).get("speed", 0) > 20 or period.get("wind", {}).get("gust", 0) > 25:
                    has_strong_winds = True
            
            # Summarize key conditions
            conditions = []
            
            # Add flight rules summary
            if worst_category != "VFR":
                if worst_category == "LIFR":
                    conditions.append("Expecting Low IFR conditions")
                elif worst_category == "IFR":
                    conditions.append("Expecting IFR conditions")
                elif worst_category == "MVFR":
                    conditions.append("Expecting Marginal VFR conditions")
            
            # Add visibility and ceiling if significant
            if min_vis != float('inf') and min_vis < 5:
                conditions.append(f"Lowest visibility {min_vis} SM")
            
            if min_ceiling != float('inf') and min_ceiling < 3000:
                conditions.append(f"Lowest ceiling {min_ceiling} feet")
            
            # Add significant weather
            if has_thunderstorm:
                conditions.append("Thunderstorms forecasted")
            
            if has_freezing_precip:
                conditions.append("Freezing precipitation expected")
            
            if has_snow:
                conditions.append("Snow expected")
            
            if has_strong_winds:
                conditions.append(f"Strong winds up to {max_wind_speed} knots")
            
            # Add the conditions to the summary
            if conditions:
                summary_parts.append(". ".join(conditions))
            else:
                summary_parts.append("Generally favorable flying conditions expected")
        
        self.pilot_summary = ". ".join(summary_parts) + "."
        self.parsed_taf["pilot_summary"] = self.pilot_summary


# Function for external use
def parse_taf(taf_string: str) -> Dict[str, Any]:
    """Parse a TAF string and return structured data with pilot-friendly summaries
    
    Args:
        taf_string: The raw TAF string to parse
        
    Returns:
        Dictionary with parsed TAF data and pilot-friendly summaries
    """
    parser = TafParser(taf_string)
    return parser.parse()