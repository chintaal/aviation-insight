"""
OpenAI Service for generating pilot-friendly weather report summaries
"""
import logging
import asyncio
import os
import json
from typing import Dict, Any, Optional, Union, List
from openai import AsyncOpenAI
from ..core.config import settings

logger = logging.getLogger(__name__)

class OpenAISummaryService:
    """Service for generating summaries of weather reports using OpenAI GPT models"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Use the directly provided key or the one from environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        if not self.api_key or self.api_key.startswith("sk-your-"):
            # Fallback to hardcoded key if needed
            self.api_key = ""
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        # Use GPT-4o for more comprehensive and accurate summaries
        self.model = "gpt-4o"  
    
    async def generate_summary(self, report_type: str, report_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate a pilot-friendly summary of a weather report
        
        Args:
            report_type: Type of report ('metar', 'taf', 'pirep', 'sigmet')
            report_data: Report data in dictionary format
            
        Returns:
            A pilot-friendly summary of the report, or None if generation failed
        """
        if not self.api_key:
            logger.warning("OpenAI API key not configured, cannot generate summary")
            return None
            
        try:
            prompt = self._create_prompt_for_report(report_type, report_data)
            
            # Log useful information for debugging
            logger.info(f"Generating summary for {report_type} using model {self.model}")
            
            # Call OpenAI API to generate summary
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(report_type)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent, factual responses
                max_tokens=600,   # Increased token length for more detailed summaries
            )
            
            if response and response.choices and len(response.choices) > 0:
                summary = response.choices[0].message.content.strip()
                logger.info(f"Generated {report_type.upper()} summary successfully")
                return summary
            else:
                logger.warning(f"No content returned from OpenAI for {report_type.upper()} summary")
                return None
                
        except Exception as e:
            logger.error(f"Error generating {report_type.upper()} summary: {str(e)}")
            # Return a fallback summary if API fails
            return self._generate_fallback_summary(report_type, report_data)
    
    def _generate_fallback_summary(self, report_type: str, report_data: Dict[str, Any]) -> str:
        """Generate a basic fallback summary when OpenAI API fails"""
        if report_type == "metar":
            return f"METAR for {report_data.get('station', 'unknown station')}.\n\nThis is automated weather data. Check the raw report for details.\n\nExercise caution and verify conditions before flight."
        
        elif report_type == "taf":
            return f"TAF forecast for {report_data.get('station', 'unknown station')}.\n\nConsult the raw forecast for detailed weather predictions.\n\nPlan your flight carefully considering all available information."
        
        elif report_type == "pirep":
            return f"Pilot report near {report_data.get('location', 'unknown location')}.\n\nReview the raw report for specific conditions reported.\n\nConsider these pilot observations in your flight planning."
        
        else:
            return f"Weather information available.\n\nRefer to the raw data for complete details.\n\nEnsure thorough preflight planning."
    
    def _get_system_prompt(self, report_type: str) -> str:
        """Get the system prompt for a specific report type"""
        base_prompt = "You are an expert aviation weather briefing assistant providing detailed, accurate summaries for pilots. Your summaries should be comprehensive yet clear, focusing on operational impact and flight safety. "
        
        if report_type == "metar":
            return base_prompt + "Analyze and summarize the METAR in plain language with a focus on flight safety. Provide a detailed assessment of ceiling, visibility, winds, pressure, and significant weather phenomena. Include implications for VFR/IFR operations and mention any concerning trends if apparent."
            
        elif report_type == "taf":
            return base_prompt + "Analyze the TAF forecast in detail, highlighting all operationally significant changes in weather conditions over the forecast period. Break down the forecast into clear time segments, focusing on changing IFR/VFR conditions, wind shifts, and hazardous weather. Include practical recommendations for flight planning."
            
        elif report_type == "pirep":
            return base_prompt + "Provide a comprehensive analysis of this pilot report focusing on turbulence, icing, cloud tops, and other flight safety hazards. Be specific about altitude-dependent conditions, severity of hazards, and potential impact on different aircraft types. Include practical avoidance strategies when appropriate."
            
        elif report_type == "sigmet":
            return base_prompt + "Thoroughly analyze this SIGMET emphasizing the hazard, affected area, altitudes, timing, and movement. Provide clear details about the safety implications for flights in or near the affected area, and suggest potential mitigation strategies."
            
        return base_prompt + "Provide a thorough and detailed analysis of this aviation weather information focusing on all flight safety implications and operational considerations."
    
    def _create_prompt_for_report(self, report_type: str, report_data: Dict[str, Any]) -> str:
        """Create a specific prompt based on the report type and data"""
        if report_type == "metar":
            return f"""Create a detailed, pilot-friendly analysis of this METAR for {report_data.get('station', 'unknown station')}:
Raw METAR: {report_data.get('raw_text', 'No raw data available')}

Include the following in your summary:
1. Flight category (VFR/MVFR/IFR/LIFR) with clear explanation of the determining factors
2. Ceiling and visibility in plain language with operational impact
3. Detailed wind conditions including gusts and crosswind components if significant
4. All precipitation and weather phenomena with severity and implications
5. Temperature/dewpoint analysis including potential for icing or fog formation
6. Pressure trends and their significance
7. Any specific hazards or concerns evident from the report

Format your response with clear sections and conclude with specific operational recommendations."""

        elif report_type == "taf":
            return f"""Create a detailed, pilot-friendly analysis of this TAF forecast for {report_data.get('station', 'unknown station')}:
Raw TAF: {report_data.get('raw_text', 'No raw data available')}

Include the following in your analysis:
1. Overall summary of weather evolution during the forecast period
2. Detailed breakdown of each significant time period in chronological order
3. Clear identification of all IFR or MVFR conditions with timing and duration
4. Comprehensive wind analysis including direction shifts and gusting conditions
5. Detailed description of all forecast weather phenomena and their intensity
6. Identification of the most challenging period(s) during the forecast
7. Specific operational considerations for takeoff, en route, and landing phases

Structure your response with clearly organized sections by time period, and conclude with practical flight planning recommendations."""

        elif report_type == "pirep":
            return f"""Create a detailed, pilot-friendly analysis of this Pilot Report:
Raw PIREP: {report_data.get('raw_text', 'No raw data available')}

Include the following in your analysis:
1. Aircraft type, precise location, and altitude of the report
2. Detailed assessment of turbulence including type, intensity, and vertical extent
3. Comprehensive icing information including type, severity, and altitude layer
4. Thorough cloud information including bases, tops, layers, and coverage
5. Visibility conditions and any obscuring phenomena
6. Time context of the report and its current relevance
7. Correlation with forecast conditions if apparent

Format your response with clear sections and conclude with specific operational recommendations for pilots in the area."""

        elif report_type == "sigmet":
            return f"""Create a detailed, pilot-friendly analysis of this SIGMET:
Raw SIGMET: {report_data.get('raw_text', 'No raw data available')}

Include the following in your analysis:
1. Precise identification of the hazard type and its severity
2. Detailed geographic description of the affected area with key landmarks/waypoints
3. Comprehensive altitude range information with flight level context
4. Specific validity timeframe and remaining duration
5. Movement, intensification, or dissipation trends of the hazard
6. Potential impact on different phases of flight and aircraft categories
7. Correlation with other weather data if apparent

Structure your response with clear sections and conclude with specific avoidance or mitigation strategies."""

        else:
            return f"Please provide a comprehensive analysis of this aviation weather information with detailed operational implications for pilots: {report_data}"
    
    async def generate_comprehensive_summary(self, all_reports: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a comprehensive, visual, and detailed summary of all weather reports for an airport.
        
        Args:
            all_reports: Dictionary containing METAR, TAF, PIREPs, and SIGMETs
            
        Returns:
            A comprehensive summary dictionary with overview, current conditions, forecast, hazards, and recommendations
        """
        if not self.api_key:
            logger.warning("OpenAI API key not configured, cannot generate comprehensive summary")
            return self._generate_fallback_comprehensive_summary(all_reports)
        
        try:
            station = all_reports.get("station", "unknown")
            
            # Build comprehensive prompt
            prompt = f"""Create a super visual and detailed comprehensive weather report summary for airport {station}.

METAR (Current Conditions):
{self._format_metar_for_summary(all_reports.get("metar"))}

TAF (Forecast):
{self._format_taf_for_summary(all_reports.get("taf"))}

PIREPs (Pilot Reports):
{self._format_pireps_for_summary(all_reports.get("pireps", []))}

SIGMETs (Weather Advisories):
{self._format_sigmets_for_summary(all_reports.get("sigmets", []))}

Please provide a comprehensive, visually structured summary with the following sections:

1. **Executive Overview**: A high-level summary of current conditions and key concerns
2. **Current Conditions Analysis**: Detailed breakdown of METAR with flight category, visibility, ceiling, winds, and weather phenomena
3. **Forecast Outlook**: Detailed TAF analysis with timeline of expected changes, IFR/VFR transitions, and significant weather
4. **Hazard Assessment**: Comprehensive analysis of PIREPs and SIGMETs, including turbulence, icing, thunderstorms, and other hazards
5. **Operational Recommendations**: Specific, actionable recommendations for flight planning, including:
   - Best times to fly
   - Altitude recommendations
   - Route considerations
   - Equipment requirements
   - Risk factors

Format the response as a structured JSON object with these keys:
- overview: string
- current_conditions: object with keys: flight_category, visibility, ceiling, winds, weather, temperature, pressure, summary
- forecast_outlook: object with keys: timeline, ifr_periods, significant_changes, summary
- hazards: object with keys: turbulence, icing, thunderstorms, other, summary
- recommendations: object with keys: flight_planning, timing, altitude, equipment, risk_assessment

Make the summary detailed, professional, and actionable for pilots."""
            
            logger.info(f"Generating comprehensive summary for {station} using model {self.model}")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert aviation weather briefing assistant. Provide comprehensive, detailed, and visually structured weather summaries that help pilots make informed flight planning decisions. Always prioritize safety and operational considerations."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            if response and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content.strip()
                try:
                    summary = json.loads(content)
                    logger.info(f"Generated comprehensive summary for {station} successfully")
                    return summary
                except json.JSONDecodeError:
                    # If JSON parsing fails, return as structured text
                    return {
                        "overview": content[:500],
                        "current_conditions": {"summary": content},
                        "forecast_outlook": {"summary": content},
                        "hazards": {"summary": content},
                        "recommendations": {"summary": content}
                    }
            else:
                logger.warning(f"No content returned from OpenAI for comprehensive summary")
                return self._generate_fallback_comprehensive_summary(all_reports)
                
        except Exception as e:
            logger.error(f"Error generating comprehensive summary: {str(e)}")
            return self._generate_fallback_comprehensive_summary(all_reports)
    
    def _format_metar_for_summary(self, metar: Optional[Dict[str, Any]]) -> str:
        """Format METAR data for summary prompt"""
        if not metar:
            return "No METAR data available"
        
        return f"""
Station: {metar.get('station', 'Unknown')}
Raw: {metar.get('raw_text', 'N/A')}
Flight Category: {metar.get('flight_category', 'N/A')}
Visibility: {metar.get('visibility', 'N/A')} SM
Ceiling: {metar.get('ceiling', 'N/A')} ft
Wind: {metar.get('wind_direction', 'N/A')}° at {metar.get('wind_speed', 'N/A')} kts
Temperature: {metar.get('temperature', 'N/A')}°C
Dewpoint: {metar.get('dewpoint', 'N/A')}°C
"""
    
    def _format_taf_for_summary(self, taf: Optional[Dict[str, Any]]) -> str:
        """Format TAF data for summary prompt"""
        if not taf:
            return "No TAF data available"
        
        valid_from = taf.get('valid_from', 'N/A')
        valid_to = taf.get('valid_to', 'N/A')
        
        return f"""
Station: {taf.get('station', 'Unknown')}
Raw: {taf.get('raw_text', 'N/A')}
Valid: {valid_from} to {valid_to}
Forecast Periods: {len(taf.get('forecast', []))} periods
"""
    
    def _format_pireps_for_summary(self, pireps: List[Dict[str, Any]]) -> str:
        """Format PIREP data for summary prompt"""
        if not pireps or len(pireps) == 0:
            return "No PIREP data available"
        
        formatted = f"Total PIREPs: {len(pireps)}\n"
        for i, pirep in enumerate(pireps[:10], 1):  # Limit to first 10
            formatted += f"""
PIREP {i}:
Location: {pirep.get('location', 'Unknown')}
Altitude: {pirep.get('altitude', 'N/A')}
Aircraft: {pirep.get('aircraft_type', 'N/A')}
Turbulence: {pirep.get('turbulence', {})}
Icing: {pirep.get('icing', {})}
Raw: {pirep.get('raw_text', 'N/A')[:200]}
"""
        return formatted
    
    def _format_sigmets_for_summary(self, sigmets: List[Dict[str, Any]]) -> str:
        """Format SIGMET data for summary prompt"""
        if not sigmets or len(sigmets) == 0:
            return "No SIGMET data available"
        
        formatted = f"Total SIGMETs: {len(sigmets)}\n"
        for i, sigmet in enumerate(sigmets[:10], 1):  # Limit to first 10
            formatted += f"""
SIGMET {i}:
Phenomenon: {sigmet.get('phenomenon', 'Unknown')}
Valid: {sigmet.get('valid_from', 'N/A')} to {sigmet.get('valid_to', 'N/A')}
Altitude: {sigmet.get('altitude', {})}
Raw: {sigmet.get('raw_text', 'N/A')[:200]}
"""
        return formatted
    
    def _generate_fallback_comprehensive_summary(self, all_reports: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a fallback comprehensive summary when OpenAI is unavailable"""
        station = all_reports.get("station", "unknown")
        metar = all_reports.get("metar")
        taf = all_reports.get("taf")
        pireps = all_reports.get("pireps", [])
        sigmets = all_reports.get("sigmets", [])
        
        return {
            "overview": f"Comprehensive weather summary for {station}. Review all available reports for complete information.",
            "current_conditions": {
                "flight_category": metar.get("flight_category", "Unknown") if metar else "No data",
                "visibility": metar.get("visibility", "N/A") if metar else "N/A",
                "ceiling": metar.get("ceiling", "N/A") if metar else "N/A",
                "winds": f"{metar.get('wind_direction', 'N/A')}° at {metar.get('wind_speed', 'N/A')} kts" if metar else "N/A",
                "weather": "Check METAR for details",
                "temperature": metar.get("temperature", "N/A") if metar else "N/A",
                "pressure": "Check METAR for details",
                "summary": f"Current conditions at {station}. Always verify with latest METAR before flight."
            },
            "forecast_outlook": {
                "timeline": "Check TAF for detailed forecast",
                "ifr_periods": "Review TAF for IFR conditions",
                "significant_changes": "Monitor TAF updates",
                "summary": f"Forecast available for {station}. Review TAF for detailed timeline."
            },
            "hazards": {
                "turbulence": f"{len([p for p in pireps if p.get('turbulence')])} PIREPs mention turbulence" if pireps else "No PIREP data",
                "icing": f"{len([p for p in pireps if p.get('icing')])} PIREPs mention icing" if pireps else "No PIREP data",
                "thunderstorms": f"{len([s for s in sigmets if 'thunderstorm' in str(s.get('phenomenon', '')).lower()])} SIGMETs for thunderstorms" if sigmets else "No SIGMET data",
                "other": "Review PIREPs and SIGMETs for other hazards",
                "summary": "Review all PIREPs and SIGMETs for comprehensive hazard assessment."
            },
            "recommendations": {
                "flight_planning": "Review all available weather data and consult with Flight Service",
                "timing": "Consider current conditions and forecast trends",
                "altitude": "Review PIREPs for altitude-specific conditions",
                "equipment": "Ensure appropriate equipment for forecast conditions",
                "risk_assessment": "Always conduct thorough preflight briefing"
            }
        }


# Create a singleton instance
openai_service = OpenAISummaryService()