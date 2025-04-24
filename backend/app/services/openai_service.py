"""
OpenAI Service for generating pilot-friendly weather report summaries
"""
import logging
import asyncio
import os
from typing import Dict, Any, Optional, Union
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


# Create a singleton instance
openai_service = OpenAISummaryService()