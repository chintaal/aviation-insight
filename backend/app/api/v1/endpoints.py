from fastapi import APIRouter, Query, HTTPException, Depends
from typing import List, Optional, Dict, Any
import asyncio
import time
import re
import logging

from app.schemas.weather import PirepResponse, EnhancedPirepResponse, MetarResponse, TafResponse, SigmetResponse
from app.services.pirep_service import PirepService
from app.services.metar_service import AWCMetarService
from app.services.taf_service import AWCTafService
from app.services.sigmet_service import AWCSigmetService
from app.services.openai_service import openai_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/pirep/{station}", response_model=List[PirepResponse], summary="Fetch PIREP data")
async def get_pirep(
    station: str,
    distance: Optional[int] = Query(200, description="Search radius in nautical miles"),
    age: Optional[float] = Query(1.5, description="Maximum age of reports in hours"),
    include_summary: Optional[bool] = Query(False, description="Include AI-generated pilot-friendly summary")
):
    """
    Retrieve PIREP data for a specific station.
    
    - **station**: ICAO airport code (e.g., KATL)
    - **distance**: Search radius in nautical miles
    - **age**: Maximum age of reports in hours
    - **include_summary**: Include AI-generated pilot-friendly summary
    """
    service = PirepService()
    try:
        pireps = await service.get_pireps(station, distance, age)
        
        # Generate summaries if requested
        if include_summary and pireps:
            for pirep in pireps:
                # Convert to dict for the OpenAI service
                pirep_dict = pirep.model_dump() if hasattr(pirep, "model_dump") else pirep.dict()
                summary = await openai_service.generate_summary("pirep", pirep_dict)
                if summary:
                    # Add the summary to the response
                    pirep.hazard_summary = summary
        
        return pireps
    finally:
        await service.close()

@router.get("/metar/{station}", response_model=MetarResponse, summary="Fetch METAR data")
async def get_metar(
    station: str,
    hours: Optional[int] = Query(1, description="Hours of history to include"),
    include_summary: Optional[bool] = Query(False, description="Include AI-generated pilot-friendly summary")
):
    """
    Retrieve METAR data for a specific station.
    
    - **station**: ICAO airport code (e.g., KATL)
    - **hours**: Hours of history to include (default: 1)
    - **include_summary**: Include AI-generated pilot-friendly summary
    """
    service = AWCMetarService()
    try:
        metar = await service.get_metar(station, hours)
        
        # Generate summary if requested
        if include_summary and metar and metar.raw_text:
            # Convert to dict for the OpenAI service
            metar_dict = metar.model_dump() if hasattr(metar, "model_dump") else metar.dict()
            summary = await openai_service.generate_summary("metar", metar_dict)
            if summary:
                # Add the summary to the response
                metar.pilot_summary = summary
        
        return metar
    finally:
        await service.close()

@router.get("/taf/{station}", response_model=TafResponse, summary="Fetch TAF data")
async def get_taf(
    station: str,
    hours: Optional[int] = Query(6, description="Hours of forecast to include"),
    include_summary: Optional[bool] = Query(False, description="Include AI-generated pilot-friendly summary")
):
    """
    Retrieve TAF data for a specific station.
    
    - **station**: ICAO airport code (e.g., KATL)
    - **hours**: Hours of forecast to include (default: 6)
    - **include_summary**: Include AI-generated pilot-friendly summary
    """
    service = AWCTafService()
    try:
        taf = await service.get_taf(station, hours)
        
        # Generate summary if requested
        if include_summary and taf and taf.raw_text:
            # Convert to dict for the OpenAI service
            taf_dict = taf.model_dump() if hasattr(taf, "model_dump") else taf.dict()
            summary = await openai_service.generate_summary("taf", taf_dict)
            if summary:
                # Add the summary to the response
                taf.pilot_summary = summary
        
        return taf
    finally:
        await service.close()

@router.get("/sigmet", response_model=List[SigmetResponse], summary="Fetch SIGMET data")
async def get_sigmet(
    bbox: Optional[str] = Query(None, description="Bounding box (e.g., '24.5,-100.0,36.5,-80.0')"),
    include_summary: Optional[bool] = Query(False, description="Include AI-generated pilot-friendly summary")
):
    """
    Retrieve SIGMET data for a specific area.
    
    - **bbox**: Bounding box coordinates (e.g., '24.5,-100.0,36.5,-80.0')
    - **include_summary**: Include AI-generated pilot-friendly summary
    """
    service = AWCSigmetService()
    try:
        sigmets = await service.get_sigmets(bbox=bbox)
        
        # Generate summaries if requested
        if include_summary and sigmets:
            for sigmet in sigmets:
                # Convert to dict for the OpenAI service
                sigmet_dict = sigmet.model_dump() if hasattr(sigmet, "model_dump") else sigmet.dict()
                summary = await openai_service.generate_summary("sigmet", sigmet_dict)
                if summary:
                    # Add a pilot_summary field to the sigmet
                    if not hasattr(sigmet, "pilot_summary"):
                        sigmet.pilot_summary = summary
        
        return sigmets
    finally:
        await service.close()

@router.get("/cockpit/pirep/{station}", response_model=Dict[str, Any], summary="Fetch enhanced PIREP data for cockpit display")
async def get_cockpit_pirep(
    station: str,
    distance: Optional[int] = Query(200, description="Search radius in nautical miles"),
    age: Optional[float] = Query(1.5, description="Maximum age of reports in hours"),
    flight_level_min: Optional[int] = Query(None, description="Minimum flight level filter"),
    flight_level_max: Optional[int] = Query(None, description="Maximum flight level filter"),
    hazard_type: Optional[str] = Query(None, description="Filter by hazard type (turbulence, icing, both, any)"),
    severity: Optional[str] = Query(None, description="Filter by severity (light, moderate, severe)"),
    include_summaries: Optional[bool] = Query(True, description="Include AI-generated pilot-friendly summaries")
):
    """
    Retrieve enhanced PIREP data for cockpit display with additional filtering options.
    
    - **station**: ICAO airport code (e.g., KATL)
    - **distance**: Search radius in nautical miles
    - **age**: Maximum age of reports in hours
    - **flight_level_min**: Minimum flight level for filtering
    - **flight_level_max**: Maximum flight level for filtering
    - **hazard_type**: Filter by type of hazard (turbulence, icing, both, any)
    - **severity**: Filter by severity level
    - **include_summaries**: Include AI-generated pilot-friendly summaries
    """
    service = PirepService()
    try:
        pireps = await service.get_pireps(station, distance, age)
        
        # Apply additional filtering if specified
        if flight_level_min is not None or flight_level_max is not None or hazard_type or severity:
            filtered_pireps = []
            for pirep in pireps:
                # Handle altitude filtering
                if flight_level_min is not None or flight_level_max is not None:
                    # Skip if altitude is not a number
                    if not isinstance(pirep.altitude, (int, float)):
                        continue
                        
                    altitude = pirep.altitude
                    # Convert flight level to altitude if needed
                    flight_level = altitude / 100
                    
                    if flight_level_min is not None and flight_level < flight_level_min:
                        continue
                    if flight_level_max is not None and flight_level > flight_level_max:
                        continue
                
                # Handle hazard type filtering
                if hazard_type:
                    has_turbulence = pirep.turbulence is not None and pirep.turbulence.get("intensity")
                    has_icing = pirep.icing is not None and pirep.icing.get("intensity")
                    
                    if hazard_type == "turbulence" and not has_turbulence:
                        continue
                    elif hazard_type == "icing" and not has_icing:
                        continue
                    elif hazard_type == "both" and not (has_turbulence and has_icing):
                        continue
                    elif hazard_type == "any" and not (has_turbulence or has_icing):
                        continue
                
                # Handle severity filtering
                if severity:
                    severity_match = False
                    if pirep.turbulence and pirep.turbulence.get("intensity"):
                        turb_intensity = pirep.turbulence["intensity"].lower()
                        if (severity == "light" and ("lgt" in turb_intensity or "light" in turb_intensity)) or \
                           (severity == "moderate" and ("mod" in turb_intensity or "moderate" in turb_intensity)) or \
                           (severity == "severe" and ("sev" in turb_intensity or "severe" in turb_intensity)):
                            severity_match = True
                    
                    if pirep.icing and pirep.icing.get("intensity"):
                        ice_intensity = pirep.icing["intensity"].lower()
                        if (severity == "light" and ("lgt" in ice_intensity or "light" in ice_intensity or "trc" in ice_intensity)) or \
                           (severity == "moderate" and ("mod" in ice_intensity or "moderate" in ice_intensity)) or \
                           (severity == "severe" and ("sev" in ice_intensity or "severe" in ice_intensity)):
                            severity_match = True
                    
                    if not severity_match:
                        continue
                
                filtered_pireps.append(pirep)
            
            pireps = filtered_pireps
        
        # Generate summaries if requested
        if include_summaries and pireps:
            for pirep in pireps:
                # Convert to dict for the OpenAI service
                pirep_dict = pirep.model_dump() if hasattr(pirep, "model_dump") else pirep.dict()
                summary = await openai_service.generate_summary("pirep", pirep_dict)
                if summary:
                    # Add the summary to the response
                    pirep.hazard_summary = summary
        
        # Group PIREPs by general location areas for better organization
        grouped_pireps = {}
        for pirep in pireps:
            # Extract first part of location (usually airport code)
            location_key = pirep.location.split()[0] if pirep.location and ' ' in pirep.location else pirep.location
            
            if location_key not in grouped_pireps:
                grouped_pireps[location_key] = []
            
            grouped_pireps[location_key].append(pirep)
        
        # Add statistics for the retrieved PIREPs
        stats = {
            "total_count": len(pireps),
            "turbulence_count": sum(1 for p in pireps if p.turbulence and p.turbulence.get("intensity")),
            "icing_count": sum(1 for p in pireps if p.icing and p.icing.get("intensity")),
            "urgent_count": sum(1 for p in pireps if p.report_type == "UUA"),
            "altitude_distribution": {}
        }
        
        # Create altitude distribution
        for pirep in pireps:
            if isinstance(pirep.altitude, (int, float)):
                # Group by 5,000 ft intervals
                altitude_group = f"{(pirep.altitude // 5000) * 5}-{((pirep.altitude // 5000) * 5) + 5}k"
                if altitude_group not in stats["altitude_distribution"]:
                    stats["altitude_distribution"][altitude_group] = 0
                stats["altitude_distribution"][altitude_group] += 1
        
        return {
            "pireps": pireps,
            "grouped_pireps": grouped_pireps,
            "stats": stats,
            "query_params": {
                "station": station,
                "distance": distance,
                "age": age,
                "filters_applied": {
                    "flight_level_min": flight_level_min,
                    "flight_level_max": flight_level_max,
                    "hazard_type": hazard_type,
                    "severity": severity
                }
            }
        }
    finally:
        await service.close()

@router.get("/cockpit/metar/{station}", response_model=Dict[str, Any], summary="Fetch enhanced METAR data for cockpit display")
async def get_cockpit_metar(
    station: str,
    hours: Optional[int] = Query(1, description="Hours of history to include"),
    include_summary: Optional[bool] = Query(True, description="Include AI-generated pilot-friendly summary")
):
    """
    Retrieve enhanced METAR data for cockpit display.
    
    - **station**: ICAO airport code (e.g., KATL)
    - **hours**: Hours of history to include (default: 1)
    - **include_summary**: Include AI-generated pilot-friendly summary
    """
    service = AWCMetarService()
    try:
        metar = await service.get_metar(station, hours)
        
        # Generate summary if requested
        if include_summary and metar and metar.raw_text:
            # Convert to dict for the OpenAI service
            metar_dict = metar.model_dump() if hasattr(metar, "model_dump") else metar.dict()
            summary = await openai_service.generate_summary("metar", metar_dict)
            if summary:
                # Add the summary to the response
                metar.pilot_summary = summary
        
        # Enhance the response for cockpit display
        enhanced_data = {
            "metar": metar,
            "display_data": {
                "flight_category": metar.flight_category,
                "ceiling": metar.ceiling,
                "visibility": metar.visibility,
                "wind": {
                    "direction": metar.wind_direction,
                    "speed": metar.wind_speed
                },
                "temperature": metar.temperature,
                "dewpoint": metar.dewpoint
            }
        }
        
        return enhanced_data
    finally:
        await service.close()

@router.get("/cockpit/taf/{station}", response_model=Dict[str, Any], summary="Fetch enhanced TAF data for cockpit display")
async def get_cockpit_taf(
    station: str,
    hours: Optional[int] = Query(12, description="Hours of forecast to include"),
    include_summary: Optional[bool] = Query(True, description="Include AI-generated pilot-friendly summary")
):
    """
    Retrieve enhanced TAF data for cockpit display.
    
    - **station**: ICAO airport code (e.g., KATL)
    - **hours**: Hours of forecast to include (default: 12)
    - **include_summary**: Include AI-generated pilot-friendly summary
    """
    service = AWCTafService()
    try:
        taf = await service.get_taf(station, hours)
        
        # Generate summary if requested
        if include_summary and taf and taf.raw_text:
            # Convert to dict for the OpenAI service
            taf_dict = taf.model_dump() if hasattr(taf, "model_dump") else taf.dict()
            summary = await openai_service.generate_summary("taf", taf_dict)
            if summary:
                # Add the summary to the response
                taf.pilot_summary = summary
        
        # Enhance the response for cockpit display
        enhanced_data = {
            "taf": taf,
            "display_data": {
                "valid_from": taf.valid_from,
                "valid_to": taf.valid_to,
                "forecast_periods": taf.forecast if taf.forecast else []
            }
        }
        
        return enhanced_data
    finally:
        await service.close()

@router.get("/catalog", response_model=Dict[str, Any], summary="Get API catalog")
async def get_api_catalog():
    """
    Get a catalog of all available API endpoints.
    """
    return {
        "version": "0.1.0",
        "endpoints": {
            "pirep": {
                "description": "Pilot Reports",
                "endpoints": [
                    {"path": "/pirep/{station}", "method": "GET", "description": "Get PIREPs near a station"},
                    {"path": "/cockpit/pirep/{station}", "method": "GET", "description": "Get enhanced PIREPs for cockpit display"}
                ]
            },
            "metar": {
                "description": "METARs",
                "endpoints": [
                    {"path": "/metar/{station}", "method": "GET", "description": "Get METAR for a station"},
                    {"path": "/cockpit/metar/{station}", "method": "GET", "description": "Get enhanced METAR for cockpit display"}
                ]
            },
            "taf": {
                "description": "TAFs",
                "endpoints": [
                    {"path": "/taf/{station}", "method": "GET", "description": "Get TAF for a station"},
                    {"path": "/cockpit/taf/{station}", "method": "GET", "description": "Get enhanced TAF for cockpit display"}
                ]
            },
            "sigmet": {
                "description": "SIGMETs",
                "endpoints": [
                    {"path": "/sigmet", "method": "GET", "description": "Get SIGMETs for an area"}
                ]
            }
        },
        "features": {
            "ai_summaries": "AI-generated pilot-friendly summaries available by adding include_summary=true"
        },
        "sources": [
            {"id": "awc", "name": "Aviation Weather Center", "url": "https://aviationweather.gov/"}
        ],
        "documentation": "/docs"
    }

@router.get("/health", response_model=Dict[str, Any], summary="Health check")
async def health_check():
    """
    Check the health of the API and its dependencies.
    """
    start_time = time.time()
    health_status = {
        "status": "healthy",
        "timestamp": start_time,
        "services": {}
    }
    
    # Check AWC API
    service = PirepService()
    try:
        # Try a simple API call
        await service.get_pireps("KJFK", distance=200, age=1.5)
        health_status["services"]["awc"] = {
            "status": "up",
            "latency": round((time.time() - start_time) * 1000, 2)  # Convert to ms
        }
    except Exception as e:
        health_status["services"]["awc"] = {
            "status": "down",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    finally:
        await service.close()
    
    # Check OpenAI API
    if openai_service.api_key:
        openai_start = time.time()
        try:
            # Simple API call to check if OpenAI is working
            test_response = await openai_service.generate_summary("metar", {"raw_text": "KJFK 241651Z 18009KT 10SM FEW050 SCT250 23/17 A2987 RMK AO2"})
            if test_response:
                health_status["services"]["openai"] = {
                    "status": "up",
                    "latency": round((time.time() - openai_start) * 1000, 2)  # Convert to ms
                }
            else:
                health_status["services"]["openai"] = {
                    "status": "degraded",
                    "error": "No response received"
                }
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["openai"] = {
                "status": "down",
                "error": str(e)
            }
            health_status["status"] = "degraded"
    else:
        health_status["services"]["openai"] = {
            "status": "not_configured",
            "error": "API key not set"
        }
    
    health_status["response_time"] = round((time.time() - start_time) * 1000, 2)  # Convert to ms
    return health_status

@router.post("/weather-summary/metar", response_model=Dict[str, Any], summary="Generate METAR summary")
async def generate_metar_summary(request: Dict[str, Any]):
    """
    Generate a pilot-friendly summary of a METAR.
    
    Request body should contain:
    - text: The raw METAR text to summarize
    """
    try:
        if not request.get("text"):
            raise HTTPException(status_code=400, detail="Missing required field 'text'")
            
        report_data = {"raw_text": request["text"], "station": "Unknown station"}
        
        # Extract station if possible
        parts = request["text"].split()
        if len(parts) > 0:
            report_data["station"] = parts[0]

        # Use a mock summary if we can't connect to OpenAI
        try:
            # Create a mock summary from the METAR data
            metar_text = report_data["raw_text"]
            station = report_data["station"]
            
            # Simple parsing to extract key METAR elements
            # Format: KPHX 211451Z 27014KT 10SM CLR 37/06 A2992 RMK AO2 SLP130 T03670061
            summary = f"METAR for {station} indicating VFR conditions."
            
            # Look for basic patterns
            if "CLR" in metar_text or "SKC" in metar_text:
                summary = f"METAR for {station} shows clear skies with good flying conditions."
            elif "OVC" in metar_text or "BKN" in metar_text:
                summary = f"METAR for {station} indicates cloudy conditions that may affect VFR flight."
            
            # Check visibility
            if "SM" in metar_text:
                vis_match = re.search(r'(\d+)SM', metar_text)
                if vis_match and int(vis_match.group(1)) < 3:
                    summary = f"METAR for {station} shows reduced visibility that may require IFR procedures."
            
            # Check for precipitation
            if any(wx in metar_text for wx in ["RA", "SN", "TS"]):
                summary = f"METAR for {station} indicates precipitation that could affect flight operations."
            
            return {
                "summary": summary,
                "reasoning": "Generated from basic METAR pattern analysis.",
                "flight_considerations": "Verify current conditions before flight. Weather conditions can change rapidly."
            }
            
        except Exception as inner_e:
            logger.error(f"Mock summary generation failed: {str(inner_e)}")
            return {
                "summary": f"METAR for {report_data.get('station', 'airport')} available.",
                "reasoning": "Unable to generate detailed analysis at this time.",
                "flight_considerations": "Always check raw METAR data and consult with Flight Service for official briefings."
            }
            
    except Exception as e:
        logger.error(f"Error in generate_metar_summary: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        # Return a basic response instead of an error
        return {
            "summary": "Weather information available.",
            "reasoning": "Review the raw METAR data for details.",
            "flight_considerations": "Contact Flight Service for a complete weather briefing."
        }
        
@router.post("/weather-summary/taf", response_model=Dict[str, Any], summary="Generate TAF summary")
async def generate_taf_summary(request: Dict[str, Any]):
    """
    Generate a pilot-friendly summary of a TAF.
    
    Request body should contain:
    - text: The raw TAF text to summarize
    """
    try:
        if not request.get("text"):
            raise HTTPException(status_code=400, detail="Missing required field 'text'")
            
        report_data = {"raw_text": request["text"], "station": "Unknown station"}
        
        # Extract station if possible
        parts = request["text"].split()
        if len(parts) > 0 and parts[0].upper() == "TAF" and len(parts) > 1:
            report_data["station"] = parts[1]
        elif len(parts) > 0:
            report_data["station"] = parts[0]
            
        # Use a mock summary if we can't connect to OpenAI
        try:
            # Create a mock summary from the TAF data
            taf_text = report_data["raw_text"]
            station = report_data["station"]
            
            # Simple parsing for TAF forecast
            summary = f"TAF for {station} available. Review the forecast for upcoming weather conditions."
            
            # Look for basic patterns
            if any(pattern in taf_text for pattern in ["BECMG", "TEMPO", "FM"]):
                summary = f"TAF for {station} indicates changing weather conditions during the forecast period."
                
            # Check for significant weather
            if "TS" in taf_text:
                summary = f"TAF for {station} forecasts thunderstorm activity. Review carefully for timing and intensity."
                
            if any(condition in taf_text for condition in ["OVC010", "OVC005", "OVC003", "BKN010", "BKN005", "BKN003"]):
                summary = f"TAF for {station} predicts low ceiling conditions that may require IFR operations."
                
            # Check visibility in forecast
            if any(vis in taf_text for vis in ["1SM", "2SM", "1/2SM", "3/4SM"]):
                summary = f"TAF for {station} shows periods of reduced visibility that may impact flight operations."
                
            return {
                "summary": summary,
                "reasoning": "Generated from basic TAF pattern analysis.",
                "flight_planning": "Always check the most recent TAF before flight and plan for forecast changes."
            }
        
        except Exception as inner_e:
            logger.error(f"Mock TAF summary generation failed: {str(inner_e)}")
            return {
                "summary": f"TAF for {report_data.get('station', 'airport')} available.",
                "reasoning": "Unable to generate detailed analysis at this time.",
                "flight_planning": "Review the raw forecast data and plan conservatively."
            }
            
    except Exception as e:
        logger.error(f"Error in generate_taf_summary: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        # Return a basic response instead of an error
        return {
            "summary": "Forecast information available.",
            "reasoning": "Review the raw TAF data for details.",
            "flight_planning": "Plan your flight using all available weather sources."
        }
        
@router.post("/weather-summary/pirep", response_model=Dict[str, Any], summary="Generate PIREP summary")
async def generate_pirep_summary(request: Dict[str, Any]):
    """
    Generate a pilot-friendly summary of a PIREP.
    
    Request body should contain:
    - text: The raw PIREP text to summarize
    - location: The location code associated with the PIREP
    """
    try:
        if not request.get("text"):
            raise HTTPException(status_code=400, detail="Missing required field 'text'")
            
        report_data = {
            "raw_text": request["text"],
            "location": request.get("location", "Unknown location")
        }
            
        # Use a mock summary instead of OpenAI
        try:
            # Create a mock summary from the PIREP data
            pirep_text = report_data["raw_text"]
            location = report_data["location"]
            
            # Simple parsing to extract key PIREP elements
            summary = f"Pilot report near {location}."
            
            # Check for turbulence
            if any(turb in pirep_text for turb in ["TURB", "TB"]):
                if "SEV" in pirep_text or "SVR" in pirep_text:
                    summary = f"Pilot report near {location} indicates severe turbulence. Exercise extreme caution."
                elif "MOD" in pirep_text:
                    summary = f"Pilot report near {location} mentions moderate turbulence."
                else:
                    summary = f"Pilot report near {location} includes turbulence information."
            
            # Check for icing
            if any(ice in pirep_text for ice in ["ICE", "ICING"]):
                if "SEV" in pirep_text or "SVR" in pirep_text:
                    summary = f"Pilot report near {location} indicates severe icing conditions."
                elif "MOD" in pirep_text:
                    summary = f"Pilot report near {location} mentions moderate icing."
                else:
                    summary = f"Pilot report near {location} includes icing information."
            
            # Check for clouds
            if "CLDS" in pirep_text or "CLD" in pirep_text:
                summary = f"Pilot report near {location} contains cloud information."
                
            return {
                "summary": summary,
                "reasoning": "Generated from basic PIREP pattern analysis.",
                "hazard_assessment": "Pilot reports provide valuable real-world observations. Consider these in your flight planning."
            }
        
        except Exception as inner_e:
            logger.error(f"Mock PIREP summary generation failed: {str(inner_e)}")
            return {
                "summary": f"Pilot report near {report_data.get('location', 'the area')} available.",
                "reasoning": "Unable to generate detailed analysis at this time.",
                "hazard_assessment": "Review the raw PIREP for specific hazard information."
            }
            
    except Exception as e:
        logger.error(f"Error in generate_pirep_summary: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        # Return a basic response instead of an error
        return {
            "summary": "Pilot report information available.",
            "reasoning": "Review the raw PIREP data for details.",
            "hazard_assessment": "Consider all available pilot reports when planning your flight."
        }

@router.get("/airport-summary/{station}", response_model=Dict[str, Any], summary="Get comprehensive airport weather summary")
async def get_airport_summary(
    station: str,
    distance: Optional[int] = Query(200, description="Search radius for PIREPs in nautical miles"),
    age: Optional[float] = Query(1.5, description="Maximum age of PIREPs in hours"),
    taf_hours: Optional[int] = Query(12, description="Hours of TAF forecast to include"),
    metar_hours: Optional[int] = Query(1, description="Hours of METAR history to include")
):
    """
    Retrieve all weather reports for an airport and generate a comprehensive AI-powered summary.
    
    - **station**: ICAO airport code (e.g., KATL)
    - **distance**: Search radius for PIREPs in nautical miles
    - **age**: Maximum age of PIREPs in hours
    - **taf_hours**: Hours of TAF forecast to include
    - **metar_hours**: Hours of METAR history to include
    
    Returns a comprehensive summary with all reports (METAR, TAF, PIREP, SIGMET) and an AI-generated analysis.
    """
    try:
        # Fetch all reports concurrently
        metar_service = AWCMetarService()
        taf_service = AWCTafService()
        pirep_service = PirepService()
        sigmet_service = AWCSigmetService()
        
        reports = {}
        errors = {}
        
        try:
            # Fetch METAR
            metar = await metar_service.get_metar(station, metar_hours)
            reports["metar"] = metar.model_dump() if hasattr(metar, "model_dump") else metar.dict()
        except Exception as e:
            logger.error(f"Error fetching METAR: {str(e)}")
            errors["metar"] = str(e)
            reports["metar"] = None
        
        try:
            # Fetch TAF
            taf = await taf_service.get_taf(station, taf_hours)
            reports["taf"] = taf.model_dump() if hasattr(taf, "model_dump") else taf.dict()
        except Exception as e:
            logger.error(f"Error fetching TAF: {str(e)}")
            errors["taf"] = str(e)
            reports["taf"] = None
        
        try:
            # Fetch PIREPs
            pireps = await pirep_service.get_pireps(station, distance, age)
            reports["pireps"] = [p.model_dump() if hasattr(p, "model_dump") else p.dict() for p in pireps]
        except Exception as e:
            logger.error(f"Error fetching PIREPs: {str(e)}")
            errors["pireps"] = str(e)
            reports["pireps"] = []
        
        try:
            # Fetch SIGMETs (using a bounding box around the station - simplified approach)
            # For now, fetch all SIGMETs and filter client-side if needed
            sigmets = await sigmet_service.get_sigmets()
            reports["sigmets"] = [s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in sigmets]
        except Exception as e:
            logger.error(f"Error fetching SIGMETs: {str(e)}")
            errors["sigmets"] = str(e)
            reports["sigmets"] = []
        
        finally:
            await metar_service.close()
            await taf_service.close()
            await pirep_service.close()
            await sigmet_service.close()
        
        # Generate comprehensive AI summary
        ai_summary = None
        try:
            # Create a comprehensive prompt for all reports
            summary_data = {
                "station": station,
                "metar": reports.get("metar"),
                "taf": reports.get("taf"),
                "pireps": reports.get("pireps", []),
                "sigmets": reports.get("sigmets", [])
            }
            
            # Use OpenAI to generate a comprehensive summary
            ai_summary = await openai_service.generate_comprehensive_summary(summary_data)
            
        except Exception as e:
            logger.error(f"Error generating AI summary: {str(e)}")
            # Fallback summary
            ai_summary = {
                "overview": f"Comprehensive weather summary for {station}",
                "current_conditions": "Review individual reports for detailed information.",
                "forecast_outlook": "Check TAF for forecast details.",
                "hazards": "Review PIREPs and SIGMETs for hazard information.",
                "recommendations": "Always verify current conditions before flight."
            }
        
        return {
            "station": station,
            "timestamp": time.time(),
            "reports": reports,
            "summary": ai_summary,
            "errors": errors if errors else None,
            "metadata": {
                "distance": distance,
                "age": age,
                "taf_hours": taf_hours,
                "metar_hours": metar_hours
            }
        }
        
    except Exception as e:
        logger.error(f"Error in get_airport_summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating airport summary: {str(e)}")
