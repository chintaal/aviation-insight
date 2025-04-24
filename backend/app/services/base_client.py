import aiohttp
import logging
import json
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class BaseApiClient:
    """Base class for all API clients"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp ClientSession"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                 headers: Optional[Dict[str, str]] = None, response_type: str = "json") -> Union[Dict[str, Any], str, list]:
        """
        Make a GET request to the API
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: HTTP headers
            response_type: Expected response type ("json" or "text")
            
        Returns:
            Response as dict/list (for JSON) or string (for plain text)
        """
        session = await self._get_session()
        
        # Prepare headers with API key if provided
        request_headers = {}
        if headers:
            request_headers.update(headers)
        if self.api_key:
            request_headers["Authorization"] = f"Bearer {self.api_key}"
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.get(url, params=params, headers=request_headers) as response:
                response.raise_for_status()
                
                # Always get the text response first
                text_response = await response.text()
                
                # If explicitly asking for text, return as text
                if response_type.lower() == "text":
                    return text_response
                
                # Try to parse as JSON regardless of content type
                try:
                    # Check if the response looks like JSON (starts with { or [)
                    if text_response.strip() and (text_response.strip()[0] in ['{', '[']):
                        return json.loads(text_response)
                except json.JSONDecodeError:
                    # If JSON parsing fails and we wanted JSON, log warning
                    if response_type.lower() == "json":
                        logger.warning(f"Failed to parse JSON response from {url}, returning as text")
                
                # If we get here, it's not valid JSON or we couldn't parse it
                return text_response
                
        except aiohttp.ClientError as e:
            logger.error(f"API request error: {e}, url='{url}'")
            raise
