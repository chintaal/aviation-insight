import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseModel):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Aviation Weather API Hub"
    
    # API keys for various weather services
    AWC_API_KEY: str = os.getenv("AWC_API_KEY", "")
    CHECKWX_API_KEY: str = os.getenv("CHECKWX_API_KEY", "")
    AVWX_API_KEY: str = os.getenv("AVWX_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

settings = Settings()
