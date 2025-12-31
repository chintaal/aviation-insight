#!/usr/bin/env python
import uvicorn
import logging

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("startup")
    logger.info("Starting Aviation Weather API Hub server...")
    
    try:
        uvicorn.run(
            "app.api.api:app", 
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
