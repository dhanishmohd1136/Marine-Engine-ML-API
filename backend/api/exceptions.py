"""
Custom exceptions and error handlers for the Marine Engine ML API.
"""

from fastapi import Request
from fastapi.responses import JSONResponse


class MarineEngineAPIException(Exception):
    """Base exception for Marine Engine API errors."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ModelNotLoadedException(MarineEngineAPIException):
    """Raised when an inference is requested but the ML model is not loaded."""
    def __init__(self, message: str = "Machine learning model pipeline is not loaded or ready."):
        super().__init__(message, status_code=503)


class InvalidSensorDataException(MarineEngineAPIException):
    """Raised when input sensor parameters violate physical engine constraints."""
    def __init__(self, message: str):
        super().__init__(message, status_code=422)


async def marine_engine_exception_handler(request: Request, exc: MarineEngineAPIException) -> JSONResponse:
    """Standardized error response for API exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "path": request.url.path,
        },
    )
