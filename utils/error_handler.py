import functools
import time
from typing import Any, Callable, Optional, Type
from utils.logger import get_logger

logger = get_logger("error_handler")


class PipelineError(Exception):
    """Base class for all pipeline errors."""
    def __init__(self, stage: str, message: str, recoverable: bool = True):
        self.stage = stage
        self.recoverable = recoverable
        super().__init__(f"[{stage}] {message}")


class IngestionError(PipelineError):
    def __init__(self, message: str):
        super().__init__("PDF_INGESTION", message)


class OCRError(PipelineError):
    def __init__(self, message: str):
        super().__init__("OCR", message)


class ExtractionError(PipelineError):
    def __init__(self, message: str):
        super().__init__("CONCEPT_EXTRACTION", message)


class GraphError(PipelineError):
    def __init__(self, message: str):
        super().__init__("GRAPH_GENERATION", message)


class CompressionError(PipelineError):
    def __init__(self, message: str):
        super().__init__("COMPRESSION", message)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    fallback: Optional[Callable] = None,
):
    """Decorator: retry with exponential backoff, optional fallback."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            wait = delay
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {exc}"
                    )
                    if attempt < max_attempts:
                        time.sleep(wait)
                        wait *= backoff
            if fallback is not None:
                logger.info(f"{func.__name__} exhausted retries — running fallback")
                return fallback(*args, **kwargs)
            raise last_exc
        return wrapper
    return decorator


def safe_stage(stage_name: str, fallback_result: Any = None):
    """Decorator: catch any exception in a pipeline stage, log, return fallback."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                logger.error(f"Stage '{stage_name}' failed: {exc}", exc_info=True)
                return fallback_result
        return wrapper
    return decorator
