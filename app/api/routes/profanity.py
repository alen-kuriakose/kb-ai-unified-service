from fastapi import Body
from fastapi.responses import JSONResponse
from app.services.profanity_service import detect_language_service

from fastapi import APIRouter
from app.schemas.requests import ProfanityCheckRequest
from app.schemas.responses import ProfanityCheckResponse
from app.services.profanity_service import check_profanity_fasttext, check_profanity_llm, check_profanity_transformer
import logging

router = APIRouter()
logger = logging.getLogger("uvicorn.error")


@router.post(
    "/fasttext",
    response_model=ProfanityCheckResponse,
    summary="Check profanity using fastText model"
)
def profanity_check_fasttext(payload: ProfanityCheckRequest):
    logger.info(f"API: Received fastText profanity check for: {payload.text}")
    result = check_profanity_fasttext(payload.text)
    return result  # Return dict directly


@router.post(
    "/profanity_validator",
    response_model=ProfanityCheckResponse,
    summary="Check profanity using LLM"
)
def profanity_check_llm(payload: ProfanityCheckRequest):
    logger.info(f"API: Received LLM profanity check for: {payload.text}")
    result = check_profanity_llm(payload.text)
    return result  # Return dict directly


# New endpoint: Transformer-based profanity detection (English/Indic)
@router.post(
    "/transformer",
    response_model=ProfanityCheckResponse,
    summary="Check profanity using transformer models (English/Indic)"
)
def profanity_check_transformer(payload: ProfanityCheckRequest):
    logger.info(
        f"API: Received transformer profanity check for: {payload.text}")
    # Accept optional language from user
    user_language = getattr(payload, 'language', None)
    if not payload.text or str(payload.text).strip() == "":
        return {
            "status": "error",
            "message": "Input text is empty",
            "responseData": None
        }
    # Validate user_language if provided
    if user_language:
        user_language_lc = str(user_language).strip().lower()
        if user_language_lc not in ("english", "indic"):
            return {
                "status": "error",
                "message": "Invalid language. Only 'english' or 'indic' are allowed.",
                "responseData": None
            }
    else:
        user_language_lc = None

    # Call service and get detected language
    result = check_profanity_transformer(payload.text)
    detected_language = None
    if result and result.get('responseData'):
        detected_language_raw = result['responseData'].get('detected_language')
        advanced_detection = result['responseData'].get('advanced_detection', {})
        
        # Map detected language to 'english' or 'indic' for backward compatibility
        if str(detected_language_raw).lower() in ("english", "mixed_english"):
            detected_language = "english"
        elif str(detected_language_raw).lower().startswith("code_mixed_"):
            detected_language = "code_mixed"
        elif str(detected_language_raw).lower() in ["hindi", "tamil", "telugu", "bengali", "kannada", "malayalam", "gujarati", "punjabi", "oriya", "marathi", "urdu"]:
            detected_language = "indic"
        else:
            detected_language = "unknown"
            
        # Add user_language and cross-verification info
        result['responseData']['user_language'] = user_language_lc
        result['responseData']['detected_language_group'] = detected_language
        if user_language_lc:
            # Enhanced matching logic
            if user_language_lc == "english" and detected_language in ["english", "code_mixed"]:
                result['responseData']['language_match'] = True
            elif user_language_lc == "indic" and detected_language in ["indic", "code_mixed"]:
                result['responseData']['language_match'] = True
            else:
                result['responseData']['language_match'] = False
        else:
            result['responseData']['language_match'] = None
            
        # Add enhanced detection details
        result['responseData']['enhanced_detection'] = {
            "xlm_prediction": advanced_detection.get('xlm_prediction'),
            "xlm_confidence": advanced_detection.get('xlm_confidence'),
            "code_mixed": advanced_detection.get('code_mixed', False),
            "script_distribution": advanced_detection.get('distribution', {})
        }
    return result

# Language detection endpoint (English/Indic only)


@router.post(
    "/detect_language",
    summary="Detect if text is English or Indic language (minimum 5 characters)",
)
def detect_language_endpoint(
    text: str = Body(..., embed=True,
                     description="Text to detect language for")
):
    logger.info(f"API: Received language detection request for: {text}")
    result = detect_language_service(text)
    if result["status"] == "error":
        return JSONResponse(status_code=400, content=result)
    return result
