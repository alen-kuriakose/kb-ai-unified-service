"""
Clean Language Detection System

XLM-RoBERTa based language detection with 97.9% accuracy across 100+ languages.

Usage:
    from app.services.language_detection_service import LanguageDetector
    
    detector = LanguageDetector()
    result = detector.detect("Hello, how are you?")
    print(f"Detected: {result['language']} ({result['confidence']:.2%})")
"""

import logging
from typing import Dict, List, Any
import pandas as pd

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning(
        "Transformers library not available. Please install with: pip install transformers torch")

logger = logging.getLogger("uvicorn.error")

# Simplified language mappings (commonly used languages)
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
    'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
    'ar': 'Arabic', 'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
    'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam', 'pa': 'Punjabi',
    'ur': 'Urdu', 'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish',
    'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish', 'cs': 'Czech', 'hu': 'Hungarian',
    'ro': 'Romanian', 'bg': 'Bulgarian', 'hr': 'Croatian', 'sk': 'Slovak', 'sl': 'Slovenian',
    'et': 'Estonian', 'lv': 'Latvian', 'lt': 'Lithuanian', 'el': 'Greek', 'he': 'Hebrew',
    'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian', 'ms': 'Malay', 'tl': 'Filipino',
    'sw': 'Swahili', 'af': 'Afrikaans', 'zu': 'Zulu', 'yo': 'Yoruba', 'ha': 'Hausa'
}

# Language groups for our profanity detection logic
INDIC_LANGUAGES = {'hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa', 'ur'}
ENGLISH_LANGUAGES = {'en'}


def get_language_name(code: str) -> str:
    """Get human-readable language name from code."""
    return LANGUAGE_NAMES.get(code.lower(), f"Unknown ({code})")


def get_language_group(code: str) -> str:
    """Get language group for profanity detection routing."""
    code_lower = code.lower()
    if code_lower in ENGLISH_LANGUAGES:
        return "english"
    elif code_lower in INDIC_LANGUAGES:
        return "indic"
    else:
        return "other"


class LanguageDetector:
    """Simple XLM-RoBERTa based language detector."""

    def __init__(self):
        """Initialize the language detector."""
        if not HF_AVAILABLE:
            logger.error(
                "Transformers library required. Install with: pip install transformers torch")
            self.classifier = None
            self.available = False
            return

        try:
            self.classifier = pipeline(
                "text-classification",
                model="ZheYu03/xlm-r-langdetect-model",
                top_k=None
            )
            self.available = True
            logger.info(
                "Successfully loaded XLM-RoBERTa language detection model")
        except Exception as e:
            logger.error(f"Failed to load language detection model: {e}")
            self.classifier = None
            self.available = False

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of the given text.

        Args:
            text (str): Text to analyze

        Returns:
            Dict with language detection results
        """
        if not self.available:
            return {
                'error': 'Language detection model not available',
                'language_code': None,
                'language': None,
                'confidence': 0.0
            }

        if not text or not text.strip():
            return {
                'error': 'Input text cannot be empty',
                'language_code': None,
                'language': None,
                'confidence': 0.0
            }

        try:
            # Get predictions from model
            results = self.classifier(text)

            # Handle different output formats
            if isinstance(results, list) and len(results) > 0:
                predictions = results[0] if isinstance(
                    results[0], list) else results
            else:
                raise ValueError("Unexpected model output format")

            # Sort by confidence and format
            sorted_predictions = sorted(
                predictions, key=lambda x: x['score'], reverse=True)

            # Get top prediction
            top_pred = sorted_predictions[0]
            language_code = top_pred['label'].lower()
            confidence = top_pred['score']
            language_name = get_language_name(language_code)
            language_group = get_language_group(language_code)

            return {
                'language_code': language_code,
                'language': language_name,
                'language_group': language_group,
                'confidence': confidence,
                'text_length': len(text),
                'all_predictions': [
                    {
                        'language_code': pred['label'].lower(),
                        'language': get_language_name(pred['label'].lower()),
                        'language_group': get_language_group(pred['label'].lower()),
                        'confidence': pred['score']
                    }
                    for pred in sorted_predictions[:5]
                ]
            }

        except Exception as e:
            logger.error(f"Language detection error: {str(e)}")
            return {
                'error': str(e),
                'language_code': None,
                'language': None,
                'confidence': 0.0
            }

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the model."""
        return {
            "name": "XLM-RoBERTa Language Detection",
            "model": "ZheYu03/xlm-r-langdetect-model",
            "accuracy": "97.9%",
            "languages": "100+ languages supported",
            "available": str(self.available)
        }


# Global instance for reuse
_language_detector = None


def get_language_detector():
    """Get or create the language detector instance"""
    global _language_detector
    if _language_detector is None:
        _language_detector = LanguageDetector()
    return _language_detector


def detect_language_service(text: str, min_chars: int = 5):
    """Enhanced language detection service using clean XLM-RoBERTa detection"""
    if not text or len(str(text).strip()) < min_chars:
        return {
            "status": "error",
            "message": f"Input text must be at least {min_chars} characters.",
            "detected_language": None
        }

    try:
        # Get the language detector
        detector = get_language_detector()

        # Use clean detection
        result = detector.detect(text)

        if 'error' in result:
            return {
                "status": "error",
                "message": result['error'],
                "detected_language": None
            }

        # Map to simplified categories for backward compatibility
        language_group = result.get('language_group', 'other')
        if language_group == 'english':
            detected_language_group = "english"
        elif language_group == 'indic':
            detected_language_group = "indic"
        else:
            detected_language_group = "other"

        return {
            "status": "success",
            "detected_language": detected_language_group,
            "raw": result['language_code'],
            "language_name": result['language'],
            "confidence": result['confidence'],
            "details": {
                "model_used": "XLM-RoBERTa (ZheYu03/xlm-r-langdetect-model)",
                "text_length": result['text_length'],
                # Top 3 predictions
                "top_predictions": result.get('all_predictions', [])[:3]
            }
        }
    except Exception as e:
        logger.error(f"Error in language detection service: {str(e)}")
        return {
            "status": "error",
            "message": f"Language detection error: {str(e)}",
            "detected_language": None
        }


def main():
    """Demo usage of the language detector."""
    print("🌐 Language Detection Demo")
    print("=" * 40)

    test_texts = [
        "Hello, how are you today?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás?",
        "नमस्ते, आप कैसे हैं?",
        "Tu literal definition hai chutiyapa ka"
    ]

    try:
        detector = LanguageDetector()
        info = detector.get_model_info()
        print(f"Using: {info['name']} ({info['accuracy']})")
        print()

        for text in test_texts:
            result = detector.detect(text)

            if 'error' not in result:
                print(f"Text: '{text}'")
                print(
                    f"Language: {result['language']} ({result['language_code']})")
                print(f"Group: {result['language_group']}")
                print(f"Confidence: {result['confidence']:.1%}")
                print()
            else:
                print(f"Error: {result['error']}")
                print()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install: pip install transformers torch")


if __name__ == "__main__":
    main()
