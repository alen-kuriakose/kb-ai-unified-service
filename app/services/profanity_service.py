# Language detection for English/Indic (service function)
import pandas as pd
import os
import logging
import fasttext
from google import genai
from google.genai import types
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
from tqdm import tqdm
import sys

# Import the new clean language detection service
from app.services.language_detection_service import get_language_detector, detect_language_service as new_detect_language_service

# Constants
CLEAN_LANG_DETECTOR_NAME = "Clean XLM-RoBERTa"
PROFANITY_CHECK_COMPLETED = "Profanity check completed"

# --- Legacy Advanced Language Detection (for backward compatibility) ---
class AdvancedLanguageDetector:
    """Advanced language detector that can identify code-mixed languages"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xlm_tokenizer = None
        self.xlm_model = None
        self.load_xlm_model()
        
        # Enhanced script ranges with more comprehensive coverage
        self.script_ranges = {
            'hindi': (0x0900, 0x097F),      # Devanagari
            'bengali': (0x0980, 0x09FF),    # Bengali
            'tamil': (0x0B80, 0x0BFF),      # Tamil
            'telugu': (0x0C00, 0x0C7F),     # Telugu
            'kannada': (0x0C80, 0x0CFF),    # Kannada
            'malayalam': (0x0D00, 0x0D7F),  # Malayalam
            'gujarati': (0x0A80, 0x0AFF),   # Gujarati
            'punjabi': (0x0A00, 0x0A7F),    # Gurmukhi
            'oriya': (0x0B00, 0x0B7F),      # Oriya
            'marathi': (0x0900, 0x097F),    # Devanagari (same as Hindi)
            'urdu': (0x0600, 0x06FF),       # Arabic script (for Urdu)
        }
        
        # Common English words that appear in code-mixed text
        self.english_indicators = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was', 
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'a', 'an', 'some', 'any', 'all',
            'no', 'not', 'very', 'so', 'too', 'quite', 'really', 'just', 'only'
        }
        
        # Common Hinglish/code-mixed patterns
        self.code_mixed_patterns = [
            r'\b(kar|kar\w+|kya|hai|hain|tha|the|ho|hota|hoti|nahi|nahin)\b',
            r'\b(main|mein|me|tum|aap|woh|yeh|koi|kuch|sab|sabko)\b',
            r'\b(bhi|bhe|se|pe|ko|ka|ke|ki|mein|mai)\b',
            r'\b(good|bad|nice|cool|awesome|great|ok|okay)\b.*[\u0900-\u097F]',
            r'[\u0900-\u097F].*\b(good|bad|nice|cool|awesome|great|ok|okay)\b'
        ]
    
    def load_xlm_model(self):
        """Load XLM-RoBERTa model for Indic/English language detection"""
        try:
            logger.info("Loading XLM-RoBERTa for Indic/English language detection...")
            # Using a smaller, faster model for language detection
            model_name = "papluca/xlm-roberta-base-language-detection"
            self.xlm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.xlm_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            
            # Focus only on Indic languages and English
            self.xlm_id2lang = {
                4: 'english',
                7: 'hindi', 
                17: 'urdu'
            }
            
            # Map other languages to 'other' for filtering
            self.xlm_relevant_ids = {4, 7, 17}  # english, hindi, urdu
            
            logger.info("✅ XLM-RoBERTa loaded for Indic/English detection")
            self.use_xlm = True
        except Exception as e:
            logger.warning(f"⚠️ Could not load XLM-RoBERTa model: {e}")
            logger.info("Using rule-based detection only")
            self.use_xlm = False
    
    def detect_script_distribution(self, text):
        """Analyze the distribution of different scripts in the text"""
        if pd.isna(text):
            return {}
        
        text = str(text)
        char_counts = {lang: 0 for lang in self.script_ranges}
        english_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():  # Only count alphabetic characters
                char_code = ord(char)
                total_chars += 1
                
                # Check if it's English (basic Latin)
                if 0x0041 <= char_code <= 0x007A:  # A-Z, a-z
                    english_chars += 1
                    continue
                
                # Check Indic scripts
                for lang, (start, end) in self.script_ranges.items():
                    if start <= char_code <= end:
                        char_counts[lang] += 1
                        break
        
        if total_chars == 0:
            return {}
        
        # Calculate percentages
        distribution = {}
        for lang, count in char_counts.items():
            if count > 0:
                distribution[lang] = count / total_chars
        
        if english_chars > 0:
            distribution['english'] = english_chars / total_chars
        
        return distribution
    
    def detect_code_mixing_patterns(self, text):
        """Detect code-mixing patterns in the text"""
        if pd.isna(text):
            return False, []
        
        text = str(text).lower()
        found_patterns = []
        
        for pattern in self.code_mixed_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_patterns.append(pattern)
        
        # Check for English words mixed with non-Latin scripts
        words = text.split()
        english_words = [word for word in words if word.lower() in self.english_indicators]
        has_non_latin = any(ord(char) > 127 for char in text)
        
        is_code_mixed = (
            len(found_patterns) > 0 or 
            (len(english_words) > 0 and has_non_latin)
        )
        
        return is_code_mixed, found_patterns
    
    def detect_language_xlm(self, text):
        """Use XLM-RoBERTa for Indic/English language detection"""
        if not self.use_xlm or pd.isna(text) or str(text).strip() == "":
            return None, 0.0
        
        try:
            # Prepare input for XLM-RoBERTa
            inputs = self.xlm_tokenizer(
                str(text), 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.xlm_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
            
            # Only return results for relevant languages (Indic + English)
            if predicted_class in self.xlm_relevant_ids:
                detected_lang = self.xlm_id2lang[predicted_class]
                return detected_lang, confidence
            else:
                # Non-relevant language detected, treat as unknown
                return 'unknown', confidence
            
        except Exception as e:
            logger.error(f"XLM-RoBERTa detection error: {e}")
            return None, 0.0
    
    def detect_language_advanced(self, text):
        """Simplified language detection focused on Indic languages, code-mixed, and English"""
        if pd.isna(text) or str(text).strip() == "":
            return "unknown", 0.0, {}
        
        text = str(text)
        
        # Step 1: Use XLM-RoBERTa for initial classification
        xlm_lang, xlm_conf = None, 0.0
        if self.use_xlm:
            xlm_lang, xlm_conf = self.detect_language_xlm(text)
        
        # Step 2: Get script distribution and code-mixing patterns
        script_dist = self.detect_script_distribution(text)
        is_code_mixed, patterns = self.detect_code_mixing_patterns(text)
        
        # Step 3: Simplified decision logic for Indic/English focus
        result_details = {
            "xlm_prediction": xlm_lang,
            "xlm_confidence": xlm_conf,
            "distribution": script_dist,
            "code_mixed": is_code_mixed,
            "patterns": patterns
        }
        
        # High confidence XLM-RoBERTa prediction
        if xlm_lang and xlm_conf > 0.8:
            if xlm_lang == 'english':
                if is_code_mixed:
                    # Check for Indic scripts to identify specific code-mixing
                    indic_scripts = [lang for lang in ['hindi', 'tamil', 'telugu', 'bengali', 'kannada', 'malayalam'] 
                                   if lang in script_dist and script_dist[lang] > 0.1]
                    if indic_scripts:
                        dominant_indic = max(indic_scripts, key=lambda x: script_dist[x])
                        return f"code_mixed_{dominant_indic}_english", xlm_conf, result_details
                    else:
                        return "mixed_english", xlm_conf, result_details
                return "english", xlm_conf, result_details
            elif xlm_lang in ['hindi', 'urdu']:
                return xlm_lang, xlm_conf, result_details
        
        # Medium confidence or no XLM - use script analysis
        if script_dist:
            sorted_scripts = sorted(script_dist.items(), key=lambda x: x[1], reverse=True)
            
            # Check for English + Indic combination (code-mixing)
            if 'english' in script_dist and any(lang in script_dist for lang in ['hindi', 'tamil', 'telugu', 'bengali', 'kannada', 'malayalam']):
                indic_langs = [lang for lang in ['hindi', 'tamil', 'telugu', 'bengali', 'kannada', 'malayalam'] if lang in script_dist]
                if indic_langs and (is_code_mixed or script_dist['english'] > 0.2):
                    dominant_indic = max(indic_langs, key=lambda x: script_dist[x])
                    confidence = script_dist['english'] + script_dist[dominant_indic]
                    return f"code_mixed_{dominant_indic}_english", confidence, result_details
            
            # Single dominant script
            if sorted_scripts:
                primary_lang, primary_conf = sorted_scripts[0]
                if primary_conf > 0.6:
                    if primary_lang in ['hindi', 'tamil', 'telugu', 'bengali', 'kannada', 'malayalam', 'urdu']:
                        return primary_lang, primary_conf, result_details
                    elif primary_lang == 'english':
                        if is_code_mixed:
                            return "mixed_english", primary_conf, result_details
                        return "english", primary_conf, result_details
        
        # Fallback: if we detect code-mixing patterns but unclear scripts
        if is_code_mixed:
            return "code_mixed_unknown", 0.5, result_details
        
        # Default to English if XLM suggested English with lower confidence
        if xlm_lang == 'english':
            return "english", xlm_conf, result_details
        
        # Final fallback
        return "unknown", 0.0, result_details

# Global instance for reuse
_advanced_language_detector = None

def get_advanced_language_detector():
    """Get or create the advanced language detector instance"""
    global _advanced_language_detector
    if _advanced_language_detector is None:
        _advanced_language_detector = AdvancedLanguageDetector()
    return _advanced_language_detector

# --- Transformer-based Profanity Detection (English/Indic) ---
_transformer_models = {
    'english': None,
    'indic': None
}

def _load_english_model():
    if _transformer_models['english'] is not None:
        return _transformer_models['english']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert").to(device)
    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: 'NOT_TOXIC', 1: 'TOXIC'}
    _transformer_models['english'] = (tokenizer, model, id2label, device)
    return _transformer_models['english']

def _load_indic_model():
    if _transformer_models['indic'] is not None:
        return _transformer_models['indic']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    # Official MuRIL model label mapping (from config.json)
    id2label = {0: 'Normal', 1: 'Abusive'}
    
    _transformer_models['indic'] = (tokenizer, model, id2label, device)
    return _transformer_models['indic']

def check_profanity_transformer(text: str):
    """
    Detect profanity using transformer models (English/Indic) with clean language detection.
    Returns: dict with status, message, responseData
    """
    logger.info(f"Checking profanity (transformer) for: {text}")
    if pd.isna(text) or str(text).strip() == "":
        return {
            "status": "error",
            "message": "Input text is empty",
            "responseData": None
        }
    
    try:
        # Use clean language detection from new service
        detector = get_language_detector()
        lang_result = detector.detect(text)
        
        if 'error' in lang_result:
            logger.warning(f"Language detection failed: {lang_result['error']}")
            # Fallback to English model if detection fails
            return _process_english_model(text, "unknown", {"model_used": CLEAN_LANG_DETECTOR_NAME, "fallback": True})
        
        detected_lang = lang_result.get('language_code', 'unknown')
        language_group = lang_result.get('language_group', 'other')
        confidence = lang_result.get('confidence', 0.0)
        
        logger.info(f"Clean detection - Language: {detected_lang} ({language_group}), Confidence: {confidence:.3f}")
        
        # Model selection based on language group
        if language_group == "english":
            # Use English model for English content
            return _process_english_model(text, detected_lang, lang_result)
        elif language_group == "indic":
            # Use Indic model for Indic languages
            return _process_indic_model(text, detected_lang, lang_result)
        else:
            # Use English model as default for other/unknown languages
            logger.info(f"Using English model as fallback for language: {detected_lang}")
            return _process_english_model(text, detected_lang, lang_result)
            
    except Exception as e:
        logger.error(f"Transformer profanity detection error: {str(e)}")
        return {
            "status": "error",
            "message": f"Transformer model error: {str(e)}",
            "responseData": None
        }

def _process_english_model(text: str, detected_lang: str, _: dict):
    """Process text using English toxic-bert model"""
    tokenizer, model, id2label, device = _load_english_model()
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    toxic_indices = [i for i, p in enumerate(probs) if p >= 0.4]
    toxic_labels = [id2label[i] for i in toxic_indices]
    toxic_confidences = [float(probs[i]) for i in toxic_indices]
    max_conf = float(max(probs)) if len(probs) > 0 else 0.0
    
    if toxic_labels:
        main_label = 'Profane'
        main_confidence = max(toxic_confidences)
    else:
        main_label = 'Non-Profane'
        main_confidence = 1.0 - max_conf
    
    # Confidence adjustment
    if main_label == 'Profane' and max_conf < 0.8:
        main_label = 'Non-Profane'
        main_confidence = 1.0 - max_conf
    
    if main_label == 'Non-Profane' and main_confidence < 0.8:
        main_label = 'Profane'
        main_confidence = 1.0 - main_confidence
    
    return {
        "status": "success",
        "message": PROFANITY_CHECK_COMPLETED,
        "responseData": {
            "text": text,
            "isProfane": main_label == 'Profane',
            "confidence": round(main_confidence*100, 2),
            "category": main_label,
            "detected_language": detected_lang,
            "model_used": "English (toxic-bert)"
        }
    }

def _process_indic_model(text: str, detected_lang: str, _: dict):
    """Process text using Indic MuRIL model"""
    tokenizer, model, _, device = _load_indic_model()
    encoding = tokenizer.encode_plus(
        str(text),
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1)
        confidence = torch.max(probabilities, dim=1)[0]
    
    pred = predicted_class.cpu().item()
    conf = confidence.cpu().item()
    
    # Use official MuRIL labels: 0='Normal', 1='Abusive'
    if pred == 0:
        category = 'Clean'
        is_profane = False
    elif pred == 1:
        category = 'Profane/Abusive' 
        is_profane = True
    else:
        category = 'Processing Error'
        is_profane = False
    
    return {
        "status": "success",
        "message": PROFANITY_CHECK_COMPLETED,
        "responseData": {
            "text": text,
            "isProfane": is_profane,
            "confidence": round(conf*100, 2),
            "category": category,
            "detected_language": detected_lang,
            "model_used": "Indic (MuRIL)"
        }
    }

logger = logging.getLogger("uvicorn.error")

# Load fastText model once (assume model is at app/services/profanity_model.bin or similar)
FASTTEXT_MODEL_PATH = os.environ.get(
    "FASTTEXT_PROFANITY_MODEL", "app/services/profanity_model_english.bin")
fasttext_model = None
if os.path.exists(FASTTEXT_MODEL_PATH):
    try:
        fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
        logger.info(f"Loaded fastText model from {FASTTEXT_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Could not load fastText model: {e}")
else:
    logger.warning(f"fastText model not found at {FASTTEXT_MODEL_PATH}")


def check_profanity_fasttext(text: str):
    logger.info(f"Checking profanity (fastText) for: {text}")
    if not fasttext_model:
        logger.error("fastText model not loaded")
        return {
            "status": "error",
            "message": "fastText model not loaded",
            "responseData": None
        }
    labels, probabilities = fasttext_model.predict(text)
    label = labels[0]
    confidence = float(probabilities[0])
    is_profane = label == "__label__offensive"
    category = "profane" if is_profane else "clean"
    logger.info(
        f"Prediction: {label}, Confidence: {confidence}, Category: {category}")
    return {
        "status": "success",
        "message": PROFANITY_CHECK_COMPLETED,
        "responseData": {
            "text": text,
            "isProfane": is_profane,
            "confidence": round(confidence*100, 2),
            "category": category
        }
    }


def check_profanity_llm(text: str):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.5-flash-preview-04-17"
    # Prepare the prompt and schema as per user logic
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"Analyze the following text for profanity and respond with a JSON object containing:\n- 'contains_profanity': boolean (true if profanity is detected, false otherwise)\n- 'confidence': number between 0-100 (confidence percentage in your assessment)\n- 'reasoning': string (brief explanation of your decision, mentioning specific words or patterns if profanity is found)\n\nText to analyze: \"{text}\"\n\nRespond only with the JSON object, no additional text.")
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["contains_profanity", "confidence", "reasoning"],
            properties={
                "contains_profanity": genai.types.Schema(
                    type=genai.types.Type.BOOLEAN,
                    description="Whether profanity was detected in the text",
                ),
                "confidence": genai.types.Schema(
                    type=genai.types.Type.NUMBER,
                    description="Confidence percentage (0-100) in the profanity assessment",
                ),
                "reasoning": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Brief explanation of the decision, mentioning specific words or patterns if profanity is found",
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text="""Analyze the following text for profanity also keep the context of entire sentence in mind and respond with a JSON object containing:
                - "contains_profanity": boolean (true if profanity is detected, false otherwise)
                - "confidence": number between 0-100 (confidence percentage in your assessment)
                - "reasoning": string (explanation of your decision, mentioning specific words or patterns if profanity is found and a brief explanation of your reasoning)

                Text to analyze: "{text}"

                Respond only with the JSON object, no additional text."""
                                 ),
        ],
    )
    try:
        output = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            output += chunk.text
        import json
        data = json.loads(output)
        is_profane = data.get("contains_profanity", False)
        confidence = data.get("confidence", 0)
        reasoning = data.get("reasoning", "")
        category = "profane" if is_profane else "clean"
        return {
            "status": "success",
            "message": PROFANITY_CHECK_COMPLETED,
            "responseData": {
                "text": text,
                "isProfane": is_profane,
                "confidence": confidence,
                "category": category,
                "reasoning": reasoning
            }
        }
    except Exception as e:
        logger.error(f"Error during LLM profanity check: {e}")
        return {
            "status": "error",
            "message": str(e),
            "responseData": None
        }
def detect_language_service(text: str, min_chars: int = 5):
    """Clean language detection service using XLM-RoBERTa detection"""
    if not text or len(str(text).strip()) < min_chars:
        return {
            "status": "error",
            "message": f"Input text must be at least {min_chars} characters.",
            "detected_language": None
        }
    
    try:
        # Use the new clean language detection service
        return new_detect_language_service(text, min_chars)
    except Exception as e:
        logger.error(f"Error in clean language detection: {str(e)}")
        return {
            "status": "error",
            "message": f"Language detection error: {str(e)}",
            "detected_language": None
        }