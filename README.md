# Role-Competency Mapping FastAPI App

A focused FastAPI application that uses Gemini LLM to map organizational roles to competencies from the provided competency framework. The application exposes a single LLM mapping endpoint.

## Features

- FastAPI framework with minimal structure
- Async/await patterns for LLM calls
- Proper error handling and validation
- API documentation with OpenAPI/Swagger
- Role to competency mapping using Gemini LLM
- Structured response with competency relevance scoring

## Project Structure

```
kb-ai-unified-service/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── role_mapping.py    # Role mapping endpoint logic
│   ├── core/
│   │   ├── config.py             # Application configuration
│   │   └── logger.py             # Logging setup
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── base.py              # Base Pydantic models
│   │   ├── competency.py        # Competency schemas
│   │   ├── requests.py          # Request schemas
│   │   └── responses.py         # Response schemas
│   └── services/
│       └── gemini_llm.py        # Gemini LLM integration
├── logs/                        # Application logs
├── main.py                     # FastAPI app entry point
├── requirements.txt            # Python dependencies
├── competency_framework.json   # Competency framework data
└── README.md                  # Project documentation
```

## Setup

1. Place your competency framework in `competency_framework.json` in the project root. Use the structure:

```
[
  {
    "name": "Behavioural",
    "competency_theme": [
      {
        "name": "Self-Awareness",
        "competency_sub_theme": [
          "Self-Analysis",
          "Self Confidence",
          "Purposefulness",
          "Self-Learning"
        ]
      }
    ]
  },
  ...
]
```

2. Copy `.env.example` to `.env` and set your Gemini LLM API key and URL.


3. Install dependencies (includes support for transformer and fastText models):

```
pip install -r requirements.txt
```

**Key dependencies for profanity detection:**
- `torch` and `transformers` (for English/Indic transformer-based detection)
- `fasttext` (for fastText-based detection)  
- `pandas` (for data processing)
- `tqdm` (for progress tracking)
- `scikit-learn` (for machine learning utilities)

**Models used:**
- **XLM-RoBERTa**: `papluca/xlm-roberta-base-language-detection` for language detection
- **English Toxicity**: `unitary/toxic-bert` for English profanity detection
- **Indic/Code-mixed**: `Hate-speech-CNERG/indic-abusive-allInOne-MuRIL` for Indic languages

4. Run the app:

```
uvicorn main:app --reload --port 5000
```

5. Test the endpoints at `/docs` (Swagger UI).


## API Endpoints

### 1. Map Role to Competencies

- **Endpoint:** `POST /api/v1/map_competencies`
- **Description:** Map a role to competencies using Gemini LLM and the provided competency framework.
- **Request Body:**
    ```json
    {
      "organization": "Ministry of Road Transport and Highways",
      "role_title": "Assistant Executive Engineer Civil",
      "department": "Engineering"  // Optional
    }
    ```
- **Response:**
    ```json
    {
      "organization": "Ministry of Road Transport and Highways",
      "role_title": "Assistant Executive Engineer Civil",
      "mapped_competencies": [
        {
          "category": "Behavioural",
          "theme": "Solution Orientation",
          "sub_themes": ["Analytical Thinking", "Systems Thinking"],
          "relevance": "High"
        }
      ],
      "mapping_rationale": "Brief explanation of why these competencies were selected"
    }
    ```


### 2. Profanity Check (fastText)

- **Endpoint:** `POST /api/v1/profanity/fasttext`
- **Description:** Check for profanity in text using a fastText model.
- **Request Body:**
    ```json
    {
      "text": "string"
    }
    ```
- **Response:**
    ```json
    {
      "status": "success",
      "message": "Profanity check completed",
      "responseData": {
        "word": "string",
        "isProfane": true,
        "confidence": 99.9,
        "category": "profane|clean"
      }
    }
    ```

### 3. Profanity Check (LLM)

- **Endpoint:** `POST /api/v1/profanity/profanity_validator`
- **Description:** Check for profanity in text using an LLM.
- **Request Body:**
    ```json
    {
      "text": "string"
    }
    ```
- **Response:**
    ```json
    {
      "status": "success",
      "message": "Profanity check completed (llm)",
      "responseData": {
        "word": "string",
        "isProfane": true,
        "confidence": 99.9,
        "category": "profane|clean"
      }
    }
    ```

### 4. Profanity Check (Transformer, English/Indic with Advanced Language Detection)

- **Endpoint:** `POST /api/v1/profanity/transformer`
- **Description:** Check for profanity in text using transformer models with advanced language detection. Supports English, Indic languages, and code-mixed content. Uses XLM-RoBERTa for language classification and intelligent model routing.
- **Request Body:**
    ```json
    {
      "text": "string",
      "language": "english" // or "indic" (optional, only these two allowed)
    }
    ```
- **Response:**
    ```json
    {
      "status": "success",
      "message": "Profanity check completed (transformer - English/Indic)",
      "responseData": {
        "word": "string",
        "isProfane": true,
        "confidence": 99.9,
        "category": "Profane|Non-Profane|Clean",
        "detected_language": "english|hindi|code_mixed_hindi_english|...",
        "model_used": "English (unitary/toxic-bert)|Indic (MuRIL)",
        "user_language": "english|indic|null",
        "detected_language_group": "english|indic|code_mixed",
        "language_match": true|false|null,
        "toxic_labels": "toxic,insult,...|null",
        "enhanced_detection": {
          "xlm_prediction": "english|hindi|...",
          "xlm_confidence": 0.95,
          "code_mixed": true|false,
          "script_distribution": {"english": 0.7, "hindi": 0.3}
        }
      }
    }
    ```

#### Advanced Language Detection Features
- **XLM-RoBERTa Integration**: Uses `papluca/xlm-roberta-base-language-detection` for accurate multilingual classification
- **Code-Mixed Detection**: Identifies Hinglish and other code-mixed content patterns
- **Script Analysis**: Analyzes Unicode character distribution across different Indic scripts
- **Intelligent Model Routing**: 
  - Pure English → `unitary/toxic-bert` (English toxicity model)
  - Indic languages & code-mixed → `Hate-speech-CNERG/indic-abusive-allInOne-MuRIL`
- **Pattern Recognition**: Detects common code-mixing patterns and English words mixed with Indic scripts

#### Language Validation
- Only `"english"` or `"indic"` are accepted for the `language` field. Any other value will return an error.
- The API will cross-verify the user-provided language with the detected language group and return a `language_match` boolean.

### 5. Language Detection (Advanced English/Indic Detection)

- **Endpoint:** `POST /api/v1/profanity/detect_language`
- **Description:** Advanced language detection using XLM-RoBERTa and script analysis. Detects English, Indic languages, and code-mixed content (minimum 5 characters required).
- **Request Body:**
    ```json
    {
      "text": "string"
    }
    ```
- **Response:**
    ```json
    {
      "status": "success",
      "detected_language": "english|indic|code_mixed|unknown",
      "raw": "english|hindi|code_mixed_hindi_english|...",
      "confidence": 0.95,
      "details": {
        "xlm_prediction": "english|hindi|...",
        "xlm_confidence": 0.98,
        "script_distribution": {"english": 0.7, "hindi": 0.3},
        "code_mixed": true|false,
        "patterns_found": 2
      }
    }
    ```

#### Language Detection Capabilities
- **Supports**: English, Hindi, Tamil, Telugu, Bengali, Kannada, Malayalam, Gujarati, Punjabi, Oriya, Marathi, Urdu
- **Code-Mixed Detection**: Hinglish, Tamil-English, and other combinations
- **XLM-RoBERTa**: High-accuracy transformer-based language identification
- **Script Analysis**: Unicode-based character distribution analysis
- **Pattern Matching**: Regex patterns for common code-mixing indicators

### 6. Health Check

- **Endpoint:** `GET /health`
- **Description:** Check the health of the service and its dependencies (e.g., Redis, competency framework).
- **Response:**
    ```json
    {
      "status": "healthy",
      "redis": "connected",
      "competency_framework": "loaded"
    }
    ```

## Data Models

- **RoleMappingRequest**
  - organization: string
  - role_title: string
  - department: string (optional)

- **CompetencyItem**
  - category: string
  - theme: string
  - sub_themes: string[]
  - relevance: string (Critical/High/Medium/Low)

- **RoleMappingResponse**
  - organization: string
  - role_title: string
  - mapped_competencies: CompetencyItem[]
  - mapping_rationale: string

## Notes
- Only one endpoint is exposed.
- All LLM and framework config is via environment variables.
- Logs are stored in the `logs` directory.
