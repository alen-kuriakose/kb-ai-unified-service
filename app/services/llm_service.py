import os
from dotenv import load_dotenv
import logging
from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger("uvicorn.error")


def map_role_to_competencies_gemini(prompt_text: str, competency_framework_json: str, organization: str, role_title: str, department: str = None):
    logger.info("Starting Gemini LLM mapping call")
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    # logger.info(f"Prompt for Gemini: {prompt_text[:200]}... (truncated)")
    model = "gemini-2.5-flash-preview-04-17"
    user_prompt = prompt_text.replace("[Insert the entire competency framework JSON here]", competency_framework_json)
    user_prompt = user_prompt.replace("[organization]", organization)
    user_prompt = user_prompt.replace("[role_title]", role_title)
    user_prompt = user_prompt.replace("[department]", department or "")
    # logger.debug(f"User prompt: {user_prompt[:200]}... (truncated)")
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="INSERT_INPUT_HERE"),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["organization", "role_title", "mapped_competencies", "mapping_rationale"],
            properties={
                "organization": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="The name of the organization",
                ),
                "role_title": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="The title of the role",
                ),
                "mapped_competencies": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    description="List of mapped competencies",
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["category", "theme", "sub_themes", "confidence"],
                        properties={
                            "category": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="The category of the competency",
                                enum=["Behavioural", "Functional", "Domain"],
                            ),
                            "theme": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="The name of the competency theme",
                            ),
                            "sub_themes": genai.types.Schema(
                                type=genai.types.Type.ARRAY,
                                description="List of competency sub-themes",
                                items=genai.types.Schema(
                                    type=genai.types.Type.STRING,
                                ),
                            ),
                            "confidence": genai.types.Schema(
                                type=genai.types.Type.INTEGER,
                                description="Confidence level (0 to 100) of the competency mapping for the role",
                            ),
                        },
                    ),
                ),
                "mapping_rationale": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Explanation of why these competencies were selected",
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text=user_prompt),
        ],
    )
    output = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            # logger.info(f"Gemini chunk: {chunk}")
            output += chunk.text
        logger.info("Gemini LLM mapping call completed successfully.")
    except Exception as e:
        logger.error(f"Error during Gemini LLM call: {e}")
        raise
    logger.debug(f"Gemini LLM output: {output[:200]}... (truncated)")
    return output
