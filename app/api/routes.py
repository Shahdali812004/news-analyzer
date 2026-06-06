from fastapi import APIRouter, HTTPException
from app.schemas.request_schemas import TranslationRequest, ExtractionRequest
from app.schemas.translation_schemas import TranslatedStory  
from app.schemas.news_schemas import NewsDetails
from app.helper.config import APP_NAME, APP_VERSION
from app.api.inference import(
    PipelineBuilder
    )

router = APIRouter()
pipeline = PipelineBuilder()
@router.get("/health", tags=["Health Check"],description="Check the health status of the API")
async def health_check():
    return {"status": "UP&RUNNING",
            "app_name": APP_NAME,
            "app_version": APP_VERSION}

@router.post("/extract-details",tags=["News Details Extraction"], 
             description="Extract details from a news story and return them in a structured JSON format."
             ,response_model=NewsDetails)
async def extract_details(request: ExtractionRequest):
    response = pipeline.extract_news_details(story=request.story
                         ,temperature=request.temperature,max_tokens=request.max_tokens)
    print("LLM RAW RESPONSE KEYS:", response.keys())  # <-- add this
    print("LLM RAW RESPONSE:", response)
    if not isinstance(response, ValueError):
        return NewsDetails(**response)
    else:
        raise HTTPException(status_code=400, detail=str(response))
    
@router.post("/translate-story",tags=["Story Translation"], 
             description="Translate a news story into a specified target language and return the translated content along with the title in a structured JSON format.")
async def translate_story(request: TranslationRequest):
    response = pipeline.translate_story(story=request.story, target_lang=request.target_lang
                         ,temperature=request.temperature,max_tokens=request.max_tokens)
    if not isinstance(response, ValueError):
        return TranslatedStory(**response)
    else:
        raise HTTPException(status_code=400, detail=str(response))