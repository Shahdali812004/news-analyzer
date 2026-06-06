from pydantic import BaseModel, Field
from typing import List, Literal, Optional
class TranslationRequest(BaseModel):
    story: str = Field(..., 
                       description="The content of the news article to be translated.")
    target_lang: Literal["English" ,"French"] = Field(..., 
                                                      description="The target language for translation.")
    temperature: Optional[float]  = Field(...
                                          , description="Sampling temperature for the model's response.") 
    max_tokens: Optional[int] = Field(...
                                      , description="Maximum number of tokens to generate in the response.")
    # model_id: Optional[str] = Field(...
    #                                 , description="The ID of the model to use for translation. If not provided, the default model will be used.")

class ExtractionRequest(BaseModel):
    story: str =Field(..., description="The content of the news article to extract details from.")
    temperature: Optional[float] = Field(...
                                         , description="Sampling temperature for the model's response.")
    max_tokens: Optional[int] = Field(...
                                      , description="Maximum number of tokens to generate in the response.")
    # model_id: Optional[str] = Field(...
    #                                 , description="The ID of the model to use for extraction. If not provided, the default model will be used.")