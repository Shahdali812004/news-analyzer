from pydantic import BaseModel,Field,model_validator
from typing import List, Literal
storycategory = Literal[
    "politics", "sports", "art", "technology", "economy",
    "health", "entertainment", "science",
    "not_specified"
    
]
EntityType = Literal[
    "person-male", "person-female", "location", "organization", "event", "time",
    "quantity", "money", "product", "law", "disease", "artifact", "not_specified"
]
#############NEWS DETAILS RESPONSE MODELS#########

class Entity(BaseModel):
  entity_value: str = Field(...,description="The actual name or value of the entity.")
  entity_type : EntityType = Field(..., description="The type of recognized entity.")
  @model_validator(mode="before")  # ← runs BEFORE Pydantic validates fields
  @classmethod
  def normalize_entity_type(cls, values):
        valid_types = {
            "person-male", "person-female", "location", "organization", 
            "event", "time", "quantity", "money", "product", 
            "law", "disease", "artifact", "not_specified"
        }
        if "entity_type" in values and values["entity_type"] not in valid_types:
            values["entity_type"] = "not_specified"
        return values

class NewsDetails(BaseModel):
  story_title: str = Field(...,min_length=5, max_length=300,
                           description= "A fully informative and SEO optimized title of the story.")
  story_keywords: List[str] = Field(...,min_length=1,max_length=10,
                                      description="Relevant keywords associated with the story.")
  story_summary: List[str] = Field(
                                    ..., min_length=1, max_length=5,
                                    description="Summarized key points about the story (1-5 points).")
  story_category: storycategory = Field(..., description="Category of the news story.")


  story_entities: List[Entity] = Field(...,min_length=1,max_length=10,
                                       description="List of identified entities in the story.")
  @model_validator(mode="before")
  @classmethod
  def normalize_fields(cls, values):
        valid_categories = {
            "politics", "sports", "art", "technology", "economy",
            "health", "entertainment", "science", "not_specified"
        }
        if "story_category" not in values and values["story_category"] not in valid_categories:
            values["story_category"] = "not_specified"
        return values
