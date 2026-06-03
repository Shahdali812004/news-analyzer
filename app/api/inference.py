# from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import requests
from openai import OpenAI
import json_repair
#CUSTOM MODULES
from app.utils.prompts import SYSTEM_MESSAGE
from app.helper.config import OPEN_AI_KEY
from app.schemas.news_schemas import NewsDetails
from app.schemas.translation_schemas import TranslatedStory
from app.helper.config import LORA_MODEL_ID, VLLM_URL

class PipelineBuilder:
    def __init__(self):
        self.vllm_url = VLLM_URL
        self.system_message = SYSTEM_MESSAGE


    def build_news_extraction_message(self,story_content:str):
        details_extraction_prompt = [
            {"role":"system",
            "content":self.system_message},
        {
            "role": "user",
            "content": "\n".join([
                "# Story:",
                story_content.strip(),
                "",
                "# Task:",
                "Extract the story details into a JSON.",
                "",
                "# Output Scheme:",
                json.dumps(
                    NewsDetails.model_json_schema(), ensure_ascii=False
                ),
                "",

                "# Output JSON:",
                "```json"
            ])
        }
        ]
        return details_extraction_prompt

    def build_translation_message(self,story_content:str, target_lang:str):
        translation_prompt = [
            {"role":"system",
            "content":self.system_message},
        {
            "role": "user",
            "content": "\n".join([
                "# Story:",
                story_content.strip(),
                "",
                "# Task:",
                f"You have to translate the story content into {target_lang} associated with a title into a JSON.",
                "",
                "# Output Scheme:", 
                json.dumps(
                    TranslatedStory.model_json_schema(), ensure_ascii=False
                ),
                "",
                "# Output JSON:",
                "```json"
            ])
        }
        ]
        return translation_prompt
    # def generate_response(self,messages,model_id=LORA_MODEL_ID, temperature=0.2, max_tokens=1000):  
    #     client = OpenAI(api_key=OPEN_AI_KEY,base_url="https://api.openai.com/v1")
    #     response = client.chat.completions.create(
    #         model=model_id,
    #         messages=messages,
    #         temperature=temperature,
    #         max_tokens=max_tokens
    #     )
    #     return response.choices[0].message.content.strip()
    def generate_response(self,messages, temperature: float = 0.2,max_tokens: int = 1000):
        client = OpenAI(
            api_key="EMPTY",  # ignored by vLLM
            base_url=self.vllm_url
        )

        response = client.chat.completions.create(
            model="news-analyzer",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()
    
    def json_parser(self,json_string):
        try:
            return json_repair.loads(json_string)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {str(e)}")

    def extract_news_details(self,story
                            ,temperature,max_tokens):

        messages = self.build_news_extraction_message(story)


        response = self.generate_response(messages, temperature, max_tokens)

        parsed = self.json_parser(response)
        return parsed
    def translate_story(self,story, target_lang
                        ,temperature,max_tokens):

        messages = self.build_translation_message(story, target_lang)

        response = self.generate_response(messages, temperature, max_tokens)

        parsed = self.json_parser(response)
        return parsed