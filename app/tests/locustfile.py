import random
from locust import HttpUser, task, between
from faker import Faker
import json
API_URL = "http://localhost:7000"

fake = Faker("ar")

class NewsAnalyzerLoadTest(HttpUser):
    wait_time = between(1, 3)

    @task()
    def extract_news(self):
        story = fake.text(max_nb_chars=random.randint(300, 800))

        payload = {
            "story": story,
            "temperature": 0.3,
            "max_tokens": 512
        }

        llm_response = self.client.post(
            f"/extract-details",
            json=payload
        )
        # print(f"llm_response: {llm_response.json()}")
        

        if llm_response.status_code == 200:
            response.raise_for_status()
            with open("./vllm_tokens.txt", "a") as dest:
                dest.write(json.dumps({
                    "prompt": story,
                    "response": llm_response.json(),
                }, ensure_ascii=False) + "\n")

