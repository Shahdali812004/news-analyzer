# =====================================================
# API HELPERS
# =====================================================
import requests
API_URL = "http://localhost:7000"
def check_health():

    response = requests.get(
        f"{API_URL}/health",
        timeout=10
    )

    response.raise_for_status()

    return response.json()


def extract_details(story, temperature, max_tokens):

    payload = {
        "story": story,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(
        f"{API_URL}/extract-details",
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    return response.json()


def translate_story(story, target_lang, temperature, max_tokens):

    payload = {
        "story": story,
        "target_lang": target_lang,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(
        f"{API_URL}/translate-story",
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    return response.json()

