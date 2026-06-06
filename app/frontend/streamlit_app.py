import json
import time
import requests
import streamlit as st

API_URL = "http://localhost:7000"

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Arabic News Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

.block-container {
    max-width: 1200px;
}

[data-testid="metric-container"] {
    border-radius: 12px;
    padding: 12px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "total_requests" not in st.session_state:
    st.session_state.total_requests = 0

if "total_latency" not in st.session_state:
    st.session_state.total_latency = 0

# =====================================================
# API HELPERS
# =====================================================

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

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("📰 News Analyzer")

    try:

        health = check_health()

        st.success("API Connected")

        st.caption(
            f"{health['app_name']} v{health['app_version']}"
        )

    except Exception:

        st.error("API Offline")

    st.divider()

    st.subheader("⚙️ Generation Settings")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=3000,
        value=1000,
        step=100
    )

    st.divider()

    st.subheader("📊 Session Statistics")

    avg_latency = (
        st.session_state.total_latency /
        max(1, st.session_state.total_requests)
    )

    st.metric(
        "Requests",
        st.session_state.total_requests
    )

    st.metric(
        "Average Latency",
        f"{int(avg_latency)} ms"
    )

# =====================================================
# HEADER
# =====================================================

st.title("📰 Arabic News Analyzer")

st.caption(
    """
    Fine-tuned Qwen2.5-1.5B model for
    Arabic news extraction and translation.
    """
)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 News Extraction",
        "🌍 Translation",
        "🕓 History"
    ]
)

# =====================================================
# TAB 1
# =====================================================

with tab1:

    left, right = st.columns(
        [1, 1],
        gap="large"
    )

    with left:

        st.subheader(
            "News Story"
        )

        story = st.text_area(
            "Paste Arabic News Story",
            height=250
        )

        run_extract = st.button(
            "Analyze Story",
            use_container_width=True
        )

    with right:

        st.subheader(
            "Analysis Result"
        )

        result_placeholder = st.empty()

    if run_extract:

        if not story.strip():

            st.warning(
                "Please enter a story."
            )

        else:

            try:

                start = time.time()

                result = extract_details(
                    story.strip(),
                    temperature,
                    max_tokens
                )

                latency = int(
                    (time.time() - start) * 1000
                )

                st.session_state.total_requests += 1
                st.session_state.total_latency += latency

                st.session_state.history.append(
                    {
                        "task": "Extraction",
                        "latency": latency,
                        "result": result
                    }
                )

                with right:

                    st.success(
                        f"Completed in {latency} ms"
                    )

                    st.markdown(
                        "### 📰 Title"
                    )
                    st.write(
                        result["story_title"]
                    )

                    st.markdown(
                        "### 📂 Category"
                    )
                    st.info(
                        result["story_category"]
                    )

                    st.markdown(
                        "### 🏷️ Keywords"
                    )

                    st.write(
                        ", ".join(
                            result["story_keywords"]
                        )
                    )

                    st.markdown(
                        "### 📝 Summary"
                    )

                    for point in result[
                        "story_summary"
                    ]:
                        st.write(
                            f"• {point}"
                        )

                    st.markdown(
                        "### 👥 Entities"
                    )

                    for entity in result[
                        "story_entities"
                    ]:

                        c1, c2 = st.columns(
                            [3, 1]
                        )

                        c1.write(
                            entity["entity_value"]
                        )

                        c2.info(
                            entity["entity_type"]
                        )

                    st.download_button(
                        "Download JSON",
                        data=json.dumps(
                            result,
                            ensure_ascii=False,
                            indent=2
                        ),
                        file_name="news_details.json",
                        mime="application/json"
                    )

            except Exception as e:

                st.error(str(e))

# =====================================================
# TAB 2
# =====================================================

with tab2:

    left, right = st.columns(
        [1, 1],
        gap="large"
    )

    with left:

        story_translation = st.text_area(
            "Story",
            height=250,
            key="translation_story"
        )

        target_lang = st.selectbox(
            "Target Language",
            [
                "English",
                "French"
            ]
        )

        run_translation = st.button(
            "Translate Story",
            use_container_width=True
        )

    with right:

        st.subheader(
            "Translation Result"
        )

    if run_translation:

        if not story_translation.strip():

            st.warning(
                "Please enter a story."
            )

        else:

            try:

                start = time.time()

                result = translate_story(
                    story_translation,
                    target_lang,
                    temperature,
                    max_tokens
                )

                latency = int(
                    (time.time() - start) * 1000
                )

                st.session_state.total_requests += 1
                st.session_state.total_latency += latency

                st.session_state.history.append(
                    {
                        "task": "Translation",
                        "latency": latency,
                        "result": result
                    }
                )

                with right:

                    st.success(
                        f"Completed in {latency} ms"
                    )

                    st.markdown(
                        "### 📰 Translated Title"
                    )

                    st.write(
                        result["translated_title"]
                    )

                    st.markdown(
                        "### 📄 Translated Content"
                    )

                    st.write(
                        result["translated_content"]
                    )

                    st.download_button(
                        "Download Translation",
                        data=json.dumps(
                            result,
                            ensure_ascii=False,
                            indent=2
                        ),
                        file_name="translation.json",
                        mime="application/json"
                    )

            except Exception as e:

                st.error(str(e))

# =====================================================
# TAB 3
# =====================================================

with tab3:

    st.subheader(
        "Inference History"
    )

    if not st.session_state.history:

        st.info(
            "No requests yet."
        )

    else:

        for idx, item in enumerate(
            reversed(
                st.session_state.history
            ),
            start=1
        ):

            with st.expander(
                f"{item['task']} • {item['latency']} ms"
            ):

                st.json(item["result"])