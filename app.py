import streamlit as st
import re

from huggingface_hub import InferenceClient


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Polite Response Generator",
    page_icon="💬",
    layout="centered"
)


# ============================================================
# MODEL
# ============================================================

MODEL_ID = "AmnaFatimah/t5-small-polite-response"


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

@st.cache_resource
def get_client():

    token = st.secrets.get(
        "HF_TOKEN",
        None
    )

    return InferenceClient(
        provider="hf-inference",
        api_key=token
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove hashtags
    text = re.sub(
        r"#\w+",
        "",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(text):

    cleaned_text = clean_text(text)

    prompt = (
        "Generate a polite response: "
        + cleaned_text
    )

    try:

        client = get_client()

        result = client.text_generation(
            prompt,
            model=MODEL_ID,
            max_new_tokens=80
        )

        return (
            cleaned_text,
            result,
            None
        )

    except Exception as error:

        return (
            cleaned_text,
            None,
            str(error)
        )


# ============================================================
# TITLE
# ============================================================

st.title("💬 Polite Response Generator")

st.write(
    "Enter a message and the fine-tuned T5-small "
    "model will generate a polite response."
)

st.info(
    "Negative → Apology  |  "
    "Neutral → Clarification  |  "
    "Positive → Appreciation"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write(
        "**Model:** "
        + MODEL_ID
    )

    st.write(
        "The model is hosted on Hugging Face."
    )


# ============================================================
# INPUT
# ============================================================

user_text = st.text_area(
    "Enter your text:",
    placeholder=(
        "Example: I do not like your service."
    ),
    height=150
)


# ============================================================
# BUTTON
# ============================================================

if st.button(
    "Generate Polite Response",
    type="primary"
):

    if not user_text.strip():

        st.warning(
            "Please enter some text first."
        )

    else:

        with st.spinner(
            "Generating response..."
        ):

            cleaned_text, response, error = (
                generate_response(user_text)
            )

        # ----------------------------------------------------
        # CLEANED TEXT
        # ----------------------------------------------------

        st.subheader("Cleaned Text")

        st.write(cleaned_text)

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        if response:

            st.subheader(
                "Generated Polite Response"
            )

            st.success(response)

        else:

            st.error(
                "The model could not generate a response."
            )

            st.code(error)


# ============================================================
# INFORMATION
# ============================================================

with st.expander(
    "How does this application work?"
):

    st.write(
        """
        Pipeline:

        User Text
             ↓
        Text Cleaning
             ↓
        T5 Prompt
             ↓
        Hugging Face
             ↓
        Polite Response

        Cleaning removes:

        • URLs
        • Mentions
        • Hashtags
        • Special characters
        • Extra spaces

        Response styles:

        • Negative → Apology
        • Neutral → Clarification
        • Positive → Appreciation
        """
    )
