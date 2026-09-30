import streamlit as st
import requests
import re


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Polite Response Generator",
    page_icon="💬",
    layout="centered"
)


# ============================================================
# HUGGING FACE MODEL
# ============================================================

HF_USERNAME = "AmnaFatimah"
MODEL_NAME = "t5-small-polite-response"

API_URL = (
    f"https://api-inference.huggingface.co/models/"
    f"{HF_USERNAME}/{MODEL_NAME}"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    # Convert to lowercase
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
# GET HUGGING FACE TOKEN
# ============================================================

def get_huggingface_token():

    try:
        return st.secrets["HF_TOKEN"]

    except Exception:
        return None


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(text):

    cleaned_text = clean_text(text)

    prompt = (
        "Generate a polite response: "
        + cleaned_text
    )

    token = get_huggingface_token()

    headers = {
        "Content-Type": "application/json"
    }

    # Add authentication only if a token exists
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 80,
            "num_beams": 4
        }
    }

    try:

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        # Check HTTP status
        if response.status_code != 200:

            try:
                error_data = response.json()
            except Exception:
                error_data = response.text

            return cleaned_text, None, str(error_data)

        result = response.json()

        # Hugging Face text-generation response
        if isinstance(result, list):

            if len(result) > 0:

                generated = result[0].get(
                    "generated_text",
                    ""
                )

                # Remove the original prompt if returned
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].strip()

                return cleaned_text, generated, None

        # Error returned by Hugging Face
        if isinstance(result, dict):

            if "error" in result:

                return (
                    cleaned_text,
                    None,
                    result["error"]
                )

        return (
            cleaned_text,
            None,
            "Unexpected response from Hugging Face."
        )

    except requests.exceptions.Timeout:

        return (
            cleaned_text,
            None,
            "The Hugging Face request timed out. Please try again."
        )

    except requests.exceptions.RequestException as error:

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
        f"**Hugging Face user:** {HF_USERNAME}"
    )

    st.write(
        f"**Model:** {MODEL_NAME}"
    )

    st.write(
        f"**Repository:** "
        f"{HF_USERNAME}/{MODEL_NAME}"
    )


# ============================================================
# USER INPUT
# ============================================================

user_text = st.text_area(
    "Enter your text:",
    placeholder=(
        "Example: I am really disappointed "
        "with this service."
    ),
    height=150
)


# ============================================================
# GENERATE BUTTON
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
        # SHOW CLEANED TEXT
        # ----------------------------------------------------

        st.subheader("Cleaned Text")

        st.write(cleaned_text)

        # ----------------------------------------------------
        # SHOW RESPONSE
        # ----------------------------------------------------

        if response:

            st.subheader(
                "Generated Polite Response"
            )

            st.success(response)

        else:

            st.error(
                "The Hugging Face model could not "
                "generate a response."
            )

            st.code(error)


# ============================================================
# HOW IT WORKS
# ============================================================

with st.expander(
    "How does this application work?"
):

    st.write(
        """
        The application follows this pipeline:

        User Text
             ↓
        Text Cleaning
             ↓
        T5 Prompt
             ↓
        Hugging Face Model
             ↓
        Polite Response

        Text cleaning removes:

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
