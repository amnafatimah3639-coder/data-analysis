import streamlit as st
import urllib.request
import urllib.error
import json
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
# HUGGING FACE SETTINGS
# ============================================================

HF_USERNAME = "AmnaFatimah"
MODEL_NAME = "t5-small-polite-response"

API_URL = (
    "https://router.huggingface.co/"
    "hf-inference/models/"
    f"{HF_USERNAME}/{MODEL_NAME}"
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
# GET HUGGING FACE TOKEN
# ============================================================

def get_hf_token():

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

    token = get_hf_token()

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 80
        }
    }

    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json"
    }

    if token:

        headers["Authorization"] = (
            "Bearer " + token
        )

    request = urllib.request.Request(
        API_URL,
        data=data,
        headers=headers,
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            response_data = response.read().decode(
                "utf-8"
            )

            result = json.loads(
                response_data
            )

        # Hugging Face returns a list
        if isinstance(result, list):

            if len(result) > 0:

                generated_text = result[0].get(
                    "generated_text",
                    ""
                )

                # Remove prompt if returned
                if generated_text.startswith(prompt):

                    generated_text = (
                        generated_text[
                            len(prompt):
                        ].strip()
                    )

                return (
                    cleaned_text,
                    generated_text,
                    None
                )

        # Hugging Face error response
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
            "Unexpected Hugging Face response."
        )

    except urllib.error.HTTPError as error:

        try:

            error_body = error.read().decode(
                "utf-8"
            )

        except Exception:

            error_body = str(error)

        return (
            cleaned_text,
            None,
            f"Hugging Face HTTP {error.code}: "
            f"{error_body}"
        )

    except urllib.error.URLError as error:

        return (
            cleaned_text,
            None,
            "Could not connect to Hugging Face: "
            + str(error.reason)
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
    "Enter a message and the fine-tuned "
    "T5-small model will generate a polite response."
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
        f"**Hugging Face account:** "
        f"{HF_USERNAME}"
    )

    st.write(
        f"**Model:** {MODEL_NAME}"
    )

    st.write(
        "The application uses the Hugging Face "
        "Inference API."
    )


# ============================================================
# USER INPUT
# ============================================================

user_text = st.text_area(
    "Enter your text:",
    placeholder=(
        "Example: I do not like your service."
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
        # CLEANED TEXT
        # ----------------------------------------------------

        st.subheader("Cleaned Text")

        st.write(cleaned_text)

        # ----------------------------------------------------
        # GENERATED RESPONSE
        # ----------------------------------------------------

        if response:

            st.subheader(
                "Generated Polite Response"
            )

            st.success(response)

        else:

            st.error(
                "The model could not generate "
                "a response."
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
        Hugging Face Model
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
