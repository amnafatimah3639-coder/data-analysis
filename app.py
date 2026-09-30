import streamlit as st
import re

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)


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

# IMPORTANT:
# Replace this with YOUR actual Hugging Face model repository.

MODEL_NAME = "YOUR_USERNAME/t5-small-polite-response"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_NAME
    )

    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


# ============================================================
# CLEAN TEXT
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
# GENERATE RESPONSE
# ============================================================

def generate_response(
    text,
    tokenizer,
    model
):

    cleaned_text = clean_text(text)

    prompt = (
        "Generate a polite response: "
        + cleaned_text
    )

    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
        truncation=True
    )

    # Generate response
    output_ids = model.generate(
        **inputs,
        max_length=80,
        num_beams=4,
        early_stopping=True
    )

    # Convert tokens back into text
    response = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return cleaned_text, response


# ============================================================
# TITLE
# ============================================================

st.title("💬 Polite Response Generator")

st.write(
    "Enter a message and the fine-tuned "
    "T5-small model will generate a polite response."
)

st.info(
    "Negative → Apology | "
    "Neutral → Clarification | "
    "Positive → Appreciation"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner("Loading T5-small model..."):

        tokenizer, model = load_model()

    st.success("Model loaded successfully!")

except Exception as e:

    st.error("Unable to load the model.")

    st.write(
        "Check that your Hugging Face model "
        "repository name is correct and public."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# INPUT
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

            cleaned_text, response = (
                generate_response(
                    user_text,
                    tokenizer,
                    model
                )
            )

        st.subheader("Cleaned Text")

        st.write(cleaned_text)

        st.subheader(
            "Generated Polite Response"
        )

        st.success(response)


# ============================================================
# INFORMATION
# ============================================================

with st.expander(
    "How does this application work?"
):

    st.write(
        """
        Pipeline:

        1. User enters text.
        2. URLs are removed.
        3. Mentions are removed.
        4. Hashtags are removed.
        5. Special characters are removed.
        6. Text is normalized.
        7. A T5 prompt is created.
        8. The fine-tuned T5-small model
           generates a polite response.

        Response styles:

        Negative → Apology
        Neutral → Clarification
        Positive → Appreciation
        """
    )
