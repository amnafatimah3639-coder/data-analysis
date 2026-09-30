import streamlit as st
import re
import torch

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Polite Response Generator",
    page_icon="💬",
    layout="centered"
)


# ============================================================
# HUGGING FACE MODEL
# ============================================================

# CHANGE THIS TO YOUR ACTUAL HUGGING FACE MODEL
MODEL_NAME = "Amna Fatimah/t5-small-polite-response"


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

    model = model.to("cpu")

    model.eval()

    return tokenizer, model


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

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
        truncation=True
    )

    inputs = {
        key: value.to("cpu")
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_length=80,
            num_beams=4,
            early_stopping=True
        )

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
    "Negative → Apology  |  "
    "Neutral → Clarification  |  "
    "Positive → Appreciation"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner("Loading model..."):

        tokenizer, model = load_model()

except Exception as error:

    st.error("Unable to load the Hugging Face model.")

    st.write(
        "Please check your Hugging Face model name "
        "and make sure the repository is public."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# TEXT INPUT
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

        User Text
             ↓
        Text Cleaning
             ↓
        T5 Prompt
             ↓
        Fine-Tuned T5-small
             ↓
        Polite Response

        Response styles:

        Negative → Apology
        Neutral → Clarification
        Positive → Appreciation
        """
    )
