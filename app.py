import streamlit as st
import re
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Polite Response Generator",
    page_icon="💬",
    layout="centered"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "Amna Fatimah/t5-small-polite-response"


# Streamlit Cloud normally runs on CPU.
device = torch.device("cpu")


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

    model = model.to(device)

    model.eval()

    return tokenizer, model


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
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

def generate_response(text, tokenizer, model):

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

    # Make sure inputs are on the same device
    # as the model.
    inputs = {
        key: value.to(model.device)
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
# APPLICATION TITLE
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

    with st.spinner("Loading model..."):

        tokenizer, model = load_model()

    st.success("Model loaded successfully!")

except Exception as e:

    st.error("The model could not be loaded.")

    st.write(
        "Please check that your Hugging Face "
        "model repository is correct and public."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# USER INPUT
# ============================================================

user_text = st.text_area(
    "Enter your text:",
    placeholder=(
        "Example: "
        "I am really disappointed with this service."
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

            cleaned_text, response = generate_response(
                user_text,
                tokenizer,
                model
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
        The application follows this pipeline:

        1. User enters text.
        2. URLs, mentions, hashtags and
           special characters are removed.
        3. The cleaned text is converted
           into a T5 prompt.
        4. The fine-tuned T5-small model
           processes the prompt.
        5. A polite response is generated.

        Response styles:

        • Negative → Apology
        • Neutral → Clarification
        • Positive → Appreciation
        """
    )
