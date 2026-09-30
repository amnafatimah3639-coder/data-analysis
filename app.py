import streamlit as st
import torch
import re

from transformers import T5Tokenizer, T5ForConditionalGeneration


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Polite Response Generator",
    page_icon="💬",
    layout="centered"
)


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

MODEL_NAME = "amna fatimah/t5-small-polite-response"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    return tokenizer, model


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags
    text = re.sub(r"#\w+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    # Remove extra spaces
    text = text.strip()

    return text


# ---------------------------------------------------------
# GENERATE RESPONSE
# ---------------------------------------------------------

def generate_response(text, tokenizer, model):

    cleaned_text = clean_text(text)

    prompt = f"Generate a polite response: {cleaned_text}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
        truncation=True
    )

    # Move input tensors to the same device as the model
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


# ---------------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------------

st.title("💬 Polite Response Generator")

st.write(
    "Enter a message and the fine-tuned T5 model will generate "
    "a polite response."
)

st.info(
    "The model was fine-tuned using TweetEval sentiment data."
)


# Load model
try:

    tokenizer, model = load_model()

    st.success("Model loaded successfully!")

except Exception as e:

    st.error("Could not load the model.")

    st.code(str(e))

    st.stop()


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

user_text = st.text_area(
    "Enter your text:",
    placeholder="Example: I am really disappointed with this service.",
    height=150
)


# ---------------------------------------------------------
# GENERATE BUTTON
# ---------------------------------------------------------

if st.button("Generate Polite Response", type="primary"):

    if not user_text.strip():

        st.warning("Please enter some text first.")

    else:

        with st.spinner("Generating response..."):

            cleaned_text, response = generate_response(
                user_text,
                tokenizer,
                model
            )

        st.subheader("Cleaned Text")

        st.write(cleaned_text)

        st.subheader("Generated Polite Response")

        st.success(response)


# ---------------------------------------------------------
# INFORMATION
# ---------------------------------------------------------

with st.expander("How does this application work?"):

    st.write(
        """
        **Pipeline:**

        1. User enters text
        2. Text is cleaned
        3. A prompt is created
        4. The fine-tuned T5-small model processes the prompt
        5. A polite response is generated

        The original training pipeline used TweetEval sentiment
        labels to create response styles:

        • Negative → Apology  
        • Neutral → Clarification  
        • Positive → Appreciation
        """
    )
