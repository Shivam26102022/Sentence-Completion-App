import streamlit as st
import tensorflow as tf
import openai
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set your OpenAI API key

# Function to train the model
def train_model(faqs):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([faqs])
    input_sequences = []
    for sentence in faqs.split('\n'):
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

        for i in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i + 1])

    max_len = max([len(x) for x in input_sequences])
    padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')
    X = padded_input_sequences[:, :-1]
    y = padded_input_sequences[:, -1]

    y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, 100, input_length=max_len - 1))
    model.add(LSTM(150))
    model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=50)

    return model, tokenizer, max_len

# Function to complete a sentence using OpenAI GPT-3
def complete_sentence(prompt_template, continuation):
    prompt = f"{prompt_template} {continuation} and end properl"  # Combine template and continuation
    response = openai.Completion.create(
        engine="text-davinci-002",  # or other available engines
        prompt=prompt,
        max_tokens=100  # adjust as needed
    )
    
    # Extract the text from the OpenAI GPT-3 response
    completed_text = response.choices[0].text.strip()
    
    # Ensure that only one sentence is returned
    sentences = completed_text.split('.')
    if len(sentences) > 1:
        completed_text = sentences[0]  # Take the first sentence
    
    return completed_text

# Streamlit app
def main():
    st.title("Sentence Completion App")

    # Text area for pasting large multiline text (faqs)
    faqs = st.text_area("Paste your data here:", value="", height=200)

    # Button to train the model
    if st.button("Train Model"):
        if faqs:
            st.info("Training the model... This may take a while.")
            model, tokenizer, max_len = train_model(faqs)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.max_len = max_len
            st.success("Model trained successfully!")
        else:
            st.warning("Please paste data before training.")

    # Text box to input the sentence for prediction
    sentence_to_predict = st.text_input("Write a sentence for prediction:")
    
    # Text box to input the number of words for prediction (n)
    n = st.number_input("Enter the number of next words to predict:", min_value=1, value=3)

    # Button to predict the next words using the trained model
    if st.button("Predict Next Words"):
        if "model" not in st.session_state or "tokenizer" not in st.session_state:
            st.warning("Please train the model first.")
        else:
            text = sentence_to_predict if sentence_to_predict else "Before you can begin to determine what"
            word_count = 0

            for i in range(n):
                token_text = st.session_state.tokenizer.texts_to_sequences([text])[0]
                padded_token_text = pad_sequences([token_text], maxlen=st.session_state.max_len - 1, padding='pre')
                pos = np.argmax(st.session_state.model.predict(padded_token_text))

                for word, index in st.session_state.tokenizer.word_index.items():
                    if index == pos:
                        text += " " + word
                        word_count += 1
                        break

                if word_count >= n:
                    break

            st.success(f"Predicted Next Words: {text}")
    openai.api_key = st.text_input("Enter your OpenAI API key:", type="password")

    # Button to complete the sentence using ChatGPT
    if st.button("Complete Sentence"):
        if not sentence_to_predict:
            st.warning("Please enter a sentence for completion.")
        else:
            # Use the sentence_to_predict as a continuation prompt
            completed_sentence = complete_sentence(sentence_to_predict, "")
            st.success(f"Completed Sentence: {completed_sentence}")

if __name__ == "__main__":
    main()
