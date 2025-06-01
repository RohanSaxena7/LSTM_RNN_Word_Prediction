import streamlit as st 
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model = load_model("model_lstm.h5")

#Load the Tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
#Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] #ensure that sequence length matches max sequence length
    token_list = pad_sequences([token_list], maxlen = max_sequence_len-1, padding = 'pre') #prepadding
    predicted = model.predict(token_list, verbose = 0)
    predicted_word_index = np.argmax(predicted, axis=1) #we retrieve the index of the predicted word by the model
    for word, index in tokenizer.word_index.items(): #we convert the index into that word
        if index == predicted_word_index:
            return word
    return None

#streamlit app
st.title("Next Word Predictor")
st.write("This is based on Hamlet by Shakespeare data")
st.write("This is built using LSTM RNN")

input_text = st.text_input("Enter the sequence of words:", "To be or not to be")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1 #retreive the max sequence length from the model
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Next word: {next_word}")
    