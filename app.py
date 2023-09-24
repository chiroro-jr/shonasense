import streamlit as st
from pickle import load
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# load the model
model = tf.keras.saving.load_model('./models/model_with_embedding.keras')

# load the tokenizer
tokenizer = load(open('./models/tokenizer.pkl', 'rb'))


def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict(encoded, verbose=0)
        # map predicted word index to word
        predicted_word_index = np.argmax(yhat)  # Find the index with the highest probability
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


"""
# ShonaSense
A text generation model that will predict the next 3-5 Shona words likely to be typed.
"""
st.divider()
"""
##### Get prediction
Enter some sample text (ideally 5 words) and the model will complete your sentence.
"""
sample_text = st.text_input("Enter the sample text",
                            placeholder="Muenzaniso: Mazuvano varimi varikunyanya kurima chibage")

if st.button("Generate Text", type="secondary"):
    if len(sample_text.split()) < 5:
        st.write("##### Result")
        st.write("No result. You need to provide at least 5 words")
    else:
        prediction = generate_seq(model, tokenizer, 10, sample_text, 5)
        st.write("##### Result")
        st.write(f"`Sample text provided`: {sample_text}")
        st.write(f"`Text generated`: {prediction}")
        st.write(f"`Combined text`: {sample_text} *{prediction}*")

else:
    st.write("##### Result")
    st.write("No result. You need to enter some text and press 'Generate Text'")

st.divider()
"""
## Reference Links
- [Github repository for project](https://github.com) - This contains all the code for this app (jupyter notebooks, 
datasets, saved models and so on.)
"""
