import streamlit as st
from utils import *
from pickle import load

st.header("This is a web to predict potential suicide from personal tweets.")
st.subheader("Be care and love your family and friends related with your life :)")

input = st.text_input("input your tweets :")

input = cleaning_one(input)
input = nltk_processing(input)
input = tf_processing([input])

with open('model.pkl', 'rb') as file:
    model = load(file)

result = 0
final_res = 0
if st.button("Click"):
    result = model.predict(input)
    final_res = 0 if result < 0.5 else 1

    if result < 0.5:
        st.text(f"the result is {final_res}, therefore it is not a potental suicide tweet with probability of {result[0][0]}")
    else:
        st.text(f"the result is {final_res}, therefore it is a potental suicide tweet with probability of {result[0][0]}")