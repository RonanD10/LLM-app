import streamlit as st
from qa_chain import create_vector_db, get_qa_chain

st.title("Wikipedia Q & A")

st.text("For illustration, trained on Wikipedia FAQs.") 
st.text("Example questions: \n - What stack does Wikipedia run on?\n - How do I edit an article?\n - Who co-founded Wikipedia?\n - Is Wikipedia a reliable source?")


question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)
    st.header("Answer: ")
    st.write(response['result'])
