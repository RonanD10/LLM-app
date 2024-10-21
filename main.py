import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

st.title("Wikipedia Q & A")

st.text("For illustration, trained on Wikipedia FAQs.") 
st.text("Example questions: \nWhat stack does Wikipedia run on?\nHow do I edit an article?\nWho co-founded Wikipedia?\nIs Wikipedia a reliable source?")


question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)
    st.header("Answer: ")
    st.write(response['result'])
