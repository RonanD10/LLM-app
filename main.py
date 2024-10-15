import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

st.title("Medical Q & A")

st.text("For illustration, trained on heart and lung medical data.") 
st.text("Example questions: \n - 'What are the symptoms of pneomonia?'\n - 'How common is asthma?'")

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)
    st.header("Answer: ")
    st.write(response['result'])
