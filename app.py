import validators
import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeAudioLoader, YoutubeLoader, UnstructuredURLLoader


### streamlit
st.set_page_config(page_title="Summarize text from youtube or website", page_icon="panda")
st.title("Summarize from youtube or website")
st.subheader("Summarize url")

## get api key
with st.sidebar:
    api_key = st.text_input("Enter the api key", value=" ", type="password")

#setting up model and prompt
llm = ChatGroq(model="gemma2-9b-it", api_key=api_key)
prompt_template = """ 
Provide a summary of the following content in 300 words.
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])


url = st.text_input("URL", label_visibility="collapsed")

if st.button("summarize the content"):
    # strip() removes the empty characters
    if not api_key.strip() or not url.strip():
        st.error("Input the info")
    elif not validators.url(url):
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Waiting ..."):
                ## loading the website data
                if "youtube.com" in url:
                    loader= YoutubeLoader.from_youtube_url(url, add_video_info = True)

                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=True, 
                                                   headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0"})
                docs= loader.load()

                ## Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt= prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f'Exception:{e}')
