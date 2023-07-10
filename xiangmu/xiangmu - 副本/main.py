import streamlit as st
# from streamlit_chat import message
# import whisper
# from audiorecorder import audiorecorder
# import magic
import os
import numpy as np
import openai

# from langchain.document_loaders import UnstructuredWordDocumentLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from datetime import datetime
# import gtts  # 文字转语音
import pinecone  # 向量数据库
import streamlit as st  # 网站创建
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering.stuff_prompt import messages
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from keys import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV
import requests
from streamlit_chat import message

prompt_template = """I want you to be my orthodontist partne. I want you to be my orthodontist partner. Your name is CHILDALIGN Al. 

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
def show_messages_chat(messages):
    for num, each in enumerate(messages):
        if each["role"] == "user":
            message = st.chat_message("user")
            message.write(each['content'])
            # message(each['content'],  key=str(num), is_user=True)
        if each["role"] == "assistant":
            message = st.chat_message("assistant")
            message.write(each['content'])
            # message(each['content'],  key=str(num))
    # if len(messages) and messages[-1]['role'] == "assistant":
        # answer = messages[-1]['content']
        # audio = gtts.gTTS(answer, lang='zh')
        # audio.save("audio.wav")
        # st.audio('audio.wav', start_time=0)
        # os.remove("audio.wav")


def say():
    st.session_state['say'] = True
    st.session_state['say_user'] = True
    st.session_state['disabled'] = True
    st.session_state['prompt_input'] = st.session_state['prompt']
    st.session_state['prompt'] = ''
    st.session_state['history'].append({'role': 'user', 'content': st.session_state['prompt_input']})


def chat_say_Childalign(query, my_bar):
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
    llm = OpenAI(temperature=0, max_tokens=-1, frequency_penalty=0.4, openai_api_key=OPENAI_API_KEY)
    print('1:' + str(llm))
    my_bar.progress(10, text='正在查询资料库')
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    print('2:' + str(embeddings))
    docsearch = Pinecone.from_existing_index('dental', embedding=embeddings)
    print('3:' + str(docsearch))
    docs = docsearch.similarity_search(query, k=4)
    print('4:' + str(docs))
    my_bar.progress(60, text='找到点头绪了')
    chain = load_qa_chain(llm, chain_type='stuff', verbose=True)
    chain.llm_chain.prompt.template = prompt_template
    print('5:' + str(chain))
    my_bar.progress(90, text='可以开始生成答案了，请稍等')
    answer = chain.run(input_documents=docs, question=query, verbose=True)
    print('6:' + str(answer))
    my_bar.progress(100, text='好了')

    ip_address = requests.get('https://icanhazip.com/').text.strip()

    folder_path = "history"

    filename = os.path.join(folder_path, f"history_{ip_address}.txt")

    with open(filename, "a") as f:
        f.write(f"{datetime.now()}:\nQ: {query}\nA: {answer}\n")

    # with open(filename, "r") as f:
    #     history = f.read()

    # st.text_area("历史消息", history, height=200)

    # if st.button('结束程序'):
    #     os._exit(0)

    st.session_state['history'].append({'role': 'assistant', 'content': answer})
    st.session_state['r'] = True
    st.session_state['disabled'] = False
    return answer


if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'r' not in st.session_state:
    st.session_state['r'] = False

if 'say' not in st.session_state:
    st.session_state['say'] = False

if 'say_user' not in st.session_state:
    st.session_state['say_user'] = False

if 'prompt_input' not in st.session_state:
    st.session_state['prompt_input'] = ''

if 'disabled' not in st.session_state:
    st.session_state['disabled'] = False

st.set_page_config(page_title="Childalign GPT",
                   page_icon="C:/Users/张/Desktop/数据/我的/Ai-Dentist-Sample-Code-main/1.jpg", layout="wide",
                   initial_sidebar_state="auto")
# st.set_page_config(page_title="Childalign GPT", page_icon="C:/Users/张/Desktop/数据/我的/Ai-Dentist-Sample-Code-main/1.jpg", initial_sidebar_state="auto")
st.title('😬Childalign GPT😬')  # 用streamlit app创建一个标题

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}     //删除页脚
            .css-hi6a2p {padding-top: 0rem;} //删除顶部
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.write('---')

message = st.chat_message("assistant")
message.write("你好，我是Childalign GPT，你可以问我关于正畸的问题，例如：孩子8岁，龅牙，怎么治疗？")
# message("你好，我是Childalign GPT，你可以问我关于正畸的问题，例如：孩子8岁，龅牙，怎么治疗？",  key='start')  
show_messages_chat(st.session_state['history'])

col_all = st.columns([1, 2000, 1])
with col_all[1]:
    if len(st.session_state['history']) == 0:
        for i in range(10):
            st.write('')
    else:
        for i in range(4):
            st.write('')

    prompt = st.text_input(f"****",
                           on_change=say, key='prompt', disabled=st.session_state['disabled'])
    my_bar = st.progress(0, text='等待投喂问题哦')

    if st.session_state['say_user']:
        st.session_state['say_user'] = False
        st.experimental_rerun()

    if st.session_state['say']:
        st.session_state['say'] = False
        en_data = st.session_state['prompt_input']
        try:
            chat_say_Childalign(en_data, my_bar)
        except:
            pass

    if st.session_state['r']:
        st.session_state['r'] = False
        st.experimental_rerun()
