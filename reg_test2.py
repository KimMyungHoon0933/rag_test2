# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 02:31:36 2025

@author: A
"""

import streamlit as st
import tiktoken
from loguru import logger

from langchain_core.messages import ChatMessage
from langchain_community.chat_models import ChatOllama

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langserve import RemoteRunnable
import re
from difflib import SequenceMatcher

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)
def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")

        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)

    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb
def remove_repeated_qa_blocks(text, question_text):
    parts = text.split(question_text)
    if len(parts) <= 2:
        return text  # 반복 안 된 경우
    return question_text + parts[1]  # 첫 QA만 유지

def strip_lines_starting_with_q_format(text):
    
    lines = text.split('\n')
    filtered = [line for line in lines if not line.strip().startswith(("Q:", "A:", "###"))]
    return "\n".join(filtered).strip()
def remove_similar_sentences(text, threshold=0.9):
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered = []
    for sentence in sentences:
        is_duplicate = any(
            SequenceMatcher(None, sentence, existing).ratio() > threshold
            for existing in filtered
        )
        if not is_duplicate:
            filtered.append(sentence)
    return ' '.join(filtered)


def remove_repeated_qa_blocks(text, question_text):
    parts = text.split(question_text)
    if len(parts) <= 2:
        return text  # 반복 안 된 경우
    return question_text + parts[1]  # 첫 QA만 유지


def clean_llm_echo_artifacts(text):
    # 날짜 제거
    text = re.sub(r'\d{4}\.\d{2}\.\d{2}', '', text)
    # 학습 포맷 제거
    text = re.sub(r'###\s*입력[:：]?', '', text)
    text = re.sub(r'###\s*응답[:：]?', '', text)
    text = re.sub(r'입력[:：]', '', text)
    text = re.sub(r'출력[:：]', '', text)
    text = re.sub(r'Q[:：]', '', text)
    text = re.sub(r'A[:：]', '', text)
    return text.strip()



def truncate_after_first_q_format(text):
    match = re.search(r'Q[:：]', text)
    if match:
        return text[:match.start()].strip()
    return text.strip()

def remove_inline_qa_patterns(text):
    return re.sub(r'Q[:：].*?A[:：].*?(?=(Q[:：]|$))', '', text, flags=re.DOTALL).strip()

def truncate_from_first_q(text):
    q_match = re.search(r'\bQ[:：]', text)
    if q_match:
        return text[:q_match.start()].strip()
    return text

def main():
    global retriever

    st.set_page_config(
        page_title="Streamlit_remote_RAG",
        page_icon=":books:"
    )

    st.title("_RAG_test4 :red[Q/A Chat]_ :books:")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 채팅 대화기록을 점검
    if "store" not in st.session_state:
        st.session_state["store"] = dict()

    def print_history():
        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

    def add_history(role, content):
        st.session_state.messages.append(ChatMessage(role=role, content=content))

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        process = st.button("Process")

    if process:
        files_text = get_text(uploaded_files)
        cleaned_text = [
            type(doc)(page_content=clean_llm_echo_artifacts(doc.page_content)) for doc in files_text
        ]
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        retriever = vectorstore.as_retriever(search_type='mmr', verbose=True)
        st.session_state["retriever"] = retriever

        st.session_state.processComplete = True

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"
        }]
    def format_docs(docs):
        # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
        return "\n\n".join(doc.page_content for doc in docs)


#    RAG_PROMPT_TEMPLATE = """
#    당신은 동서대학교 컴퓨터소프트웨어과 안내 AI 입니다.
#    검색된 문맥을 사용하여 질문에 맞는 답변을 3문장 이내로 하세요.
#    답을 모른다면 모른다고 답변하세요.

#    Question: {question}
#    Context: {context}
#    Answer:"""
    RAG_PROMPT_TEMPLATE = """
    당신은 문서를 참고하여 사용자 질문에 답하는 AI입니다.
    
    - '###입력:', '###출력:', '###Instruction:', 'Input:', 'Q:', 'A:', 와 같은 포맷은 절대 사용하지 마세요.
    - 날짜(예: 2022.05.17)나 질문을 반복하지 마세요.
    - 문서에 기반하여 평문으로 자연스럽고 간결하게 1~2문장 이내로만 답하세요.
    - 문서에 없는 내용이라면 '문서에 해당 정보가 없습니다.'라고 답하세요.
    
    [문서 내용]
    {context}
    
    [질문]
    {question}
    """
    print_history()

    if user_input := st.chat_input("메시지를 입력해 주세요:"):
        # 사용자가 입력한 내용
        add_history("user", user_input)
        st.chat_message("user").write(f"{user_input}")

        with st.chat_message("assistant"):
            llm = RemoteRunnable("https://perfectly-lasting-halibut.ngrok-free.app/llm/")
            chat_container = st.empty()

            if st.session_state.processComplete == True:
                prompt1 = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

                retriever = st.session_state["retriever"]
                # 체인을 생성합니다.
                rag_chain = (
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough(),
                    }
                    | prompt1
                    | llm
                    | StrOutputParser()
                )

                answer = rag_chain.stream(user_input)
                chunks = []
                partial_text = ""
                for chunk in answer:
                    chunks.append(chunk)
                    partial_text += chunk
                
                    # 실시간 후처리 적용
                    step1 = remove_inline_qa_patterns(partial_text)
                    step2 = truncate_from_first_q(step1)
                    step3 = clean_llm_echo_artifacts(step2)
                
                    # 실시간 출력 (후처리된)
                    chat_container.markdown(step3)

                final_msg = "".join(chunks)
                
                # 먼저 Q: / A: / ### 제거 줄 단위
                step1 = strip_lines_starting_with_q_format(final_msg)
                
                # Q: 이후 줄 제거 (혹시 남아 있으면)
                step2 = truncate_after_first_q_format(step1)
                
                # 반복 QA 제거
                step3 = remove_repeated_qa_blocks(step2, user_input)
                
                # 텍스트 클렌징
                step4 = clean_llm_echo_artifacts(step3)
                
                # 중복 문장 제거는 마지막에 (안그러면 Q:가 붙어 있던 문장과 중복 처리될 수 있음)
                output_msg = remove_similar_sentences(step4)
                
                # 최종 저장
                add_history("ai", output_msg)
            else:
#                prompt2 = ChatPromptTemplate.from_template(
#                    "다음의 질문에 간결하게 답변해 주세요:\n{input}"
#                )
                prompt2 = ChatPromptTemplate.from_messages([
                    ("system",
                     "'###입력:', '###출력:', '###Instruction:', 'Input:', 'Q:', 'A:' 와 같은 포맷은 절대 사용하지 마세요. "
                     "같은 문장이나 질문을 반복하지 말고, 날짜나 질문을 다시 출력하지 마세요. "
                     "예: ‘2022.05.17’ 같은 날짜도 포함하지 마세요. "
                     "질문을 반복하거나, 다음 질문을 예상하지 마세요. "
                     "당신의 임무는 사용자의 질문에 대해 한 번만 대답하는 것입니다."
                    ),
                    ("user", "{input}")
                ])
                # 체인을 생성합니다.
                chain = prompt2 | llm | StrOutputParser()

                answer = chain.stream(user_input)  # 문서에 대한 질의
                chunks = []
                partial_text = ""
                for chunk in answer:
                    chunks.append(chunk)
                    partial_text += chunk
                
                    # 실시간 후처리 적용
                    step1 = remove_inline_qa_patterns(partial_text)
                    step2 = truncate_from_first_q(step1)
                    step3 = clean_llm_echo_artifacts(step2)
                
                    # 실시간 출력 (후처리된)
                    chat_container.markdown(step3)

                final_msg = "".join(chunks)
                
                # 먼저 Q: / A: / ### 제거 줄 단위
                step1 = strip_lines_starting_with_q_format(final_msg)
                
                # Q: 이후 줄 제거 (혹시 남아 있으면)
                step2 = truncate_after_first_q_format(step1)
                
                # 반복 QA 제거
                step3 = remove_repeated_qa_blocks(step2, user_input)
                
                # 텍스트 클렌징
                step4 = clean_llm_echo_artifacts(step3)
                
                # 중복 문장 제거는 마지막에 (안그러면 Q:가 붙어 있던 문장과 중복 처리될 수 있음)
                output_msg = remove_similar_sentences(step4)
                
                # 최종 저장
                add_history("ai", output_msg)
if __name__ == "__main__":
  main()
