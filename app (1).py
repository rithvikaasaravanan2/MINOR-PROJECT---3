# app.py
# Domain Specific RAG Chatbot
# Streamlit Interface

import streamlit as st
import os
from dotenv import load_dotenv
from document_loader import extract_text_from_pdf,make_chunks
from vector_store import load_embedding_model,build_faiss_index
from rag_pipeline import get_answer

load_dotenv()

st.set_page_config(page_title='Domain Specific RAG Chatbot',layout='wide')
st.title('Domain Specific RAG Chatbot')
st.caption('Chatbot answers only from the pdf files you upload. Please verify important info yourself, answers can be wrong.')
st.markdown('---')

MAX_FILE_MB=20

# =====================
# session state
# =====================
if 'index' not in st.session_state:
  st.session_state.index=None
if 'chunks' not in st.session_state:
  st.session_state.chunks=None
if 'chat_history' not in st.session_state:
  st.session_state.chat_history=[]

#loading the embedding model once and keeping it cached
@st.cache_resource
def get_model():
  return load_embedding_model()

embed_model=get_model()

# =====================
# sidebar - upload
# =====================
st.sidebar.header('Upload your PDFs')
uploaded_files=st.sidebar.file_uploader('Choose PDF files',type=['pdf'],accept_multiple_files=True)

process_btn=st.sidebar.button('Process Documents')
clear_btn=st.sidebar.button('Clear Chat')

if clear_btn:
  st.session_state.chat_history=[]

if process_btn:
  if not uploaded_files:
    st.sidebar.warning('please upload atleast one pdf first')
  else:
    good_files=[]
    for f in uploaded_files:
      size_mb=f.size/(1024*1024)
      if size_mb>MAX_FILE_MB:
        st.sidebar.error(f.name+' is too big, max allowed is '+str(MAX_FILE_MB)+'MB')
      else:
        good_files.append(f)

    if len(good_files)>0:
      with st.spinner('reading pdfs and building index...'):
        all_pages=[]
        for f in good_files:
          pages=extract_text_from_pdf(f,f.name)
          all_pages.extend(pages)

        chunks=make_chunks(all_pages)

        if len(chunks)==0:
          st.sidebar.error('could not find any readable text in these pdfs')
        else:
          index=build_faiss_index(chunks,embed_model)
          st.session_state.index=index
          st.session_state.chunks=chunks
          names=[f.name for f in good_files]
          st.sidebar.success('processed: '+', '.join(names))

st.sidebar.markdown('---')
st.sidebar.caption('dont upload confidential files without permission')

# =====================
# main chat section
# =====================
if st.session_state.index is None:
  st.info('upload pdf files from the sidebar and click Process Documents to start chatting')
else:
  #showing old messages
  for msg in st.session_state.chat_history:
    with st.chat_message(msg['role']):
      st.write(msg['content'])
      if msg['role']=='assistant' and len(msg.get('sources',[]))>0:
        st.caption('Sources:')
        for s in msg['sources']:
          st.caption('- '+s['source']+', page '+str(s['page']))

  user_question=st.chat_input('ask something about your documents')

  if user_question:
    st.session_state.chat_history.append({'role':'user','content':user_question})
    with st.chat_message('user'):
      st.write(user_question)

    with st.chat_message('assistant'):
      with st.spinner('thinking...'):
        try:
          answer,sources=get_answer(user_question,st.session_state.index,st.session_state.chunks,embed_model)
        except Exception as e:
          answer='something went wrong: '+str(e)
          sources=[]
      st.write(answer)
      if len(sources)>0:
        st.caption('Sources:')
        for s in sources:
          st.caption('- '+s['source']+', page '+str(s['page']))

    st.session_state.chat_history.append({'role':'assistant','content':answer,'sources':sources})
