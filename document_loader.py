# document_loader.py
# For to read pdf files and split them into chunks

import pypdf
try:
  #newer langchain moved text splitters to their own package
  from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
  from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file,filename):
  #reads every page and keeps filename+page number as metadata
  reader=pypdf.PdfReader(file)
  pages_data=[]
  page_num=0
  for page in reader.pages:
    page_num+=1
    text=page.extract_text()
    if text and text.strip()!='':
      pages_data.append({'text':text,'source':filename,'page':page_num})
    #skipping empty pages so they dont mess up the chunks
  return pages_data

def make_chunks(pages_data,chunk_size=800,chunk_overlap=120):
  splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
  all_chunks=[]
  for page in pages_data:
    small_pieces=splitter.split_text(page['text'])
    for piece in small_pieces:
      all_chunks.append({'text':piece,'source':page['source'],'page':page['page']})
  return all_chunks
