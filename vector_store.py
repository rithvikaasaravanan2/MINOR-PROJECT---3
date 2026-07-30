# vector_store.py
# For to make embeddings and search them using faiss

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME='all-MiniLM-L6-v2'

def load_embedding_model():
  model=SentenceTransformer(MODEL_NAME)
  return model

def build_faiss_index(chunks,model):
  #convert all chunk texts to embeddings
  texts=[c['text'] for c in chunks]
  embeddings=model.encode(texts)
  embeddings=np.array(embeddings).astype('float32')

  dimension=embeddings.shape[1]
  index=faiss.IndexFlatL2(dimension)
  index.add(embeddings)
  return index

def search_index(query,index,chunks,model,top_k=4):
  query_embedding=model.encode([query])
  query_embedding=np.array(query_embedding).astype('float32')

  distances,indexes=index.search(query_embedding,top_k)
  results=[]
  for i in indexes[0]:
    if i!=-1 and i<len(chunks):
      results.append(chunks[i])
  return results
