# rag_pipeline.py
# For to combine retrieval + calling the llm and get final answer

import os
import requests
from prompt import build_prompt
from vector_store import search_index

GROQ_URL='https://api.groq.com/openai/v1/chat/completions'

def make_context(chunks):
  #join the retrieved chunks together with their source tag on top
  context=''
  for c in chunks:
    tag='['+c['source']+' - page '+str(c['page'])+']'
    context=context+tag+'\n'+c['text']+'\n\n'
  return context

def call_llm(prompt_text):
  api_key=os.getenv('GROQ_API_KEY')
  headers={'Authorization':'Bearer '+str(api_key),'Content-Type':'application/json'}
  payload={
    'model':'llama-3.1-8b-instant',
    'messages':[{'role':'user','content':prompt_text}],
    'temperature':0
  }
  response=requests.post(GROQ_URL,headers=headers,json=payload)
  data=response.json()
  #groq gives the answer inside choices[0].message.content, same as openai format
  if 'choices' not in data:
    raise Exception('groq api error: '+str(data))
  answer=data['choices'][0]['message']['content']
  return answer

def get_answer(question,index,chunks,model,top_k=4):
  matched_chunks=search_index(question,index,chunks,model,top_k)

  if len(matched_chunks)==0:
    return "I could not find this information in the uploaded documents.",[]

  context=make_context(matched_chunks)
  final_prompt=build_prompt(context,question)
  answer=call_llm(final_prompt)

  sources=[]
  for c in matched_chunks:
    sources.append({'source':c['source'],'page':c['page']})

  return answer,sources
