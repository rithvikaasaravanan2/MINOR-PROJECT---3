# Domain Specific RAG Chatbot

This is a chatbot that answers questions from pdf files you upload. It uses RAG
(Retrieval Augmented Generation) so it only answers using the content of the pdfs,
not from its own general knowledge.

## What it does

- upload one or more pdf files (company policy, notes, manuals etc)
- app reads the text, splits it into small chunks
- chunks are converted to embeddings and stored using FAISS
- when you ask a question, it finds the most relevant chunks
- those chunks + your question are sent to the LLM (Groq llama model)
- answer is shown along with the source pdf name and page number
- if the answer is not found in the pdf it says so instead of making stuff up

## Project structure

```
domain_rag_chatbot/
|-- app.py                 -> streamlit ui
|-- document_loader.py     -> reads pdf and makes chunks
|-- vector_store.py        -> embeddings + faiss search
|-- prompt.py               -> the strict prompt template
|-- rag_pipeline.py         -> connects retrieval + llm call
|-- requirements.txt
|-- .env                    -> put your groq api key here
|-- documents/              -> 2 sample pdfs to test with
|-- tests/test_questions.csv
```

## How to run

1. clone this repo
2. create a virtual environment (optional but recommended)
3. install the requirements
```
pip install -r requirements.txt
```
4. get a free api key from groq (console.groq.com) and put it in the `.env` file:
```
GROQ_API_KEY=your_key_here
```
5. run the app
```
streamlit run app.py
```
6. upload the sample pdfs from the `documents/` folder (or your own pdf), click
   "Process Documents", then start asking questions in the chat box

## Sample documents

Two sample pdfs are included in `documents/` - company_policy.pdf and
employee_handbook.pdf. You can use these to test the chatbot, for example:

- "What is the leave policy?"
- "How is attendance calculated?"
- "Who is the company CEO?" (this should return not available since its not
  in the documents)

## Testing

`tests/test_questions.csv` has 17 test questions with the expected source page,
used to check if retrieval and answers are working correctly.

## Notes

- answers are generated only from the uploaded documents, not verified facts,
  so please double check anything important
- dont upload confidential files without permission
- max file size allowed is 20MB per pdf
