import openpyxl
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_text_chunks(text):
    ts = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = ts.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    emb = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vs = FAISS.from_texts(text_chunks, embedding=emb)
    #vs.save_local(f'C:/Users/ishwarya.thirumuruga/Desktop/Ishwarya/Pdf chat/uploaded')
    vs.save_local(f'vectors_storage')


def get_conversational_chain():
    prmt_temp = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not found in the context, just say "Unable to find the answer", don't provide a wrong answer.
    context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prmt_temp, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(ques):
    emb = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local(f'vectors_storage', emb, allow_dangerous_deserialization=True)
    print(f"Performing similarity search for: {ques}")
    dox = new_db.similarity_search(ques)
    chain = get_conversational_chain()
    res = chain.invoke(
        {"input_documents": dox, "question": ques},
    )
    return res["output_text"]

import psycopg2

dbname = 'Ishwarya'
user = 'postgres'
password = 'Password1!'
host = 'localhost'  
port = '5432'  

try:
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    print("Connected to database!")

    cursor = conn.cursor()

    cursor.execute('SELECT id,source_number,attachment_number,file_name,content_type,created_by,created_on FROM integrity.attachment_ ORDER BY id ASC ')
    rows = cursor.fetchall()

    formatted_data=""
    for row in rows:
        formatted_row = "ID: {}\nSource Number: {}\nAttachment Number: {}\nFile Name: {}\nContent Type: {}\nCreated By: {}\nCreated On: {}\n\n".format(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6]
            )
        formatted_data += formatted_row
    
    chunks = get_text_chunks(formatted_data)
    get_vector_store(chunks)

    print(formatted_data)
    
    conn.commit()
    cursor.close()

except psycopg2.Error as e:
    print("Error connecting to PostgreSQL:", e)

finally:
    
    if conn is not None:
        conn.close()
        print('Database connection closed.')

import tkinter as tk
def submit_question_answer():
    question = question_entry.get()
    ans=user_input(question)
    answer_text.insert(tk.INSERT,question+"?\n"+ans+"\n")
    
    question_entry.delete(0, tk.END)
    answer_text.see(tk.END)

root = tk.Tk()
root.title("PDF Viewer and Question/Answer")
root.geometry("800x600")


uploaded_label = tk.Label(root, text="")
uploaded_label.pack(pady=5)


question_label = tk.Label(root, text="Question:")
question_label.pack(pady=5)


question_entry = tk.Entry(root, width=50)
question_entry.pack(pady=5)


submit_button = tk.Button(root, text="Submit", command=submit_question_answer)
submit_button.pack(pady=10)


answer_label = tk.Label(root, text="Answer:")
answer_label.pack(pady=5)


answer_text = tk.Text(root, width=80, height=5)
answer_text.pack(pady=5)


root.mainloop()
