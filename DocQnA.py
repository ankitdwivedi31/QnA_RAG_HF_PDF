# Use a pipeline as a high-level helper
from transformers import pipeline
import gradio as gr
import pandas as pd
from PyPDF2 import PdfReader

model_path= "C:/Users/ankitdwivedi/OneDrive - Adobe/Desktop/NLP Projects/Video to Text Summarization/Model/models--deepset--roberta-base-squad2/snapshots/cbf50ba81465d4d8676b8bab348e31835147541b"

question_answer = pipeline("question-answering", model=model_path)
#question_answer = pipeline("question-answering", model="deepset/roberta-base-squad2")

# context = """Mark Elliot Zuckerberg ( born May 14, 1984) is an American businessman. He co-founded the social media service Facebook, along with his Harvard roommates in 2004, and its parent company Meta Platforms (formerly Facebook, Inc.), of which he is chairman, chief executive officer and controlling shareholder"""

# question = "Who is the founder of Facebook?"



def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        num_pages = len(pdf_reader.pages)
        content = ''
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            content += page.extract_text()
    return content


def get_answer(file, question):
    context = read_pdf(file)
    answer = question_answer(question=question, context=context)
    return answer['answer']

gr.close_all()

demo = gr.Interface(fn=get_answer,
                    inputs=[gr.File(label="Upload your File") ,gr.Textbox(label="Input the question",lines = 2)],
                    outputs=[gr.Textbox(label="Here is the answer",lines = 10)],
                    title="Project 5: Query your PDF",
                    description="""This app will be used for QnA.""")
demo.launch()