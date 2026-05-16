import os
import re

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


# Load secrets from .env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


st.title("📚 Theology RAG Assistant")
st.write("Upload theology PDFs and ask questions about them.")


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Ask a theological question")


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    text = " ".join(pages)

    # normalize weird spacing
    text = re.sub(r'\s+', ' ', text)

    # fix words smashed together from camel-case-ish extraction
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    return text.strip()


def chunk_text(text, chunk_size=1500, overlap=300):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # try to end at sentence boundary
        if end < len(text):
            period = text.rfind(".", start, end)
            if period != -1:
                end = period + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def create_embedding(text):
    """Turn text into a vector embedding."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def cosine_similarity(vector_a, vector_b):
    """Compare two vectors and return a similarity score."""
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


def find_relevant_chunks(question, chunks, top_k=3):
    """Find the chunks most relevant to the user's question."""
    question_embedding = create_embedding(question)

    scored_chunks = []

    for chunk in chunks:
        chunk_embedding = create_embedding(chunk)
        similarity = cosine_similarity(question_embedding, chunk_embedding)

        scored_chunks.append(
            {
                "chunk": chunk,
                "similarity": similarity
            }
        )

    scored_chunks.sort(key=lambda item: item["similarity"], reverse=True)

    top_chunks = scored_chunks[:top_k]

    return [item["chunk"] for item in top_chunks]


def ask_ai(context, question):
    """Ask the AI to answer using only retrieved document context."""
    prompt = f"""
You are a careful theology research assistant.

Use ONLY the provided document context to answer the user's question.
If the answer is not found in the document context, say:
"I could not find that in the uploaded document."

Document context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You answer theology questions carefully, accurately, and based on the provided text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    document_text = extract_text_from_pdf(uploaded_file)

    st.write("Characters extracted:", len(document_text))

    chunks = chunk_text(document_text)

    st.write("Chunks created:", len(chunks))

    with st.expander("Preview first chunk"):
        st.text(chunks[0])

    if question:
        with st.spinner("Searching document and thinking..."):
            relevant_chunks = find_relevant_chunks(question, chunks)
            context = "\n\n---\n\n".join(relevant_chunks)
            answer = ask_ai(context, question)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved chunks used"):
            st.text(context)