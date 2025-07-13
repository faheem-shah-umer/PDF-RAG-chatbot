import os
import json
import hashlib
import pymupdf4llm
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from langchain.text_splitter import NLTKTextSplitter
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings
from PIL import Image
from unstructured.partition.pdf import partition_pdf
import re

# Load configuration
with open("config.json", "r") as config_file:
    config = json.load(config_file)

pdf_directory = config["data_sources"]["pdf"]["directory"]
vector_store_path = config["vector_store"]["path"]

os.makedirs(vector_store_path, exist_ok=True)

# Initialize Qdrant Vector Store
def create_vstore(vector_store_path):
    vstore = Qdrant.from_documents(
        [Document(page_content='')],
        HuggingFaceEmbeddings(),
        path=vector_store_path,
        collection_name='all',
    )
    return vstore

vstore = create_vstore(vector_store_path)

# Utility Functions
def compute_hash(content):
    """Compute a unique hash for content to avoid duplicates in the vector store."""
    if isinstance(content, str):
        return hashlib.sha512(content.encode('utf-8')).hexdigest()
    return None

def compute_pdf_hash(file_path):
    """Compute a hash for an entire PDF file to avoid processing duplicates."""
    hasher = hashlib.sha512()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

UNWANTED_PATTERNS = [
    r"all rights reserved",
    r"provided by ihs under license",
    r"published by .*",
    r"date of issuance:.*",
    r"page \\d+ of \\d+",
    r"asme y14\\.5.*",
    r"table of contents",
    r"annex [a-z]",
    r"mailto:.*",
    r"https?://orcid\\.org/.*",
    r"this international standard was developed.*",
    r"creative commons",
    r"ccs concepts",
    r"\\bfigure \\d+\\b",
    r"^\\s*[\\d\\.\\-]+(?:,\\s*[\\d\\.\\-]+)+\\s*$",
    r"^\\s*$"
]
MIN_WORDS = 5
MIN_CHARS = 10

def filter_text(chunks):
    filtered = []
    for chunk in chunks:
        if not chunk or len(chunk.strip()) < MIN_CHARS or len(chunk.split()) < MIN_WORDS:
            continue
        if any(re.search(pattern, chunk.strip().lower()) for pattern in UNWANTED_PATTERNS):
            continue
        filtered.append(chunk)
    return filtered

def process_text_to_vstore(text, metadata):
    """Process extracted text, chunk it, and store embeddings in the vector database."""
    if not text.strip():
        return
    
    doc_hash = compute_hash(text)
    metadata["doc_hash"] = doc_hash

    text_splitter = NLTKTextSplitter(chunk_size=2000, chunk_overlap=100, language='english')
    text_chunks = text_splitter.split_text(text)
    filtered_chunks = filter_text(text_chunks)
    documents = [Document(page_content=chunk, metadata=metadata) for chunk in filtered_chunks]
    vstore.add_documents(documents)

def extract_text_equations(file_path):
    """Extract structured text and equations from a PDF using PyMuPDF4LLM."""
    try:
        return pymupdf4llm.to_markdown(doc=file_path, page_chunks=True)
    except Exception as e:
        print(f"Error extracting text/equations: {e}")
        return []

def extract_tables_images(file_path):
    """Extracts text, tables, and images from the PDF."""
    try:
        return partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True
        )
    except Exception as e:
        print(f"Error extracting tables/images: {e}")
        return []

def process_pdf(file_path):
    """Processes a PDF, extracts text, tables, and images, and stores embeddings."""
    pdf_hash = compute_pdf_hash(file_path)
    existing_hashes = {doc.metadata.get("pdf_hash") for doc in vstore.similarity_search("", k=100)}
    if pdf_hash in existing_hashes:
        print(f" Skipping {file_path}, already processed.")
        return
    
    print(f" Processing: {file_path}")
    extracted_data = extract_tables_images(file_path)
    extracted_text = extract_text_equations(file_path)

    tables = []
    images = []
    
    for chunk in extracted_data:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        if "Image" in str(type(chunk)):
            images.append(chunk)

    # Store extracted text
    for page_number, text in enumerate(extracted_text, start=1):
        metadata = {"filename": os.path.basename(file_path), "page": page_number, "pdf_hash": pdf_hash}
        process_text_to_vstore(str(text), metadata)
        print(f" Page {page_number} - Text Stored in Vstore")
    
    # Store tables
    for table in tables:
        metadata = {"filename": os.path.basename(file_path), "page": table.metadata.page_number, "pdf_hash": pdf_hash}
        try:
            table_html = table.metadata.text_as_html
            vstore.add_documents([Document(page_content=table_html, metadata=metadata)])
            print(f" Table from Page {table.metadata.page_number} stored in Vstore")
        except Exception as e:
            print(f"Error processing table from page {table.metadata.page_number}: {e}")
    
    # Store images
    for idx, image in enumerate(images):
        metadata = {"filename": os.path.basename(file_path), "page": image.metadata.page_number, "pdf_hash": pdf_hash}
        try:
            img_b64 = image.metadata.image_base64
            if img_b64:
                vstore.add_documents([Document(page_content=img_b64, metadata=metadata)])
                print(f"Image from Page {image.metadata.page_number} stored in Vstore")
            else:
                print(f"Image from Page {image.metadata.page_number} missing Base64 data")
        except Exception as e:
            print(f"Error processing image from page {image.metadata.page_number}: {e}")

    print(f"Processing completed: {file_path}\n" + "="*100)

def main():
    for file in os.listdir(pdf_directory):
        if file.endswith(".pdf"):
            process_pdf(os.path.join(pdf_directory, file))

if __name__ == "__main__":
    main()
