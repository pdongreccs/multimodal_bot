from dotenv import load_dotenv
import os
import json
import hashlib
import base64
import boto3
import fitz  # pymupdf
import tabula
import numpy as np
import faiss
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from botocore.exceptions import ClientError
import streamlit as st

# === Load .env and AWS credentials ===
load_dotenv()

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_DEFAULT_REGION")

session = boto3.Session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)

# === Constants ===
SAVE_DIR = "stored_indices"
os.makedirs(SAVE_DIR, exist_ok=True)


# === File utilities ===
def compute_file_hash(file_or_bytes):
    if isinstance(file_or_bytes, bytes):
        file_bytes = file_or_bytes
    elif hasattr(file_or_bytes, "read"):
        file_or_bytes.seek(0)
        file_bytes = file_or_bytes.read()
    else:
        with open(file_or_bytes.name, "rb") as f:
            file_bytes = f.read()
    return hashlib.md5(file_bytes).hexdigest()


def create_directories(base_dir):
    for folder in ["images", "text", "tables", "page_images"]:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)


# === Index Save/Load ===
def save_index(filename, index, items, suffix):
    serializable_items = []
    for item in items:
        item_copy = item.copy()
        if "embedding" in item_copy and isinstance(item_copy["embedding"], np.ndarray):
            item_copy["embedding"] = item_copy["embedding"].tolist()
        serializable_items.append(item_copy)

    faiss.write_index(index, f"{SAVE_DIR}/{filename}_{suffix}.index")
    with open(f"{SAVE_DIR}/{filename}_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(serializable_items, f, ensure_ascii=False)



def load_index(filename, suffix):
    index_path = f"{SAVE_DIR}/{filename}_{suffix}.index"
    items_path = f"{SAVE_DIR}/{filename}_{suffix}.json"
    if not os.path.exists(index_path) or not os.path.exists(items_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(items_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        if "embedding" in item:
            item["embedding"] = np.array(item["embedding"], dtype=np.float32)
    return index, items


# === PDF Processing ===
# def process_tables(doc, page_num, base_dir, items, filename):
#     try:
#         tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True)
#         if not tables:
#             return
#         for table_idx, table in enumerate(tables):
#             table_text = "\n".join([" | ".join(map(str, row)) for row in table.values])
#             table_file = f"{base_dir}/tables/{filename}_table_{page_num}_{table_idx}.txt"
#             with open(table_file, 'w', encoding='utf-8') as f:
#                 f.write(table_text)
#             items.append({"page": page_num, "type": "table", "text": table_text, "path": table_file})
#     except Exception as e:
#         print(f"Error extracting tables from page {page_num}: {str(e)}")


def process_text_chunks(text, splitter, page_num, base_dir, items, filename):
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        file = f"{base_dir}/text/{filename}_text_{page_num}_{i}.txt"
        with open(file, 'w', encoding='utf-8') as f:
            f.write(chunk)
        items.append({"page": page_num, "type": "text", "text": chunk, "path": file})


def process_images(page, page_num, base_dir, items, filename):
    images = page.get_images()
    for idx, image in enumerate(images):
        try:
            xref = image[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n >= 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            file = f"{base_dir}/images/{filename}_image_{page_num}_{idx}_{xref}.png"
            pix.save(file)
            if os.path.getsize(file) == 0:
                print(f"Skipping empty image file: {file}")
                continue
            try:
                with Image.open(file) as img:
                    img.verify()
            except Exception as e:
                print(f"Skipping invalid image file: {file} — {e}")
                continue
            with open(file, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf8')
            if encoded:
                items.append({"page": page_num, "type": "image", "path": file, "image": encoded})
        except Exception as e:
            print(f"Error processing image on page {page_num}, index {idx}: {e}")


def process_page_images(page, page_num, base_dir, items, filename):
    try:
        pix = page.get_pixmap()
        if pix.n >= 5:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        file = os.path.join(base_dir, f"page_images/{filename}_page_{page_num:03d}.png")
        pix.save(file)
        if os.path.getsize(file) == 0:
            print(f"Skipping empty page image: {file}")
            return
        try:
            with Image.open(file) as img:
                img.verify()
        except Exception as e:
            print(f"Skipping invalid page image: {file} — {e}")
            return
        with open(file, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        if image_b64:
            items.append({"page": page_num, "type": "page", "path": file, "image": image_b64})
    except Exception as e:
        print(f"Error processing full-page image on page {page_num}: {e}")


# def process_uploaded_pdf(pdf_file, base_dir="data"):
#     global filepath, doc

#     if hasattr(pdf_file, 'read'):
#         pdf_bytes = pdf_file.read()
#         original_name = pdf_file.name if hasattr(pdf_file, "name") else "uploaded.pdf"
#         filename = os.path.splitext(os.path.basename(original_name))[0]
#         filepath = os.path.join(base_dir, f"{filename}.pdf")
#         with open(filepath, "wb") as f:
#             f.write(pdf_bytes)
#         doc = fitz.open(filepath)
#     else:
#         filepath = pdf_file.name
#         filename = os.path.splitext(os.path.basename(filepath))[0]
#         doc = fitz.open(pdf_file.name)

#     create_directories(base_dir)
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
#     items = []

#     for page_num in tqdm(range(len(doc)), desc="Processing PDF pages"):
#         page = doc[page_num]
#         text = page.get_text()
#         process_text_chunks(text, text_splitter, page_num, base_dir, items, filename)
#         # process_tables(doc, page_num, base_dir, items, filename)
#         process_images(page, page_num, base_dir, items, filename)
#         process_page_images(page, page_num, base_dir, items, filename)

#     return filename, items

def process_uploaded_pdf(pdf_file, filename, base_dir="data"):
    global filepath, doc

    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, f"{filename}.pdf")

    # Handle BytesIO or file-like objects
    if hasattr(pdf_file, 'read'):
        pdf_bytes = pdf_file.read()
        with open(filepath, "wb") as f:
            f.write(pdf_bytes)
        doc = fitz.open(filepath)
    else:
        filepath = pdf_file.name
        doc = fitz.open(filepath)

    create_directories(base_dir)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
    items = []

    for page_num in tqdm(range(len(doc)), desc="Processing PDF pages"):
        page = doc[page_num]
        text = page.get_text()

        # process_tables(doc, page_num, base_dir, items, filename)
        process_text_chunks(text, text_splitter, page_num, base_dir, items, filename)
        process_images(page, page_num, base_dir, items, filename)
        process_page_images(page, page_num, base_dir, items, filename)

    return filename, items

# === Titan Embeddings ===
def generate_multimodal_embeddings(prompt=None, image=None, output_embedding_length=384):
    if not prompt and not image:
        raise ValueError("Need prompt or image")

    client = session.client("bedrock-runtime")
    model_id = "amazon.titan-embed-image-v1"

    body = {"embeddingConfig": {"outputEmbeddingLength": output_embedding_length}}
    if prompt: body["inputText"] = prompt
    if image: body["inputImage"] = image

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        return result.get("embedding")
    except ClientError as e:
        print("Titan error:", e.response["Error"]["Message"])
        return None


def process_multimodal_embeddings(items):
    text_items = [i for i in items if i["type"] in ["text", "table"]]
    image_items = [i for i in items if i["type"] == "image"]
    vector_dim = 384

    with tqdm(total=len(items), desc="Generating embeddings") as pbar:
        for item in items:
            try:
                if item["type"] in ["text", "table"]:
                    item["embedding"] = generate_multimodal_embeddings(prompt=item["text"], output_embedding_length=vector_dim)
                elif item["type"] == "image" and item.get("image"):
                    item["embedding"] = generate_multimodal_embeddings(image=item["image"], output_embedding_length=vector_dim)
                else:
                    continue
            
            except Exception as e:
                print("Embedding error:", e)
                item["embedding"] = None
            pbar.update(1)

    # Only include items with valid embeddings
    text_items = [i for i in text_items if isinstance(i.get("embedding"), list) or isinstance(i.get("embedding"), np.ndarray)]
    image_items = [i for i in image_items if isinstance(i.get("embedding"), list) or isinstance(i.get("embedding"), np.ndarray)]
    
    text_vecs = np.array([i["embedding"] for i in text_items], dtype=np.float32)
    image_vecs = np.array([i["embedding"] for i in image_items], dtype=np.float32)

    text_index = faiss.IndexFlatL2(vector_dim)
    image_index = faiss.IndexFlatL2(vector_dim)
    text_index.add(text_vecs)
    image_index.add(image_vecs)

    return text_index, image_index

# === Claude Response ===
def invoke_claude3_sonnet_with_images(prompt, matched_items, chat_history=None):
    client = session.client("bedrock-runtime")

    blocks = [{
        "type": "text",
        "text": "You are a helpful assistant for answering user questions based on retrieved context."
    }]

    # Add last 5 message pairs (user + assistant) if available
    if chat_history:
        for q, a in chat_history[-5:]:
            blocks.append({"type": "text", "text": f"User: {q}"})
            blocks.append({"type": "text", "text": f"Assistant: {a}"})

    # Add matched context
    for item in matched_items:
        # # get metadata
        # page = item.get("page", "?")
        # item_type = item.get("type", "").capitalize()

        if item["type"] in ["text", "table"]:
            text = item.get("text", "").strip()
            if text:
                blocks.append({"type": "text", "text": text[:2000]})
        elif item["type"] == "image":
            img_b64 = item.get("image")
            if not img_b64 or not isinstance(img_b64, str):
                continue  # Skip invalid image field

            try:
                # Validate that it's decodable
                base64.b64decode(img_b64)

                # blocks.append({"type": "text", "text": f"[{item_type} — Page {page}]\n\n[Image shown below]"})
                
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64
                    }
                })
            
            except Exception as e:
                print(f"Skipping bad image (page {item.get('page', '?')}): {e}")
                continue

    # Add the user prompt
    # blocks.append({"type": "text", "text": prompt})
    blocks.append({"type": "text", "text": f"User: {prompt}"})

    # Validate: at least one meaningful text block must be present
    valid_text_blocks = [b for b in blocks if b["type"] == "text" and b["text"].strip()]
    if not valid_text_blocks:
        print("No valid text content available for Claude API.")
        return "Sorry, no valid context could be used to answer your question."

    # Claude API call payload
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": blocks}],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    try:
        response = client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        print("Claude API invocation failed:", e)
        return "Error occurred while generating the answer. Please try again."

# === Query Handling ===
def is_meta_query(query: str) -> str:
    query = query.lower()
    if any(kw in query for kw in ["summarize", "summary", "describe", "overview", "gist", "main idea"]):
        return "summary"
    return None

def answer_query(query, text_items, image_items, text_index, image_index, chat_history=None):
    query_type = is_meta_query(query)

    if query_type == "summary":
        # Concatenate the top N chunks for summarization
        context_blocks = [i["text"] for i in text_items[:10] if i.get("text")]
        merged_text = "\n\n".join(context_blocks)
        return invoke_claude3_sonnet_with_images(f"Give a summary of the document below:\n{merged_text}", [])

    # --- Else: do standard retrieval ---    
    vector_dim = 384
    q_vec = np.array(generate_multimodal_embeddings(prompt=query, output_embedding_length=vector_dim), dtype=np.float32).reshape(1, -1)

    img_dists, img_results = image_index.search(q_vec, k=3)
    txt_dists, txt_results = text_index.search(q_vec, k=3)

    matched = []
    matched += [{k: v for k, v in image_items[i].items() if k != 'embedding'} for i in img_results[0]]
    matched += [{k: v for k, v in text_items[i].items() if k != 'embedding'} for i in txt_results[0]]

    return invoke_claude3_sonnet_with_images(query, matched, chat_history)

# === Streamlit UI ===
st.set_page_config(page_title="Multimodal PDF Chatbot")
st.title("Multimodal PDF Q&A Assistant")

# === Initialize session state ===
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = {}  # filename -> dict with index, items, history
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None
if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# === Upload and Process PDF ===
if uploaded_file:
    original_name = uploaded_file.name
    filename = os.path.splitext(os.path.basename(original_name))[0]

    # Detect new upload
    is_new_upload = filename != st.session_state.last_uploaded_filename

    if is_new_upload:
        st.session_state.last_uploaded_filename = filename

        with st.spinner("Processing PDF..."):
            pdf_bytes = uploaded_file.read()

            # Save PDF locally
            os.makedirs("data", exist_ok=True)
            local_path = os.path.join("data", original_name)
            with open(local_path, "wb") as f:
                f.write(pdf_bytes)

            # Try loading existing FAISS index
            text_index, text_items = load_index(filename, "text")
            image_index, image_items = load_index(filename, "image")

            if not (text_index and image_index and text_items and image_items):
                _, items = process_uploaded_pdf(BytesIO(pdf_bytes), filename=filename)
                text_index, image_index = process_multimodal_embeddings(items)
                text_items = [i for i in items if i["type"] in ["text", "table"]]
                image_items = [i for i in items if i["type"] == "image"]
                save_index(filename, text_index, text_items, "text")
                save_index(filename, image_index, image_items, "image")

            # Save to session state with fresh chat
            st.session_state.pdf_data[filename] = {
                "text_index": text_index,
                "image_index": image_index,
                "text_items": text_items,
                "image_items": image_items,
                "chat_history": []
            }

            # Switch sidebar to the new PDF
            st.session_state.selected_pdf = filename

# === Sidebar PDF Selection ===
st.sidebar.title("Uploaded PDFs")
available_pdfs = list(st.session_state.pdf_data.keys())
if available_pdfs:
    selected_pdf = st.sidebar.radio(
        "Select PDF:", available_pdfs,
        index=available_pdfs.index(st.session_state.selected_pdf) if st.session_state.selected_pdf in available_pdfs else 0
    )
    st.session_state.selected_pdf = selected_pdf
    selected_data = st.session_state.pdf_data[selected_pdf]
else:
    selected_pdf, selected_data = None, None

# === Chat Interface ===
if selected_data:
    st.markdown(f"### Ask about: `{selected_pdf}.pdf`")

    if st.button("Clear Chat"):
        selected_data["chat_history"] = []

    user_input = st.chat_input("Ask a question about the PDF...")
    
    if user_input:
        with st.spinner("Generating answer..."):
            answer = answer_query(
                user_input,
                selected_data["text_items"],
                selected_data["image_items"],
                selected_data["text_index"],
                selected_data["image_index"],
                chat_history=selected_data["chat_history"]
            )
            
            # Save Q&A pair
            selected_data["chat_history"].append((user_input, answer))

    for q, a in selected_data["chat_history"]:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
else:
    st.info("Upload a PDF to begin.")