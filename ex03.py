import os
from dotenv import load_dotenv

# โหลด API Key จากไฟล์ .env
load_dotenv()

# ตรวจสอบว่ามี OpenAI API Key หรือไม่
# โปรแกรมเช็คก่อนว่ามี "กุญแจ" (API Key) สำหรับใช้บริการ AI ของ OpenAI หรือยัง (เหมือนต้องมีบัตรผ่านเพื่อเข้าใช้บริการ) ถ้าไม่มีก็จะบอกให้ไปตั้งค่าก่อน
if os.getenv("OPENAI_API_KEY") is None:
    print("กรุณาตั้งค่า OPENAI_API_KEY ในไฟล์ .env ก่อน")
    exit()

# โหลด Library ต่างๆ ที่จำเป็นสำหรับการทำงาน
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# --- 1. โหลดและแบ่งเอกสาร PDF ---

# ระบุตำแหน่งไฟล์ PDF (ตรวจสอบให้แน่ใจว่า path ถูกต้อง)
pdf_path = "D:/doc.pdf" 

# ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
if not os.path.exists(pdf_path):
    print(f"ไม่พบไฟล์ PDF ที่: {pdf_path}")
    exit()

try:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"โหลดเอกสาร PDF สำเร็จ มี {len(documents)} หน้า")

    # แบ่งเอกสารเป็น Chunks เล็กๆ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"แบ่งเอกสารออกเป็น {len(texts)} ส่วน (chunks)")

except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดหรือแบ่ง PDF: {e}")
    exit()

# --- 2. สร้าง Embeddings และ Vector Store (FAISS) ---

try:
    # สร้าง Embeddings โดยใช้ OpenAI
    embeddings = OpenAIEmbeddings()

    # สร้าง FAISS Vector Store จาก texts และ embeddings และเก็บไว้ใน memory
    # FAISS.from_documents จะสร้าง index ใน memory โดยอัตโนมัติ
    db = FAISS.from_documents(texts, embeddings)
    print("สร้าง FAISS Vector Store ใน Memory สำเร็จ")

    # (ทางเลือก) บันทึก index ลงดิสก์เผื่อใช้ภายหลัง
    # db.save_local("faiss_index_doc")
    # print("บันทึก FAISS index ลงดิสก์แล้ว (ที่โฟลเดอร์ faiss_index_doc)")

except Exception as e:
    print(f"เกิดข้อผิดพลาดในการสร้าง Embeddings หรือ FAISS Vector Store: {e}")
    exit()

# --- 3. สร้าง RetrievalQA Chain ---

try:
    # สร้าง Retriever จาก Vector Store
    retriever = db.as_retriever(search_kwargs={"k": 3}) # ดึงเอกสารที่เกี่ยวข้อง 3 ชิ้น

    # เลือกโมเดล LLM (ในที่นี้คือ ChatOpenAI)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # หรือ gpt-4

    # สร้าง RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # หรือ "map_reduce", "refine", "map_rerank"
        retriever=retriever,
        return_source_documents=True # (ทางเลือก) แสดงเอกสารต้นฉบับที่ใช้ตอบ
    )
    print("สร้าง RetrievalQA Chain สำเร็จ")

except Exception as e:
    print(f"เกิดข้อผิดพลาดในการสร้าง RetrievalQA Chain: {e}")
    exit()

# --- 4. ทดสอบถามคำถาม ---

print("\n--- เริ่มการถามคำถาม ---")
while True:
    query = input("ป้อนคำถามของคุณ (หรือพิมพ์ 'exit' เพื่อออก): ")
    if query.lower() == 'exit':
        break
    if not query:
        continue

    try:
        # ส่งคำถามไปยัง Chain
        result = qa_chain.invoke({"query": query})

        # แสดงผลลัพธ์
        print("\nคำตอบ:")
        print(result["result"])
        print("-" * 30)

    except Exception as e:
        print(f"เกิดข้อผิดพลาดระหว่างการถามคำถาม: {e}")

print("จบการทำงาน")