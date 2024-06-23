import os
from torch import cuda, bfloat16
import transformers
import sys
from torch import cuda
import shutil



# Verileri ekleme splitleme için gerekli kütüphaneler
from langchain.chains.question_answering import load_qa_chain

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Qdrant için gerekli kütüphaneler
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models


# rag yapısı için gerekli kütüphaneler
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# data preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pdfplumber

#pip install unstructured ve pip install unstructured[pdf] indirdik

# verilerimizi yüklüyoruz

def directory(input_url):
    collection_name = "chatbotDB"
    url = "http://localhost:6333"
    print("input url :", input_url)
    # input data directory
    data_directory = os.path.join(os. getcwd(), "admin_app/admin_data" + input_url) 
    print("html url:",data_directory)
    # Sonuçları kaydedilecek dizin
    output_directory = os.path.join(os. getcwd(), "admin_app/admin_processed_data")  
    result_pdf_path=DataPreprocessing(data_directory,output_directory,input_url)
    #DataPreprocessing(data_directory, output_directory)
    EmbedData(output_directory, collection_name, url,result_pdf_path)
    
def DataLoad(pdf_dir):
    # pdf_directory = pdf_dir
    # # Dizin içindeki tüm PDF dosyalarını alın
    # pdf_files = [file for file in os.listdir(pdf_directory) if file.endswith(".pdf")]
    pdf_directory = pdf_dir
    pdf_files = os.path.join(pdf_dir, pdf_directory)
    print("data var mii", len(pdf_files))
    return pdf_files

# PDF'den metni çıkarma ve işleme
def pdf_to_text(pdf_path):
    if os.path.exists(pdf_path):
        print("YESSSSSSSSSS")
    else:
        print("NNOOOOOO")
        print(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    text_path = os.path.splitext(pdf_path)[0] + "_converted.txt"
    with open(text_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)
    return text_path

# Metin önişleme adımları
def preprocess_text(text):
    # HTML etiketlerini kaldırma
    text = re.sub('<.*?>', '', text)
    # Gereksiz karakterleri temizleme
    text = re.sub(r'\W', ' ', text)
    # Biçimlendirme işaretlerini temizleme
    text = re.sub(r'\s+', ' ', text)
    # Küçük harflere dönüştürme
    text = text.lower()
    # Tokenizasyon
    tokens = word_tokenize(text)
    # Durma kelimelerini kaldırma
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Kök çıkartma veya lemmatizasyon
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Tekrar birleştirme
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# En önemli kelimeleri bir PDF dosyasına kaydetme
def save_to_pdf(preprocessed_text, top_keywords, pdf_file_path):
    # PDF dosyasını oluştur
    doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)
    styles = getSampleStyleSheet()

    # Başlık
    title = "Processed Text"
    doc_title = Paragraph(title, styles["Title"])
    # İşlenmiş metni ekle
    processed_text = Paragraph(preprocessed_text, styles["Normal"])
    # En önemli kelimeleri başlık altında ekle
    keywords_title = Paragraph("Top Keywords", styles["Heading1"])
    keywords = [Paragraph(f"{keyword}: {score}", styles["Normal"]) for keyword, score in top_keywords]

    # Dokümana ekle
    doc.build([doc_title, processed_text, keywords_title] + keywords)


def DataPreprocessing(data_directory, output_directory, input_url):
    
    pdf_path= DataLoad(data_directory)
    print("pdf files", pdf_path)
    os.makedirs(output_directory, exist_ok=True)
    print("output directory",output_directory)

    # Her PDF dosyası için işlem yapın
    
    # # PDF dosyasının tam yolunu oluşturun
    # pdf_path = os.path.join(data_directory, pdf_file)

    # PDF'i metin dosyasına dönüştürme
    text_path = pdf_to_text(pdf_path)
   
    # Metin dosyasını okuma
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Metin önişleme
    preprocessed_text = preprocess_text(text)

    # TF-IDF vektörleme
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])

    # Kelime-frekans eşleştirmesi
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    word_frequencies = dense.tolist()[0]

    # En yüksek TF-IDF skoruna sahip kelimeleri bulma
    top_keywords = [(feature_names[i], word_frequencies[i]) for i in range(len(feature_names))]
    top_keywords.sort(key=lambda x: x[1], reverse=True)

    # İşlenmiş metni bir PDF dosyasına kaydetme
    result_pdf_path = os.path.join(output_directory + os.path.splitext(input_url)[0] + "_processed.pdf")
    print("save pdf:",result_pdf_path)
    save_to_pdf(preprocessed_text, top_keywords, result_pdf_path)

    print(f"{pdf_path} dosyası işlendi. İşlenmiş metin {result_pdf_path} dosyasına kaydedildi.")

    return result_pdf_path

# verileri bölüyoruz
def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs

def EmbedData(data_directory, collection_name, url,result_pdf_path):

    client = QdrantClient(url=url)
    # client.create_collection(
    #     collection_name=collection_name,
    #     vectors_config=models.VectorParams(size=1024, distance=models.Distance.MANHATTAN, on_disk=True),
    #     quantization_config=models.BinaryQuantization(
    #         binary=models.BinaryQuantizationConfig(
    #         # type=models.ScalarType.INT8,
    #         always_ram=True,
    #         ),
    #     ),
    # )

    # initializing the embeddings
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )
    print("bitti1******************")
    pdf_files = result_pdf_path
    # pdf_files = DataLoad(data_directory)
    print("pdf file:",pdf_files)
    print(len(pdf_files))
   
    loader = PyPDFLoader(pdf_files)
    pages = loader.load()
    docs = split_docs(pages)

    vectordb = Qdrant(client=client,collection_name=collection_name,embeddings=embed_model)
    vectordb.from_documents(
        documents=docs,
        embedding=embed_model,
        url = url,
        prefer_grpc=False,
        collection_name=collection_name,
    ) 
    print("bitti2******************")