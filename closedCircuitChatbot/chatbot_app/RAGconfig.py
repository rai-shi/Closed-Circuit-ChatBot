import os
from torch import cuda, bfloat16
import transformers
import sys
from torch import cuda
import torch
import gc

from django.conf import settings

# Verileri ekleme splitleme için gerekli kütüphaneler
from langchain.chains.question_answering import load_qa_chain

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Qdrant için gerekli kütüphaneler
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient


# rag yapısı için gerekli kütüphaneler
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import time 



from langfuse.callback import CallbackHandler
from langfuse import Langfuse
from langsmith import Client
import langsmith 
import concurrent.futures
from multiprocessing import Process, Pipe

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from uuid import uuid4

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"  # Update to your API key

# Used by the agent in this tutorial
# os.environ["OPENAI_API_KEY"] = "<YOUR-OPENAI-API-KEY>"

class RAG():

    def __init__(self):
        print("*******************************RAG CONFIG START *******************************")
        self.embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        # print("device: ", device)
        self.initializeEmbeddingModel(device)        

        self.model_id = 'meta-llama/Llama-2-13b-chat-hf'
        self.hf_auth = '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' #1. token
        # self.hf_auth = "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" # 2. token

        # localden almak için LLM klasörü altına eklendiler
        # self.llm_path = "/home/ktu/Masaüstü/Closed-Circuit-ChatBot/closedCircuitChatbot/LLM/Llama-2-7b-chat-hf"
        # self.llm_path = os.path.join(settings.BASE_DIR, "LLM/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590" ) 
        # print("llm_path = ",self.llm_path)
        self.initializeLLM()

        self.data_collection_name = "chatbotDB"
        self.db_url = "http://localhost:6333"
        self.initializeDBclient()
        self.initializesearchDB()

        self.LLMpipeline()
        self.configSystemPrompt()
        self.RAGpipeline()
        

        self.chat_history = []

        

        print("*******************************RAG CONFIG DONE *******************************")
    
    def initializeEmbeddingModel(self, device):
        # print("******************************* initializeEmbeddingModel *******************************")
        
        # initializing the embeddings
        # self.embed_model = HuggingFaceEmbeddings(
        #     model_name=self.embed_model_id,
        #     model_kwargs={'device': device},
        #     encode_kwargs={'device': device, 'batch_size': 32}
        # )
        self.embed_model = FastEmbedEmbeddings()
        # print("******************************* embed model started *******************************")
    
    def initializeLLM(self):
        # print("******************************* initializeLLM *******************************")
        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        # begin initializing HF items, need auth token for these
        model_config = transformers.AutoConfig.from_pretrained(
            self.model_id,
            # use_auth_token=self.hf_auth
            token=self.hf_auth
        )
        self.LLMmodel = transformers.AutoModelForCausalLM.from_pretrained(
            # pretrained_model_name_or_path= self.llm_path, # eklendi
            self.model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            # device_map = {"":0},
            # use_safetensors = True,
            token=self.hf_auth # bu şekilde ileriki versiyonda kaldırılacak uyarısı vermiyor
            #use_auth_token=self.hf_auth
        )
        self.LLMmodel.eval()
        # print("******************************* llm started *******************************")

    # veri tabanı erişimi için client oluşturuluyor
    def initializeDBclient(self):
        # print("******************************* initializeDBclient *******************************")
        
        self.DBclient = QdrantClient(
                        url = self.db_url,
                        prefer_grpc=False
                        )
        # print("******************************* dbclient started *******************************")
        
    #  veri tabanında arama yapmak için db nesnei oluşturuldu
    def initializesearchDB(self):
        # print("******************************* initializesearchDB *******************************")
        
        self.searchDB = Qdrant(
                    client = self.DBclient,
                    embeddings = self.embed_model,
                    collection_name = self.data_collection_name 
                )
        # print("******************************* searchdb started *******************************")
        
    def LLMpipeline(self):
        # print("******************************* LLMpipeline *******************************")
        
        # llm nesnesi için gerekli bir değişken 
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.model_id,
                    token=self.hf_auth
                )
        # sadece llm için bir pipeline
        pipeline = transformers.pipeline(
                    model=self.LLMmodel, tokenizer=tokenizer,
                    return_full_text=False,  # langchain expects the full text
                    task='text-generation',
                    # we pass model parameters here too
                    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                    max_new_tokens=512,  # mex number of tokens to generate in the output
                    repetition_penalty=1.1,
                    do_sample=False
                    # without this output begins repeating
                )
        self.llm = HuggingFacePipeline(pipeline=pipeline)
        # print("******************************* llm pipeline is done *******************************")
        
    def configSystemPrompt(self):
        # print("******************************* configSystemPrompt *******************************")
        
        # sohbet geçmişi için alt prompt ve genel system promptu tanımlıyoruz

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""

        # prompt = PromptTemplate.from_template(template)
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),                
            ], 
        )


        # print("******************************* system prompt is done *******************************")


    def RAGpipeline(self):
        # print("******************************* RAGpipeline *******************************")
        
        retriever = self.searchDB.as_retriever(search_kwargs={"k":4})

        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, self.contextualize_q_prompt
        )

        # alt zincir
        question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        # ana zincir
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        print("******************************* rag pipeline is done *******************************")

    

    def ragQA(self, question, history):
        # SystemMessagePromptTemplate
        st=time.time()
        print(history)
        ai_msg = self.rag_chain.invoke({"input": question, "chat_history": history})
        # self.chat_history.extend([HumanMessage(content=question), ai_msg["answer"]])   
          
        print("AI MSG:\n",ai_msg)   
        et=time.time()
        print("sure:",(et-st))
        return ai_msg

    def updateChatHistory(self, chat:list):
        pass

    def clear_cache(self):
        torch.cuda.empty_cache()
        gc.collect()
