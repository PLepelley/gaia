
#load & split data
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
# embed data
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# vector store
from langchain_community.vectorstores import FAISS
# prompt
from langchain.prompts import PromptTemplate
# memory
from langchain.memory import ConversationBufferMemory
#llm
from langchain_mistralai.chat_models import ChatMistralAI

#chain modules
from langchain.chains import RetrievalQA
from langchain.chains import create_sql_query_chain

from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter


# import PyPDF2
import os
import re
from dotenv import load_dotenv
load_dotenv()
from collections import defaultdict


class RagModule():
    def __init__(self):
        self.mistral_api_key = os.environ['MISTRAL_API_KEY']
        self.model_name_embedding = "mistral-embed"
        self.embedding_model = MistralAIEmbeddings(model=self.model_name_embedding, mistral_api_key=self.mistral_api_key)
        self.chunk_size = 1000
        self.chunk_overlap = 120
        self.text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.db_faiss_path = "../data/vector_store"
        #params llm
        self.llm_model = "mistral-small"
        self.max_new_tokens = 512
        self.top_p = 0.5
        self.temperature = 0.1
    

    def read_pdf_file(self, file):
        """read pdf file

        Args:
            file (_type_): _description_

        Returns:
            _type_: _description_
        """
        pdf = PyPDF2.PdfReader(file.path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
        
        return pdf_text

    def split_text(self, text:str) -> list:
        """Split the text into chunk

        Args:
            text (str): _description_

        Returns:
            list: _description_
        """
        texts = self.text_splitter.split_text(text)
        return texts
    
    def get_metadata(self, texts:list) -> list:
        """_summary_

        Args:
            texts (list): _description_

        Returns:
            list: _description_
        """
        metadatas = [{"source": f'Paragraphe: {i}'} for i in range(len(texts))]
        return metadatas
    
    def get_faiss_db(self):
        """load local faiss vector store containing all embeddings 

        """
        db = FAISS.load_local(self.db_faiss_path, self.embedding_model)
        return db

    def set_custom_prompt(self, prompt_template:str):
        """Instantiate prompt template for Q&A retreival for each vectore stores

        Args:
            prompt_template (str): description of the prompt
            input_variables (list): variables in the prompt
        """
        prompt = PromptTemplate.from_template(
            template=prompt_template,
            )
    
        return prompt
    
    def load_mistral(self):
        """instantiate LLM
        """

        model_kwargs = {
        "mistral_api_key": self.mistral_api_key,
        "model": self.llm_model,
        "max_new_tokens": self.max_new_tokens,
        "top_p": self.top_p,
        "temperature": self.temperature,
        }

        llm = ChatMistralAI(**model_kwargs)
        
        return llm

    def retrieval_qa_memory_chain(self, db, prompt_template):
        """_summary_
        """
        llm = self.load_mistral()
        prompt = self.set_custom_prompt(prompt_template)
        memory = ConversationBufferMemory(
            memory_key = 'history',
            input_key = 'question'
        )
        chain_type_kwargs= {
            "prompt" : prompt,
            "memory" : memory
            }
        
        qa_chain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type = 'stuff',
            retriever = db.as_retriever(search_kwargs={"k":5}),
            chain_type_kwargs = chain_type_kwargs,
            return_source_documents = True,
            )

        return qa_chain

    def retrieval_qa_chain(self, db, prompt_template):
        """_summary_
        """
        llm = self.load_llm()
        prompt = self.set_custom_prompt(prompt_template)
       
        chain_type_kwargs= {
            "prompt" : prompt,
            }
        
        qa_chain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type = 'stuff',
            retriever = db.as_retriever(search_kwargs={"k":3}),
            chain_type_kwargs = chain_type_kwargs,
            return_source_documents = True,
            )

        return qa_chain
    
        
    def sql_query_chain(self, db):
        llm = self.load_llm()
        sql_chain = create_sql_query_chain(
                llm = llm,
                db = db,
                )
        
        return sql_chain
    
    def sql_chain(self, db, prompt_template:str):

        llm=self.load_llm()
        prompt = self.set_custom_prompt(prompt_template)

        write_query = create_sql_query_chain(llm, db)
        execute_query = QuerySQLDataBaseTool(db=db)
        answer = prompt | llm | StrOutputParser()
        chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
                )
                | answer
            )

        return chain
    
    def get_sources_document(self, source_documents:list) -> dict:
        """generate dictionnary with path (as a key) and list of pages associated to one path

        Args:
            source_document (list): list of documents containing source_document of rag response

        Returns:
            dict: {
                path/to/file1 : [0, 1, 3],
                path/to/file2 : [5, 2]
                }
        """
        sources = defaultdict(list)
        for doc in source_documents:
            sources[doc.metadata["source"]].append(doc.metadata["page"])
        
        return sources

    def shape_answer_with_source(self, answer: str, sources: dict):
        """_summary_

        Args:
            answer (str): _description_
            source (dict): _description_
        """
        pattern = r"^(.+)\/([^\/]+)$"
        
        source_msg = ""
        for path, page in sources.items():
            file = re.findall(pattern, path)[0][1]
            source_msg += f"\nFichier: {file} - Page: {page}"
    
        answer += f"\n{source_msg}"

        return answer