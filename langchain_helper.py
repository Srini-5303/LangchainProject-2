from langchain_community.llms import GooglePalm
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
#from chroma.config import Settings
from langchain.vectorstores import FAISS
import os
from few_shots import few_shots
import pickle


# from dotenv import load_dotenv
# load_dotenv()

file_path = "faiss_index_vector.pkl"


API_KEY = "yourOpenAIKey"
llm = GooglePalm(google_api_key = API_KEY, temperature = 0.2)

def get_few_shot_db_chain():

    db_user = "root"
    db_password = "Srini%405303"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info=3)

    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    to_vectorize = [" ".join([str(value) for value in example.values()]) for example in few_shots]

    vectorstore_faiss = FAISS.from_texts(to_vectorize, embeddings, metadatas = few_shots)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_faiss, f)
        
    with open(file_path,"rb") as f:
        vectorstore = pickle.load(f)

    
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore = vectorstore,
        k = 2
    )

    example_prompt = PromptTemplate(
        input_variables = ["Question","SQLQuery","SQLResult","Answer"],
        template = "\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}"
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector = example_selector,
        example_prompt = example_prompt,
        prefix = _mysql_prompt,
        suffix = PROMPT_SUFFIX,
        input_variables =  ["input","table_info","top_k"],
    )

    chain = SQLDatabaseChain.from_llm(llm, db, verbose = True, prompt = few_shot_prompt)

    return chain



