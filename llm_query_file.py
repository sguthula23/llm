# %%
import os
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://optimo-openai-sbx.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "88ea0ee6c81f44dea5df38a985512575"

# %%
pip install langchain-openai

# %%
pip install langchain

# %%
pip install pypdf

# %%
pip install faiss-cpu

# %%
pip install langchain-openai

# %% [markdown]
# #### Instantiating the GPT35

# %%
import langchain_openai

# %%
pwd

# %%
import langchain_openai
from langchain_openai import AzureChatOpenAI

# %%
model = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="GPT3516K",    
       
)

# %%
model

# %%
import langchain
from langchain.schema import HumanMessage

# %% [markdown]
# #### testing Model Query and Respone

# %%
message = HumanMessage(
    content="Capital of India."
)

# %%
model([message])

# %%
# import getpass
# import os

# os.environ["OPENAI_API_KEY"] = getpass.getpass()

# %%
# import langchain_openai
# from langchain_openai import OpenAIEmbeddings

# %%
# from langchain_openai import AzureOpenAIEmbeddings

# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment="",
#     openai_api_version="2023-05-15",
# )

# %%
# text = "this is a test document"

# %%
# query_result = embeddings.embed_query(text)

# %%
# embeddings =  langchain_openai.AzureOpenAIEmbeddings()


# %% [markdown]
# #https://python.langchain.com/docs/integrations/text_embedding/openai

# %%
# pip install langchain-openai

# %%

# import langchain_openai
# from langchain_openai import OpenAIEmbeddings


# %%
# from langchain_community.embeddings.openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(
#     deployment="text-embedding-ada-002-new",
#     model="text-embedding-ada-002-new",
#     openai_api_base="https://your-endpoint.openai.azure.com/",
#     openai_api_type="azure",
# )
# text = "This is a test query."
# query_result = embeddings.embed_query(text)[*Deprecated*] OpenAI embedding models.

# %%
# import os
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://optimo-openai-sbx.openai.azure.com/"
# os.environ["AZURE_OPENAI_API_KEY"] = "88ea0ee6c81f44dea5df38a985512575"

# %%
!pip install langchain_community

# %% [markdown]
# #### Creating instance of AzureEmbedddings

# %%
from langchain_openai import AzureOpenAIEmbeddings

openai_embedding = AzureOpenAIEmbeddings(model="text-embedding-ada-002", deployment="text-embedding-ada-002-new", openai_api_key="88ea0ee6c81f44dea5df38a985512575" )

# %%
text = "This is a test query."
query_result = openai_embedding.embed_query(text)

# %%
# # from langchain_community.embeddings.openai import OpenAIEmbeddings
# import langchain_community
# embeddings = langchain_community.embeddings.openai.OpenAIEmbeddings(
#     deployment="text-embedding-ada-002-new",
#     model="text-embedding-ada-002",
#     openai_api_base="https://optimo-openai-sbx.openai.azure.com/",
#     openai_api_type="azure",
#     openai_api_key="88ea0ee6c81f44dea5df38a985512575"
# )
# text = "This is a test query."
# query_result = embeddings.embed_query(text)

# %%
# # from langchain_community.embeddings.openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(
#     deployment="text-embedding-ada-002-new",
#     model="text-embedding-ada-002-new",
#     openai_api_key="88ea0ee6c81f44dea5df38a985512575"
# )
# text = "This is a test query."
# query_result = embeddings.embed_query(text)

# %%
# from langchain.document_loaders import PyPDFDirectoryLoader

# %%
# pip install pypdf

# %%
 !pwd

# %%


# %%


# %%
# Users/sguthula/.pdf

# %% [markdown]
# ##### Reading a document and creating Chunks

# %%
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
directory_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/sguthula1/code/Users/sguthula/Documents"
loader_docx = DirectoryLoader(directory_path, glob="**/*.docx", loader_cls=Docx2txtLoader)

# %%
data_docx=loader_docx.load_and_split()

# %%
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
directory_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/sguthula1/code/Users/sguthula"
loader_pdf = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)

# %%
data_pdf = loader_pdf.load_and_split()

# %%
pages = data_docx+data_pdf

# %%
from langchain_community.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
# Create the TextLoader object using the file path
path= "/mnt/batch/tasks/shared/LS_root/mounts/clusters/sguthula1/code/Users/sguthula"
loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
#loader.load()

# %%


# %%
# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("/mnt/batch/tasks/shared/LS_root/mounts/clusters/sguthula1/code/Users/sguthula/Change_request.pdf")
# # docs = loader.load()
# pages = loader.load_and_split()

# %%
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings

# faiss_index = FAISS.from_documents(pages, openai_embedding())
 # docs = faiss_index.similarity_search("How will the community be engaged?", k=2)
 # for doc in docs:
 #     print(str(doc.metadata["page"]) + ":", doc.page_content[:300])

# %% [markdown]
# #### sett up Meta FAISS as Vector DB Store

# %%
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(pages, openai_embedding)

# %%
db_retreiver = vectorstore.as_retriever()

# %% [markdown]
# ##### Retreiving the simlar chinks available in VectorDB based on input search

# %%
# Retreiving the relevant chucks for the query
db_retreiver.get_relevant_documents("")

# %% [markdown]
# #### Creating a conversational chain

# %%
from langchain.chains import ConversationChain
from langchain_community.llms import OpenAI

conversation = ConversationChain(llm=model)

# %%
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# %%
memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )

# %%
# chat_history = []

# # query = "buero  of labour experience year for swathi guthula"
# query = " swathi guthula highest degree of education"
# result = conversation_chain({"question": query, "chat_history": chat_history})

# print(result['answer'])

# %%
# chat_history = []

# query = "who is Oliver the owl in the story"

# result = conversation_chain({"question": query, "chat_history": chat_history})

# print(result['answer'])

# %%
from InputQuery import input_Query
query1 = input_Query()

# %%
chat_history = []
#query = "where Steve job when to  5th grade?"
query = query1
result = conversation_chain({"question": query, "chat_history": chat_history})

print(result['answer'])

response = result['answer']

# %%
def response_output():
    return response

# %%
response_output()

# %%
class llm_query:
    def response_output():
        return response 

# %%



