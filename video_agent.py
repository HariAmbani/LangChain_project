from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

#FAISS - A library for efficient similarity search and clustering of dense vectors.
#FAISS is a vector database that allows you to store and query embeddings efficiently.
#alternative for vector db (FAISS) - Pinecone, Weaviate, Milvus, Chroma, etc. 

load_dotenv()

embeddings = OpenAIEmbeddings()

video_url = "https://www.youtube.com/watch?v=lG7Uxts9SXs"

def create_vectorDB_from_youtube_url(video_url):
    loader = YoutubeLoader.from_youtube_url(
        video_url,
        add_video_info=True
    )
    transcripts = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(transcripts)
    db = FAISS.from_documents(docs, embeddings)

    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([doc.page_content for doc in docs])

    llm = OpenAI(model="text-davinci-003", temperature=0.5)

    prompt_template = PromptTemplate(
        input_variables=["query", "docs"],
        template="""
        you are a helpful assistant for answering questions based on the videos transcript.
        
        answer the following question : {question} based on the following transcript of the video by searched in the {docs}. 
        
        if you don't know the answer, say you don't know.
        
        your answer should be based on the above transcript : {docs} and should be concise and to the point and detailed too.
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({"question": query, "docs": docs_page_content})
    response = response.replace("\n", "")
    return response

#print(create_vectorDB_from_youtube_url(video_url))pip install langchain

 
    
    
    