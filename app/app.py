# Chroma compatibility issue resolution
# https://docs.trychroma.com/troubleshooting#sqlite
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from tempfile import NamedTemporaryFile

import chainlit as cl
from chainlit.types import AskFileResponse

import chromadb
from chromadb.config import Settings
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore

from prompt import EXAMPLE_PROMPT, PROMPT, WELCOME_MESSAGE


namespaces = set()


def process_file(*, file: AskFileResponse) -> list:
    if file.type != "application/pdf":
        raise TypeError("Only PDF files are supported")
    

    with NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)

        loader = PDFPlumberLoader(tempfile.name)

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=100
        )

        docs = text_splitter.split_documents(documents)

        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"

        if not docs:
            raise ValueError("PDF file parsing failed.")

        return docs


def create_search_engine(*, file: AskFileResponse) -> VectorStore:
    
    # Process and save data in the user session
    docs = process_file(file=file)
    cl.user_session.set("docs", docs)
    
    encoder = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )
    
    # Initialize Chromadb client and settings, reset to ensure we get a clean
    # search engine
    client = chromadb.EphemeralClient()
    client_settings=Settings(
        allow_reset=True,
        anonymized_telemetry=False
    )
    search_engine = Chroma(
        client=client,
        client_settings=client_settings
    )
    search_engine._client.reset()

    search_engine = Chroma.from_documents(
        client=client,
        documents=docs,
        embedding=encoder,
        client_settings=client_settings 
    )

    return search_engine


@cl.on_chat_start
async def start():

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=WELCOME_MESSAGE,
            accept=["application/pdf"],
            max_size_mb=20,
        ).send()
  
    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    try:
        search_engine = await cl.make_async(create_search_engine)(file=file)
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
        raise SystemError

    llm = ChatOpenAI(
        model='gpt-3.5-turbo-16k-0613',
        temperature=0,
        streaming=True
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=search_engine.as_retriever(max_tokens_limit=4097),

        chain_type_kwargs={
            "prompt": PROMPT,
            "document_prompt": EXAMPLE_PROMPT
        },
    )

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):

    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    response = await chain.acall(message.content, callbacks=[cb])
    answer = response["answer"]
    sources = response["sources"].strip()
    source_elements = []

    # Get the documents from the user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    # Adding sources to the answer
    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()
