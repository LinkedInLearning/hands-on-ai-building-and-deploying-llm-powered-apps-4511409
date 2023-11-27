# Chroma compatibility issue resolution
# https://docs.trychroma.com/troubleshooting#sqlite
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from tempfile import NamedTemporaryFile
from typing import List

import chainlit as cl
from chainlit.types import AskFileResponse
import chromadb
from chromadb.config import Settings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore

from prompt import EXAMPLE_PROMPT, PROMPT


def process_file(*, file: AskFileResponse) -> List[Document]:
    """Processes one PDF file from a Chainlit AskFileResponse object by first
    loading the PDF document and then chunk it into sub documents. Only
    supports PDF files.

    Args:
        file (AskFileResponse): input file to be processed

    Raises:
        ValueError: when we fail to process PDF files. We consider PDF file
        processing failure when there's no text returned. For example, PDFs
        with only image contents, corrupted PDFs, etc.

    Returns:
        List[Document]: List of Document(s). Each individual document has two
        fields: page_content(string) and metadata(dict).
    """
    if file.type != "application/pdf":
        raise TypeError("Only PDF files are supported")

    with NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)

        loader = PDFPlumberLoader(tempfile.name)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

        # Adding source_id into the metadata to denote which document it is
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"

        if not docs:
            raise ValueError("PDF file parsing failed.")

        return docs


def create_search_engine(
    *, docs: List[Document], embeddings: Embeddings
) -> VectorStore:
    """Takes a list of Langchain Documents and an embedding model API wrapper
    and build a search index using a VectorStore.

    Args:
        docs (List[Document]): List of Langchain Documents to be indexed into
        the search engine.
        embeddings (Embeddings): encoder model API used to calculate embedding

    Returns:
        VectorStore: Langchain VectorStore
    """
    # Initialize Chromadb client to enable resetting and disable telemtry
    client = chromadb.EphemeralClient()
    client_settings = Settings(allow_reset=True, anonymized_telemetry=False)

    # Reset the search engine to ensure we don't use old copies.
    # NOTE: we do not need this for production
    search_engine = Chroma(client=client, client_settings=client_settings)
    search_engine._client.reset()
    search_engine = Chroma.from_documents(
        client=client,
        documents=docs,
        embedding=embeddings,
        client_settings=client_settings,
    )

    return search_engine


@cl.on_chat_start
async def on_chat_start():
    """This function is written to prepare the environments for the chat
    with PDF application. It should be decorated with cl.on_chat_start.

    Returns:
        None
    """
    # Asking user to to upload a PDF to chat with
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please Upload the PDF file you want to chat with...",
            accept=["application/pdf"],
            max_size_mb=20,
        ).send()
    file = files[0]

    # Process and save data in the user session
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    docs = process_file(file=file)
    cl.user_session.set("docs", docs)
    msg.content = f"`{file.name}` processed. Loading ..."
    await msg.update()

    # Indexing documents into our search engine
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    try:
        search_engine = await cl.make_async(create_search_engine)(
            docs=docs, embeddings=embeddings
        )
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
        raise SystemError
    msg.content = f"`{file.name}` loaded. You can now ask questions!"
    await msg.update()

    model = ChatOpenAI(
        model="gpt-3.5-turbo-16k-0613", temperature=0, streaming=True
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=search_engine.as_retriever(max_tokens_limit=4097),
        chain_type_kwargs={"prompt": PROMPT, "document_prompt": EXAMPLE_PROMPT},
    )

    # We are saving the chain in user_session, so we do not have to rebuild
    # it every single time.
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    # Let's load the chain from user_session
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain

    response = await chain.acall(
        message.content,
        callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)],
    )
    answer = response["answer"]
    sources = response["sources"].strip()

    # Get all of the documents from user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    # Adding sources to the answer
    source_elements = []
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
