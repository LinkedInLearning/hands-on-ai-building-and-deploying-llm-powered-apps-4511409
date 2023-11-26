from tempfile import NamedTemporaryFile
from typing import List

import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.chains import LLMChain

from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
            chunk_size=3000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

        # We are adding source_id into the metadata here to denote which
        # source document it is.
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
async def on_chat_start():
    """This function is written to prepare the environments for the chat
    with PDF application. It should be decorated with cl.on_chat_start.

    Returns:
        None
    """

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please Upload the PDF file you want to chat with...",
            accept=["application/pdf"],
            max_size_mb=20,
        ).send()
    file = files[0]

    # Send message to user to let them know we are processing the file
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    try:
        search_engine = await cl.make_async(create_search_engine)(file=file)
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
        raise SystemError

    model = ChatOpenAI(
        model="gpt-3.5-turbo-16k-0613",
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Chainlit GPT, a helpful assistant.",
            ),
            (
                "human",
                "{question}"
            ),
        ]
    )
    chain = LLMChain(llm=model, prompt=prompt, output_parser=StrOutputParser())

    # We are saving the chain in user_session, so we do not have to rebuild
    # it every single time.
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):

    # Let's load the chain from user_session
    chain = cl.user_session.get("chain")  # type: LLMChain

    response = await chain.arun(
        question=message.content, callbacks=[cl.LangchainCallbackHandler()]
    )

    await cl.Message(content=response).send()

