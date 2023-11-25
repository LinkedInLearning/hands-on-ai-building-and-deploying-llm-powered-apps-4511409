import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
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

