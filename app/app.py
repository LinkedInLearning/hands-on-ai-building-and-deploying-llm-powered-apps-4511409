import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain

@cl.on_chat_start
async def on_chat_start():
    ##########################################################################
    # Exercise 1a:
    # Our Chainlit app should initialize the LLM chat via Langchain at the
    # start of a chat session.
    # 
    # First, we need to choose an LLM from OpenAI's list of models. Remember
    # to set streaming=True for streaming tokens
    ##########################################################################
    model = ChatOpenAI(
        ...
    )

    ##########################################################################
    # Exercise 1b:
    # Next, we will need to set the prompt template for chat. Prompt templates
    # is how we set prompts and then inject informations into the prompt.
    # 
    # Please create the prompt template using ChatPromptTemplate. Use variable
    # name "question" as the variable in the template.
    # Refer to the documentation listed in the README.md file for reference.
    ##########################################################################
    prompt = ChatPromptTemplate.from_messages(
        ...
    )
    ##########################################################################
    # Exercise 1c:
    # Now we have model and prompt, let's build our Chain. A Chain is one or a
    # series of LLM calls.We will use the default StrOutputParser to parse the
    # LLM outputs.
    ##########################################################################
    chain = LLMChain(
        llm=...,
        prompt=...,
        output_parser=StrOutputParser()
    )

    # Let's save the chain from user_session so we do not have to rebuild
    # every single time we receive a message
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):

    # Let's load the chain from user_session
    chain = cl.user_session.get("chain")  # type: LLMChain

    ##########################################################################
    # Exercise 1d:
    # Everytime we receive a new user message, we will get the chain from 
    # user_session. We will run the chain with user's question and return LLM
    # response to the user.
    ##########################################################################
    response = await chain.arun(
        ...,
        callbacks=[cl.LangchainCallbackHandler()]
    )

    await cl.Message(content=response).send()

