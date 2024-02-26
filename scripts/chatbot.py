import chainlit as cl
from rag_module import RagModule
from sql_db import SqlDb

import pandas as pd



@cl.on_chat_start
async def on_chat_start():

    
    msg = cl.Message(content=f"Bienvenue sur le chatbot qui connait tout sur tout. Quelle est ta question ?", author = "Mistral")
    await msg.send()


@cl.on_message
async def main(message):

    rag = RagModule()
    llm = rag.load_mistral()

    response = llm.invoke(message.content)

    # send response back to user
    await cl.Message(content = response.content).send()

