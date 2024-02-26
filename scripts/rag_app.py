import re
import json
import chainlit as cl

from langchain_community.vectorstores import FAISS
from rag_module import RagModule

from collections import defaultdict



prompt_template = """
Tu t'appelles GAIA et tu travailles pour Politique Agricole Commune (PAC). Tu es un agent intelligent spécialisé sur les aides financières agricoles. 
Tu es chargé de donner des conseils sur les aides financières disponibles pour les agriculteurs.
Tu comprends et génère les réponses en français, jamais en anglais.
Merci de bien vouloir répondre aux questions en utilisant seulement le contexte suivant.
contexte: {context}

historique: {history}

question: {question}
réponse:
"""



##------------ CHAINLIT ---------------##
@cl.on_chat_start
async def start():
    rag = RagModule()
    
    db = rag.get_faiss_db()

    qa_chain = rag.retrieval_qa_memory_chain(db, prompt_template)

    msg = cl.Message(content="Lancement du bot...", author = "Gaia")
    await msg.send()
    msg.content = "Bonjour et bienvenue sur le bot qui connait tout sur toutes les aides de la PAC (ou à peu près). Quelle est ta question ?"
    await msg.update()
    cl.user_session.set("chain", qa_chain)

@cl.on_message
async def main(message):
    rag = RagModule()

    chain = cl.user_session.get("chain")
    
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True, 
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
  
    cb.answer_reached=True
    response = await chain.ainvoke(message.content, callbacks=[cb])

    answer = response.get('result')
    sources = rag.get_sources_document(response.get('source_documents'))

    elements = [cl.Pdf(name = "Pdf", display ="inline", path = path) for path in sources]
    
 
    if response.get('source_documents'):
        answer = rag.shape_answer_with_source(answer, sources)
    else:
        answer += f"\nNo sources found"
    
    await cl.Message(content=answer, elements=elements, author="Gaia").send()
    # await cl.Message(content=answer, author = "Chatbot Eureden").send()


