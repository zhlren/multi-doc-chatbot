"""Main entrypoint for the app."""
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates

from query.chain_by_doc import get_chain
from web.schemas import ChatResponse
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

app = FastAPI()
templates = Jinja2Templates(directory="templates")

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # 设置全局日志level，不设置默认WARN

# save log to file
file_handler = logging.FileHandler('logs/app.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)

# print to screen
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

@app.on_event("startup")
async def startup_event():
    print("startup begin")
    print("startup end")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    qa = get_chain()

    await websocket.accept()
    chat_history = []
    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            logger.info("question:" + question)
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())



            # result = pdf_qa(
            #     {"question": query, "chat_history": chat_history})
            #
            # result = await qa_chain.acall(
            #     {"question": question, "chat_history": chat_history}
            # )

            result = qa({"question": question, "chat_history": chat_history})

            chat_history.append((question, result["answer"]))

            answer_resp = ChatResponse(sender="bot", message=result["answer"], type="stream")
            await websocket.send_json(answer_resp.dict())

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn
    logger.info("main start")
    uvicorn.run(app, host="0.0.0.0", port=9000)
