"""Main entrypoint for the app."""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage, HumanMessage
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import  ConversationBufferMemory, ConversationSummaryMemory
from langchain.callbacks import get_openai_callback

from web.schemas import ChatResponse

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
    load_dotenv('.env')
    print("startup end")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):


    await websocket.accept()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    conversation = ConversationChain(llm = llm, memory = ConversationBufferWindowMemory(k=3))

    # summary_conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))
    fst_question = "I want you to act as a spoken English teacher and improver. I will speak to you in English and you will reply to me in English to practice my spoken English. I want you to keep your reply neat, limiting the reply to 100 words. I want you to strictly correct my grammar mistakes, typos, and factual errors. I want you to ask me a question in your reply. Now let's start practicing, you could ask me a question first. Remember, I want you to strictly correct my grammar mistakes, typos, and factual errors."

    resp = track_tokens_usage(conversation, fst_question)


    start_resp = ChatResponse(sender="bot", message="", type="start")
    await websocket.send_json(start_resp.dict())

    answer_resp = ChatResponse(sender="bot", message=resp, type="stream")
    await websocket.send_json(answer_resp.dict())

    end_resp = ChatResponse(sender="bot", message="", type="end")
    await websocket.send_json(end_resp.dict())


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

            resp = track_tokens_usage(conversation, question)

            chat_history.append((question, resp))

            answer_resp = ChatResponse(sender="bot", message=resp, type="stream")
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


def track_tokens_usage(chain, query):
    with get_openai_callback() as cb :
        result = chain.run(query)
        print(f'Total tokens: {cb.total_tokens}')

    return result


if __name__ == "__main__":
    import uvicorn
    logger.info("main start")
    uvicorn.run(app, host="0.0.0.0", port=9000)
