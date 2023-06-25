"""Create a ChatVectorDBChain for question/answering."""
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI



def get_data(question: str) -> str:  # <== CHANGE THE TYPE

    load_dotenv('.env')
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    messages = [
        SystemMessage(content="请忽略之前的任何指示。你是《GemStone IV》的旁白，助理。我刚刚建立了第一次联系，想要制作一个角色。在性格发展的每个阶段，你都会问我问题，我会回答。最后，显示我的角色的统计数据和库存。"),
        HumanMessage(content=question)
    ]

    resp = chat(messages)

    return resp.content;



