import os
import chainlit as cl
from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from agents.tool import function_tool

load_dotenv()

gemini_api_key= os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)

config = RunConfig(
    model=model,
    
    model_provider=client,
    tracing_disabled=True
)

@function_tool("get weather")
def get_weather(Location: str, unit:str = "C")-> str:
    
    return f"The weather in {Location} is 22 degrees {unit}."

@function_tool("giaic_students_finder")
def student_finder(roll_no : int)-> str:
    
    data = {298640: "Sheza", 40789: "Sahil", 308976: "Kinza" }

    return data.get(roll_no, "Not Found")
    
agent = Agent (
    name= "Support Agent",
    instructions="You are a helpful assistant that can answer questions and help with tasks",
    model=model,
    tools=[get_weather, student_finder]
)

@cl.on_chat_start
async def chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hey, How can I help you today?").send()
    
@cl.on_message
async def handle_message(message=cl.Message):
    history = cl.user_session.get("history")
    
    msg = cl.Message(content="")
    await msg.send()
    
    history.append({"role": "user", "content": message.content})
    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=config
    )
    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(event.data.delta)
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    # await cl.Message(content=result.final_output).send()