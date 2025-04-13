import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4o")


async def main():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                # make sure you start your weather server on port 8000
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
        }
    ) as client:
        agent = create_react_agent(model, client.get_tools())
        math_question = "what's (3 + 5) x 12?"
        weather_question = "what is the weather in nyc?"

        print(f"Math question: {math_question}")
        print(f"Weather question: {weather_question}")

        math_response = await agent.ainvoke({"messages": math_question})
        weather_response = await agent.ainvoke({"messages": weather_question})

        print(f"Math response: {math_response['messages'][-1].content}")
        print(f"Weather response: {weather_response['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
