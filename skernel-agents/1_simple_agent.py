import asyncio
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
import os
from dotenv import load_dotenv

load_dotenv() 

async def main():
    # Updated parameter name: deployment_name instead of chat_deployment_name
    service = AzureChatCompletion(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),  # Changed from chat_deployment_name
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    )
    
    # Initialize a chat agent with basic instructions
    agent = ChatCompletionAgent(
        service=service,
        name="SK-Assistant",
        instructions="You are a helpful assistant.",
    )

    # Get a response to a user message
    response = await agent.get_response(messages="Write a haiku about Semantic Kernel.")
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())