import asyncio
import logging

# from autogen_core import SingleThreadedAgentRuntime

# from semantic_kernel.agents import ChatCompletionAgent, OpenAIAssistantAgent
from semantic_kernel.agents import MagenticOrchestration

# from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
# from semantic_kernel.kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.agents import AzureAIAgentSettings
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel import Kernel


logging.basicConfig(level=logging.WARNING)  # Set default level to WARNING
logging.getLogger("semantic_kernel.agents.orchestration.magentic_one").setLevel(
    logging.DEBUG
)  # Enable DEBUG for group chat pattern


async def main():
    # Create kernel for function selection strategy
    kernel = Kernel()
    
    # Create a research agent
    research_agent = ChatCompletionAgent(
        name="ResearchAgent",
        description="A helpful assistant with access to web search. Ask it to perform web searches and find information about buildings.",
        instructions=(
            "You are a Researcher. You find information about buildings, their heights, "
            "and locations. You have access to web search capabilities."
        ),
        service=AzureChatCompletion(
            deployment_name="your-deployment-name",  # Replace with your deployment name
            api_version="2024-02-01",  # Or your preferred API version
        ),
        kernel=kernel,
    )

    # Create an Azure OpenAI coder agent (instead of OpenAI Assistant)
    coder_agent = ChatCompletionAgent(
        name="CoderAgent",
        description="A helpful assistant with code analysis capability for data processing.",
        instructions=(
            "You solve questions using code analysis and data processing. You can analyze data, "
            "create tables, perform calculations, and provide structured output. When given building data, "
            "you can organize it into tables and calculate statistics. You should write Python code "
            "snippets to demonstrate your analysis approach."
        ),
        service=AzureChatCompletion(
            deployment_name="your-deployment-name",  # Replace with your deployment name
            api_version="2024-02-01",
        ),
        kernel=kernel,
    )

    # Create agent group chat with selection strategy
    selection_strategy = kernel_function(
        kernel=kernel,
        function_name="select_next_speaker",
        agents=[research_agent, coder_agent],
    )
    
    # Create group chat settings
    settings = AzureAIAgentSettings(
        selection_strategy=selection_strategy,
        termination_strategy=None,  # Will terminate when task is complete
        max_rounds=10,
    )
    
    # Create the group chat
    group_chat = AgentGroupChat(
        agents=[research_agent, coder_agent],
        settings=settings,
    )
    
    # Start the conversation
    task_message = (
        "What are the 50 tallest buildings in the world? Create a table with their names "
        "and heights grouped by country with a column showing the average height of the buildings "
        "in each country. ResearchAgent, please start by finding information about the tallest buildings."
    )
    
    print(f"Starting task: {task_message}")
    
    async for message in group_chat.invoke(task_message):
        print(f"{message.agent.name}: {message.content}")
        if hasattr(message, 'metadata') and message.metadata:
            print(f"Metadata: {message.metadata}")


if __name__ == "__main__":
    asyncio.run(main())