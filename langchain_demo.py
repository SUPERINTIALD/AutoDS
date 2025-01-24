# ------------------------------------------------------------
#   LangChain Simple Conversational Demo
#   Proof of concept and reference code
#   Documentation: https://langchain.readthedocs.io/en/latest/
#   1/23/2025 Trimble Capstone Team
# ------------------------------------------------------------

# -------------------- Imports -------------------- #
# LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
# API Key Management
from dotenv import load_dotenv
import os

# -------------------- Setup -------------------- #

# API Key
    # To load the openai api key from the .env file
    # 1. Create a file ".env" in the root directory of the project
    # 2. Add the following line: OPENAI_API_KEY="sk-blah-blah-blah" in the .env file
    # 3. Make sure the venv is configured and activated with the python-dotenv package installed.

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# -------------------- Main -------------------- #


# Step 1: Create an LLM instance: 

    # The LLM is the core reasoning engine that processes user input, decides which tools to use, and generates responses.
    # The LangChain framework can use any LLM instance
    # Here, we use the ChatOpenAI integration helper provided by LangChain to create an LLM instance that interfaces with OpenAI GPT.

    # See more: https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100, api_key=OPENAI_API_KEY)


# Step 2: Add Memory

    # Memory allows the agent to retain context from previous interactions.
    # This is crucial for maintaining coherent conversations, especially in multi-turn dialogues in a chain.
    # ! The way of handling memory below is currently depricated and not suited for long term use.
    # LangChain reccomends using LangGraph for memory management, but we should explore our options first (azure blob storage, etc.)
    # TODO: Find a better solution, explore the docs, evaluate our options... TODO: This will be someone's job to explore.

    # See more: https://python.langchain.com/api_reference/langchain/memory.html 

memory = ConversationBufferMemory(k=3) # Simple but depricated memory managment: DEMO ONLY 


# Step 3: Define a Tool

    # Tools are functions or APIs that the agent can call to perform specific tasks and extend its capabilities beyond just generating text.
    # In this case, we define a simple tool that just queries the LLM. 
    # This tool takes an input string and returns the LLM's response.
    # ! TODO: Check out docs before implementing your own tools.

    # See more: https://python.langchain.com/api_reference/community/tools.html#module-langchain_community.tools

def llm_query_tool(input_text: str) -> str:
    """A simple tool that queries the LLM."""
    return llm(input_text)

query_tool = Tool(
    name="LLM Query Tool",
    func=llm_query_tool,
    description="Use this tool to query the LLM with any question or prompt."
)

# Step 4: Initialize the Agent
    # The agent is a class that is the core component that decides which tool to use and how to respond.
    # Agents are the actors that use an LLM as their "brain" to decide what to do.
    # Agents have the LLM, memory, and tools at their disposal.
    # This is the main reason to use LangChain over just using an LLM directly, since we can utilize a robust abstraction layer.

    # See more: https://python.langchain.com/api_reference/langchain/agents.html#module-langchain.agents

agent = initialize_agent(
    tools=[query_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iteration=5,
    memory=memory  # Add memory for truncation
)


# Step 5: Using an Agent

if __name__ == "__main__":

    print("LangChain Demonstration: Type 'exit' to quit.")

    # Loop to keep agent running for multiple interactions
    while True:

        # Get user input
        user_input = input("Input:")
        
        # Exit the loop if the user types "exit"
        if user_input.lower() == "exit": break 

        # Run the agent with the input and store the response
        response = agent.run(user_input)

        # Print back the response to the user
        print(f"Agent: {response}")
