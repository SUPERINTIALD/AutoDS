from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Step 1: Initialize LangChain Agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Step 2: Add Memory
# Retain only the last 3 interactions
memory = ConversationBufferMemory(k=3)

# Step 3: Define a tool
# Tools define what actions the agent can perform.
def llm_query_tool(input_text: str) -> str:
    """A simple tool that queries the LLM."""
    return llm(input_text)

# Create a tool object
query_tool = Tool(
    name="LLM Query Tool",
    func=llm_query_tool,
    description="Use this tool to query the LLM with any question or prompt."
)

# Step 4: Initialize the Agent
agent = initialize_agent(
    tools=[query_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iteration=5,
    memory=memory  # Add memory for truncation
)

# Step 5: Use the Agent
if __name__ == "__main__":
    print("Welcome to the LangChain LLM Agent!")
    while True:
        user_input = input("Ask me anything (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = agent.run(user_input)
        print(f"Agent: {response}")
