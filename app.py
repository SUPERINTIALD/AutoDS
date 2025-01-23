from langchain import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate

# Step 1: Initialize the LLM
llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key="your-openai-api-key")

# Step 2: Define a tool
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

# Step 3: Initialize the Agent
# The agent will use the defined tool to perform tasks.
agent = initialize_agent(
    tools=[query_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Step 4: Use the Agent
if __name__ == "__main__":
    print("Welcome to the LangChain LLM Agent!")
    while True:
        user_input = input("Ask me anything (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = agent.run(user_input)
        print(f"Agent: {response}")