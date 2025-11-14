
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
import json

load_dotenv()


llm = ChatOpenAI(model="gpt-4o",temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


vectorstore = Chroma(
    persist_directory="./chroma_langchain_db",
    collection_name="prototype",
    embedding_function=embeddings
)



retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5},
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


base_path = os.getcwd()
json_path = os.path.join(base_path, "meta_k10.json")

with open(json_path, "r", encoding="utf-8") as f:
    meta_k10 = json.load(f)
    meta_k10 = {int(k): v for k, v in meta_k10.items()}

@tool
def page_retriever_tool(page_no: int) -> str:
    """
    Retrieve the full text content of a specific page from the Meta K-10 document.

    Args:
        page_no (int): The page number to retrieve (1-based index).

    Returns:
        str: A formatted string containing the complete text of the requested page.
    """
    try:
        page_content = meta_k10[page_no]
        return f"The Content of Page {page_no} is:\n-----\n{page_content}\n-----\n"
    except:
        return f"No Content found for Page {page_no}."




@tool
def context_retriever_tool(query: str) -> str:
    """
    Retrieve relevant chunks from the document for a given natural language query.
    Parameters
    ----------
    query : str
        The natural language query used to fetch relevant chunks.

    Returns
    -------
    str
        A formatted string containing page numbers and associated text content
        from the retrieved documents. Each chunk is separated by a dashed line.
    """
    print(f"Query: {query}")

    docs = retriever.invoke(query)
    relevant_content = ""

    for item in docs:
        metadata = item.metadata or {}
        page_label = metadata.get("page_label", "Unknown Page")
        page_content = item.page_content

        relevant_content += (
            f"Page No: {page_label}\n"
            f"{page_content}\n"
            "------------------------------------------\n"
        )

    return relevant_content



@tool
def add(x: float, y: float) -> float:
    """
    Add two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The result of x + y.
    """
    return x + y


@tool
def sub(x: float, y: float) -> float:
    """
    Subtract two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The result of x - y.
    """
    return x - y


@tool
def mul(x: float, y: float) -> float:
    """
    Multiply two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The result of x * y.
    """
    return x * y


@tool
def div(x: float, y: float) -> float:
    """
    Divide two numbers.

    Args:
        x (float): First number.
        y (float): Second number (must not be zero).

    Returns:
        float: The result of x / y.

    Raises:
        ValueError: If y == 0.
    """
    if y == 0:
        raise ValueError("Division by zero is not allowed.")
    return x / y



tools = [page_retriever_tool,context_retriever_tool,add,sub,div,mul]
llm = llm.bind_tools(tools)


def route_tools(state: AgentState):
    """
    Decide where to route based on the LLM tool calls.

    Returns:
        "retriever" if any retriever tool is called
        "calculator" if any calculator tool is called
        "none" if no tool calls exist
    """

    last_msg = state["messages"][-1]

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return "none"

    # Check each tool call
    for tc in last_msg.tool_calls:
        tool_name = tc["name"]

        if tool_name in ["page_retriever_tool","context_retriever_tool"]:
            return "retriever"

        if tool_name in ["add", "sub", "mul", "div"]:
            return "calculator"

    return "none"



system_prompt = """
You are an intelligent AI assistant who answers questions about the Meta Platforms, Inc. 10-K filing.

You will now receive only the user's query (no pre-extracted context).
Your job is to determine which tool to call based on the user's request.

Available tools:
- context_retriever_tool → returns only relevant text chunks, not full pages
- page_retriever_tool → returns complete pages for given page numbers
- Calculator tools (add, sub, mul, div) → must be used for all numeric reasoning and calculations

Your responsibilities:

1. Determine what information is needed to answer the user's query.

2. If needed, call `context_retriever_tool` to identify relevant chunks.
   - This tool returns partial context; inspect the chunk metadata (page_number).
   - If the chunk appears truncated, incomplete, or insufficient for a correct answer:
       → Use `page_retriever_tool` to retrieve the full page.
   - If the content appears to continue across pages (e.g., tables, sections):
       → Also fetch adjacent pages.

3. When the user question requires calculations:
   - ALWAYS use the calculator tool (add/sub/mul/div).
   - NEVER perform arithmetic manually.

4. After retrieving all necessary information:
   - Synthesize a final answer strictly based on the retrieved pages and calculator outputs.
   - Clearly cite the page numbers used.
   - Do not hallucinate or infer information not retrieved.

5. Only produce a final natural-language answer when no further tool calls are required.

Always rely strictly on retrieval and calculator tools.
Never guess. Never answer based on incomplete evidence.
"""



tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools


tools_dict


#LLM_Agent
def call_llm(state: AgentState)->AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)]+messages[:-1]+[messages[-1]]
    response = llm.invoke(messages)
    return {'messages':[response]}


# Retriever Agent
def retrieve_document(state: AgentState)->AgentState:
    """Execute Page Retriver from the LLM's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    print(f"Tool Calls: ",tool_calls)
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args']}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'])
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


# Calculator Agent
def calculate(state: AgentState) -> AgentState:
    """
    Executes calculator tool calls triggered by the LLM.
    
    This agent handles tool calls such as add, sub, mul, and div.
    For each tool call:
      - Validates the tool name
      - Executes the tool with provided arguments
      - Returns a ToolMessage back to the LLM

    Returns:
        AgentState: Contains ToolMessage objects for each executed tool.
    """

    last_ai_msg = state["messages"][-1]
    tool_calls = last_ai_msg.tool_calls
    results = []

    print(f"\nTool Calls Received: {tool_calls}")

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        call_id = tool_call["id"]

        print(f"→ Calling tool '{tool_name}' with args: {args}")

        # Handle invalid tool names
        if tool_name not in tools_dict:
            result_text = f"Error: Tool '{tool_name}' not found. Valid tools: {list(tools_dict.keys())}"
            print(result_text)

        else:
            # Execute tool
            try:
                result_text = tools_dict[tool_name].invoke(args)
            except Exception as e:
                result_text = f"Tool '{tool_name}' failed: {str(e)}"
                print(result_text)

        # Append ToolMessage for LLM to continue
        results.append(
            ToolMessage(
                tool_call_id=call_id,
                name=tool_name,
                content=str(result_text)
            )
        )

    print("→ Calculator Tools Execution Complete. Returning control to LLM.\n")
    return {"messages": results}



graph = StateGraph(AgentState)

# Add nodes
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", retrieve_document)
graph.add_node("calculator_agent", calculate)

# Entry point: first retrieve context for user query
graph.set_entry_point("llm")

# Conditional routing after LLM
graph.add_conditional_edges(
    "llm",
    route_tools,
    {
        "retriever": "retriever_agent",
        "calculator": "calculator_agent",
        "none": END,
    }
)

# After tools → return to LLM
graph.add_edge("retriever_agent", "llm")
graph.add_edge("calculator_agent", "llm")

# Compile graph
rag_agent = graph.compile()

# from IPython.display import Image, display
# display(Image(rag_agent.get_graph().draw_mermaid_png()))

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = rag_agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]

    print("\n=== ANSWER ===")
    print(result['messages'][-1].content)
    conversation_history = conversation_history[-30:] 
    user_input = input("Enter: ")





