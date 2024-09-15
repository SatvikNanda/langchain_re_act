#langchain hub is a very famous marketplace for prompts where people share their prompts that come in of great use 
from typing import Union, List
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.tools.render import render_text_description
from callbacks import AgentCallBackHandler



load_dotenv()

#@tool is a Decorator: marks the functions as tool that can be used by the agent within the langchain
@tool
def get_text_length(text:str) -> int:
    """Returns the length of the text by characters"""
    text = text.strip("'\n").strip('"') #stripping away non alphabet characters just in case
    print(f"get_text_length enter with {text=}")
    
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

"""Purpose:

    Searches through a list of tools to find a tool with a specific name.

Parameters:

    tools (List[Tool]): A list of available tools.
    tool_name (str): The name of the tool to find.

Logic:

    Iterates over each tool in the list and checks if its name matches tool_name.
    Returns the tool if found.
    Raises a ValueError if the tool is not found."""

if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools])
    )

    llm = ChatOpenAI(temperature=0, stop = ["\nObservation", "Observation"], callbacks=[AgentCallBackHandler()])
    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of 'HAPPY' in characters?",
            "agent_scratchpad": intermediate_steps
        }
    )

    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
        intermediate_steps.append((agent_step, str(observation)))

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of 'HAPPY' in characters?",
            "agent_scratchpad": intermediate_steps
        }
    )

    print(agent_step)

    if isinstance(agent_step, AgentFinish):
        print("### AgentFinish ###")
        print(agent_step.return_values)