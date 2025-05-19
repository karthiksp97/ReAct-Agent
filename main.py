import re
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.render import render_text_description
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

def find_tool_by_name(tools, tool_name):
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Couldn't find the tool {tool_name}")

@tool
def get_text_length(text: str) -> int:
    """ find the length of input"""
    return len(text.strip("'\n").strip('"'))

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
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template).partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools])
)

llm = ChatOllama(
    model="llama3",
    temperature=0.8,
    num_predict=256,
   
    verbose=True
)

agent = ({"input": lambda x: x["input"], "agent_scratchpad": lambda x: x["agent_scratchpad"]}
         | prompt
         | llm
         | StrOutputParser())

# Scratchpad text that accumulates the chain of thought + actions + observations
scratchpad = ""

# The user question
question = "what is the text length  of in characters?"

max_iters = 15
iters = 0
scratchpad = ""

while iters < max_iters:
    iters += 1
    output = agent.invoke({"input": question, "agent_scratchpad": scratchpad})
    print(f"\nLLM output (iteration {iters}):\n", output)

    final_answer_match = re.search(r"Final Answer:\s*(.*)", output, re.IGNORECASE)
    if final_answer_match:
        print("\n✅ Final answer:", final_answer_match.group(1).strip())
        break

    action_match = re.search(r"Action:\s*(\w+)", output)
    input_match = re.search(r"Action Input:\s*(.*)", output)

    if action_match and input_match:
        tool_name = action_match.group(1).strip()
        tool_input_raw = input_match.group(1).strip().strip('"').strip("'")
        tool_to_use = find_tool_by_name(tools, tool_name)
        observation = tool_to_use.invoke({"text": tool_input_raw})

        scratchpad += f"\nThought: {output.split('Thought:')[1].split('Action:')[0].strip()}" \
                      f"\nAction: {tool_name}" \
                      f"\nAction Input: {tool_input_raw}" \
                      f"\nObservation: {observation}\n"

        print(f"\n🛠️ Tool used: {tool_name}")
        print(f"📝 Input: {tool_input_raw}")
        print(f"📏 Observation: {observation}")
    else:
        print("⚠️ Could not parse Action or Action Input, stopping.")
        break
else:
    print("⚠️ Max iterations reached without a final answer.")

