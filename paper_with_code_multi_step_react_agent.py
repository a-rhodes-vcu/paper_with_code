"""
LangChain Multi-Step Problem Solving (Reason + Act) agent.

What it does
- Uses a ReAct-style agent to solve questions that require multiple tool calls.
- Tools included:
  1) wikipedia[query]  -> fetch facts
  2) calculator[expr]  -> compute arithmetic reliably

Install
  pip install -U langchain langchain-openai langchain-community wikipedia datasets langchain-classic

Env
  export OPENAI_API_KEY="..."

Run
  python paper_with_code_multi_step_react_agent.py
"""

from __future__ import annotations

import re
from typing import Any

from langchain_openai import ChatOpenAI

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI()
# -------------------------
# 1) Tools
# -------------------------
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1800)
)

def safe_calculator(expression: str) -> str:
    """
    Small safe-ish calculator:
    - allows digits, + - * / % ( ) . and whitespace
    - blocks names/letters to avoid code execution
    """
    expr = expression.strip()
    if not expr:
        return "Calculator error: empty expression"
    if re.search(r"[^0-9\.\+\-\*\/\%\(\)\s]", expr):
        return "Calculator error: invalid characters in expression"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Calculator error: {e}"

tools = [
    Tool(
        name="wikipedia",
        description="Look up factual information on Wikipedia. Input: a search query.",
        func=wiki.run,
    ),
    Tool(
        name="calculator",
        description="Evaluate a math expression. Input: arithmetic like '12*(3+4)'.",
        func=safe_calculator,
    ),
]


# -------------------------
# 2) ReAct prompt (paper-ish)
# -------------------------
REACT_PROMPT = PromptTemplate.from_template(
    """You are a tool-using agent that solves multi-step problems by interleaving reasoning and tool use.

You have access to these tools:
{tools}

Use this format:

Question: the input question you must answer
Thought: what you should do next
Action: the tool name, one of [{tool_names}]
Action Input: the input to the tool
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Final: the final answer (be concise and correct)

Important rules:
- If you need a fact, use wikipedia.
- If you need arithmetic, use calculator.
- Don't guess tool outputs.
- Keep going until you can answer confidently.

Question: {input}
Thought:{agent_scratchpad}"""
)


# -------------------------
# 3) Build agent + executor
# -------------------------
def build_agent(model: str = "gpt-4o-mini", temperature: float = 0.0) -> AgentExecutor:
    llm = ChatOpenAI(model=model, temperature=temperature)
    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)

    # verbose=True prints the tool calls + intermediate steps (very helpful for debugging)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
    )
    return executor


# -------------------------
# 4) Demo problems (multi-step)
# -------------------------
def run_question(executor: AgentExecutor, question: str) -> str:
    result: Any = executor.invoke({"input": question})
    return result["output"]


if __name__ == "__main__":

    from datasets import load_dataset

    # Specify the dataset name on the Hugging Face Hub, e.g., "imdb"
    dataset_name = "hotpotqa/hotpot_qa"

    # Load the dataset
    # This automatically downloads and caches the dataset, typically splitting it into train/test sets
    hotpotqa = load_dataset(dataset_name,'distractor')

    # Access specific splits
    list_of_questions_from_hotpotqa = []
    for q in hotpotqa['train']['question']:
        list_of_questions_from_hotpotqa.append(q)
    list_of_questions_from_hotpotqa = list_of_questions_from_hotpotqa[0:100]

    agent_exec = build_agent()

    questions = list_of_questions_from_hotpotqa

    for q in questions:
        print("\n" + "=" * 80)
        print("Q:", q)
        answer = run_question(agent_exec, q)
        print("A:", answer)
