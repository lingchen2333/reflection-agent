from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages, StateGraph, END

from chains import generate_chain, reflect_chain

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)

def should_continue(state: MessageGraph):
    if len(state["messages"]) > 3:
        return END
    return REFLECT


#add dotted edge
builder.add_conditional_edges(GENERATE, should_continue, {
    REFLECT:REFLECT,
    END:END
})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path='flow.png')

if __name__ == "__main__":
    print("hello world")
    inputs = {
        "messages":[
        HumanMessage(content=""" Make this tweet better: "
    @LangChainAI
    - newly Tool Calling feature is seriously underrated.
    After a long wait, it's here - making the implementation of agents across different models with function calling - super easy.
    Made a video covering their newest blog post
    """)]}

    res = graph.invoke(inputs)
    print(res['messages'][-1].content)

