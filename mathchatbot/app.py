import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain, LLMChain
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler


st.title("MATH chatbot")
groq_api_key=st.sidebar.text_input(label="enter your groq api key here",type="password")

if not groq_api_key:
    st.info("please add your api key")
    st.stop()

llm=ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key)

wikipedia=WikipediaAPIWrapper()
wiki_tool=Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="tool for searching the information on wikipedia")

math=LLMMathChain.from_llm(llm=llm)
math_tool=Tool(
    name="Math_tool",
    func=math.run,
    description="tool for calculating the answer for given mathematical expression"
)

prompt="you are the agent used for calculating the answer for the given mathematical expression, find the solution for the question and provide the answer in point wise manner"
promptTemplate=PromptTemplate(
    template=prompt,
    input_variables=["question"]
)

chain=LLMChain(llm=llm,prompt=promptTemplate)
reasoning_agent=Tool(
    name="reasoning agent",
    func=chain.run,
    description="tool for answering logic based and reasoning based question"
)
assistant_agent=initialize_agent(
    tools=[wiki_tool,math_tool,reasoning_agent],
    verbose=False,
    handle_parsing_errors=True,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION

)

if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role":"assitant","content":"hi how can i help you today"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

question=st.text_input("enter the question","what is 5+5?")

if st.button:
    if question:
        with st.spinner("generate answer..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message('user').write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({"role":"assitant","content":response})
            st.write("#response")
            st.success(response)
    else:
        st.warning("please enter the question")
