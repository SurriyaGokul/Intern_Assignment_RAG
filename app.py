from typing import Union
import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.tools import render_text_description
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, Tool, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import re
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os, re

# Initialize session state
if 'agent_executor' not in st.session_state:
    llm = ChatOllama(model="mistral", temperature=0)
    embedding_model = OllamaEmbeddings(model="mistral")

    # Define tools
    def calc_tool(input: str) -> str:
        try: return str(eval(input, {"__builtins__": {}}))
        except: return "Invalid expression"

    def rag_tool_func(query: str) -> str:
        if os.path.exists("faiss_index_intern"):
            vectorstore = FAISS.load_local(
                "faiss_index_intern",
                embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            pdf_dir = r"C:\Users\surri\AI\GenAI"
            pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
            all_docs = []
            for pdf in pdf_files:
                all_docs += PyPDFLoader(pdf).load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
            docs = splitter.split_documents(all_docs)
            vectorstore = FAISS.from_documents(docs, embedding_model)
            vectorstore.save_local("faiss_index_intern")

        prompt = PromptTemplate.from_template("""
You are given Toshiba printer brochure containing all the information regarding its printer models in PDF form.The Prices are not mentioned in the document but more the specifications and capable a model the more expensive it is so keep this in mind.Most specs are in tables, so scan thoroughly and answer the question based on the tables and text in the document. Answer all the questions with relevant models like the e-STUDIO series, and if the model is not mentioned in the document, say "I don't know". Only give to the point answer dont give greeting and thanks.

{context}

Based on the above, answer as economically as possible (no extra features):
Question: {input}
""")
        stuff_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        qa_chain = create_retrieval_chain(retriever, stuff_chain)
        out = qa_chain.invoke({"input": query})
        ans = out["answer"].strip()
        return re.sub(r"^(Observation:|Final Answer:)", "", ans, flags=re.IGNORECASE).strip()

    tools = [
        Tool(name="Calculator", func=calc_tool, description="Use for math expressions"),
        Tool(name="RAGPipeline", func=rag_tool_func, description="Use for printer specs"),
    ]

    # Create agent
    agent_prompt = PromptTemplate.from_template("""
You are a helpful assistant restricted to using the tools provided below. 
                                                
**Strict Formatting Rules:**
1. For math questions:
   - Use ONLY the Calculator tool
   - Return ONLY the numerical answer after calculation
   - Never combine Action and Final Answer in the same response

2. For printer questions:
   - Use ONLY the RAGPipeline tool
   - Return ONLY facts from the documents

**Tool Selection Protocol:**
- Use Calculator EXCLUSIVELY for pure math expressions (e.g., "5*2")
- Use RAGPipeline for ALL other queries

**Output Format Requirements:**
- ONE tool invocation per response
- NEVER combine Action and Final Answer in the same output
- For math results: ONLY the numerical answer
                    
IMPORTANT: Never output both Action and Final Answer in the same response. Always output ONLY Action/Action Input, OR ONLY Final Answer, never both.

When you have a final answer, ALWAYS format it as:
Final Answer: Your answer here

Available tools:
{tools}

Response format:
Thought: [Reasoning]
Action: [Tool Name]
Action Input: [Input]
Observation: [Result]
Final Answer: [Final Response]

Begin!

Question: {input}
{agent_scratchpad}
""").partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    class DualOutputParser(AgentOutputParser):
        """
        Custom parser that handles both Action+Input and Final Answer in same response
        Prioritizes Final Answer if both are present
        """
        
        def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
            # Check for final answer first
            final_answer_match = re.search(r"Final Answer:\s*(.*?)(?:\n|$)", text, re.DOTALL|re.IGNORECASE)
            if final_answer_match:
                return AgentFinish(
                    return_values={"output": final_answer_match.group(1).strip()},
                    log=text
                )

            # Then check for action pattern with improved regex
            action_match = re.search(
                r"Action:\s*([\w\s]+).*?Action Input:\s*(.*?)(?:\n|$)",
                text,
                re.DOTALL|re.IGNORECASE
            )
            
            if action_match:
                tool = action_match.group(1).strip()
                tool_input = action_match.group(2).strip()
                # Handle potential quotes around tool input
                if (tool_input.startswith('"') and tool_input.endswith('"')) or \
                   (tool_input.startswith("'") and tool_input.endswith("'")):
                    tool_input = tool_input[1:-1]
                
                return AgentAction(
                    tool=tool,
                    tool_input=tool_input,
                    log=text
                )

            last_line = text.splitlines()[-1].strip()
            if last_line:
                return AgentFinish(
                    return_values={"output": last_line},
                    log=text
                )
                    
            raise ValueError(f"Could not parse output: {text}")

    def create_agent():
        output_parser = DualOutputParser()
        agent = create_react_agent(
            llm=llm, 
            tools=tools, 
            prompt=agent_prompt,
            output_parser=output_parser
        )
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

    st.session_state.agent_executor = create_agent()

# Streamlit UI
st.set_page_config(page_title="Toshiba Printer Assistant", page_icon="🖨️")
st.title("Toshiba Printer Specifications Assistant")

user_input = st.text_input("Ask about Toshiba printers:", placeholder="What's the maximum print speed of e-STUDIO2329A?")

if user_input:
    with st.spinner("Analyzing specs..."):
        # Create container for execution trace
        trace_container = st.container()
        st_callback = StreamlitCallbackHandler(trace_container)
        
        try:
            response = st.session_state.agent_executor.invoke(
                {"input": user_input},
                {"callbacks": [st_callback]}
            )
            
            st.subheader("Final Answer:")
            st.write(response["output"])
            
            with st.expander("Raw Execution Details"):
                st.json(response["intermediate_steps"])
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try rephrasing your question or check the input format.")
