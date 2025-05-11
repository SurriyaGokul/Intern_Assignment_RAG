## Toshiba Printer Specifications Assistant

This Streamlit app is an intelligent assistant for answering questions about Toshiba printer specifications. It leverages [LangChain](https://www.langchain.com/), a powerful language model (LLM) framework, and a custom retrieval-augmented generation (RAG) pipeline to provide fact-based answers from official Toshiba printer brochures.

---

### Features

- **Conversational Q&A:** Ask about Toshiba printer models, specifications, and comparisons.
- **Math Support:** Perform calculations related to printer specs using a built-in calculator tool.
- **Document Grounding:** Answers are sourced from PDF brochures using semantic search and document retrieval.
- **Strict Output Parsing:** Custom output parser ensures robust, error-free agent responses.

---

### How It Works

- **RAG Pipeline:** The app loads and indexes Toshiba printer PDF brochures. When you ask a question, it retrieves the most relevant document snippets and uses the LLM to answer based on those.
- **Tools:** The agent can use two tools:
  - `RAGPipeline` for document-based questions.
  - `Calculator` for math expressions.
- **Output Parsing:** A custom parser ensures the agent's responses are always in the correct format, preventing parsing errors.

---

### Requirements

- Python 3.9+
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com/) (for local LLM and embeddings)
- [FAISS](https://github.com/facebookresearch/faiss) (for vector search)
- PDF brochures of Toshiba printers (place them in the specified directory in the script)

Install dependencies with:
\`\`\`bash
pip install streamlit langchain langchain-community faiss-cpu
\`\`\`

---

### Usage

1. **Prepare PDFs:** Place your Toshiba printer PDF brochures in the folder specified in the script (e.g., `C:\\Users\\surri\\AI\\GenAI`).
2. **Run the App:**
   \`\`\`bash
   streamlit run your_script.py
   \`\`\`
3. **Ask Questions:** Type your question about Toshiba printers in the input box (e.g., "What is the max print speed of e-STUDIO2329A?").

---

### Customization

- **PDF Directory:** Change the \`pdf_dir\` variable in the script to point to your own folder of Toshiba printer PDFs.
- **LLM Model:** The script uses the \`mistral\` model via Ollama. You can switch to another supported model if desired.

---

### Troubleshooting

- If you see parsing errors, ensure your question is clear and follows the expected format.
- Make sure the PDFs are present and readable in the specified directory.
- For best results, keep the PDF documents up to date.

---

### License

This project is for educational and demonstration purposes.

---

**Enjoy exploring Toshiba printer specs with AI!**
