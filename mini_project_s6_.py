import streamlit as st
import logging
import requests
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from bs4 import BeautifulSoup
from fpdf import FPDF
import tempfile
import os


def clean_html(html_content):
    """Removes HTML tags and extracts readable text."""
    return BeautifulSoup(html_content, "html.parser").get_text(separator=" ").strip()



def load_document(url):
    """Fetches web page content with a custom User-Agent and extracts text."""
    headers = {
    "User-Agent":"Mozilla/5.0(WindowsNT10.0;Win64;x64)AppleWebKit/537.36(KHTML,like Gecko)Chrome/134.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        text_content = clean_html(response.text)
        if not text_content:
            return "No content extracted. Try a different URL."
        return text_content
    
    except Exception as e:
        logging.error(f"Failed to load document: {e}")
        return f"Error loading document: {e}"



def setup_summarization_chain():
    """Sets up the LLM summarization chain using Ollama."""
    prompt_template = PromptTemplate.from_template(
    template="""As a professional summarizer,
    create a detailed and comprehensive summary of the provided text, be it an article, post, conversation, 
    or passage, while adhering to these guidelines:
    1. Craft a summary that is detailed, thorough, in-depth,
      and complex, while maintaining clarity.
    2. Incorporate main ideas and essential information, eliminating extraneous
      language and focusing on critical aspects.
    3. Rely strictly on the provided text, without including external information.
    4. Format the summary in paragraph form for easy understanding.
    5. Give clear titles to portions of the summary to enhance readability.
    6.Add subheading to the main headings in the smmary generated .
        "{text}"
        DETAILED SUMMARY:"""
    )

    llm = ChatOllama(model="llama3:instruct", base_url="http://localhost:11434")
    return prompt_template | llm


def chunk_text(text, max_tokens=500):
    """Splits text into smaller chunks for processing."""
    words = text.split()
    return [" ".join(words[i:i+max_tokens])for i in range(0,len(words),max_tokens)]


def generate_pdf(summary_text):
    """Generates a PDF from the summary text and returns the file path."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)


    for line in summary_text.split('\n'):
        pdf.multi_cell(0, 10, line)


    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)


    return temp_file.name




def streamlit_mode():
    """Runs script in Streamlit mode with a web UI."""
    st.title("📄 WEBSUM: Web Page Summarizer")

    if "summary" not in st.session_state:
        st.session_state["summary"] = ""

    url = st.text_input("Enter URL to summarize:")

    if st.button("Summarize"):
        if not url:
            st.warning("Please enter a valid URL.")
        else:
            with st.spinner("Fetching document..."):
                document_text = load_document(url)

            if document_text.startswith("Error"):
                st.error(document_text)
                return

            with st.spinner("Summarizing..."):
                llm_chain = setup_summarization_chain()
                summaries = []
                
                for chunk in chunk_text(document_text):
                    try:
                        summary = llm_chain.invoke({"text": chunk})
                        result_text = getattr(summary, "content", str(summary))
                        summaries.append(result_text)
                    except Exception as e:
                        logging.error(f"Error in summarization: {e}")
                        summaries.append(f"Error processing chunk: {e}")

                st.session_state["summary"] = "\n\n".join(summaries)
                st.success("✅ Summary generated successfully!")

    if st.session_state["summary"]:
        st.markdown("### 🗒️ Summary")
        st.markdown(st.session_state["summary"])

        # PDF generation
        pdf_path = generate_pdf(st.session_state["summary"])
        with open(pdf_path, "rb") as file:
            st.download_button(
                label="📥 Download Summary as PDF",
                data=file,
                file_name="summary.pdf",
                mime="application/pdf"
            )



if __name__ == "__main__":
    streamlit_mode()
