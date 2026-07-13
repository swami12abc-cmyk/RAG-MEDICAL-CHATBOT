from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import traceback

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """
Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""


def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )


def create_qa_chain():
    try:
        logger.info("Loading Vector Store...")

        db = load_vector_store()

        if db is None:
            raise Exception("Vector Store could not be loaded.")

        logger.info("Vector Store Loaded Successfully.")

        logger.info("Loading LLM...")

        llm = load_llm()

        if llm is None:
            raise Exception("LLM could not be loaded.")

        logger.info("LLM Loaded Successfully.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": set_custom_prompt()
            },
        )

        logger.info("QA Chain Created Successfully.")

        return qa_chain

    except Exception as e:
        logger.error("Failed to create QA Chain")
        traceback.print_exc()
        print("\n========== RETRIEVER ERROR ==========")
        print(e)
        print("=====================================\n")
        return None