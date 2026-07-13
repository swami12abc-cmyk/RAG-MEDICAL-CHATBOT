import os
import traceback

from langchain_community.vectorstores import FAISS

from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)


def save_vector_store(text_chunks):
    """
    Creates a FAISS vector database and saves it locally.
    """

    try:
        if not text_chunks:
            raise Exception("No text chunks found.")

        logger.info("Loading Embedding Model...")
        embedding_model = get_embedding_model()

        logger.info("Creating FAISS Vector Store...")
        db = FAISS.from_documents(
            documents=text_chunks,
            embedding=embedding_model
        )

        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

        logger.info(f"Saving Vector Store at: {DB_FAISS_PATH}")

        db.save_local(DB_FAISS_PATH)

        logger.info("✅ Vector Store Saved Successfully.")

        return db

    except Exception as e:

        logger.error("Failed to Save Vector Store")
        traceback.print_exc()

        print("\n========== SAVE VECTOR STORE ERROR ==========")
        print(e)
        print("=============================================\n")

        return None


def load_vector_store():
    """
    Loads an existing FAISS vector database.
    """

    try:

        logger.info("Loading Embedding Model...")
        embedding_model = get_embedding_model()

        logger.info(f"Checking Vector Store Path: {DB_FAISS_PATH}")

        if not os.path.exists(DB_FAISS_PATH):

            print(f"\n❌ Vector Store not found at: {DB_FAISS_PATH}")
            return None

        logger.info("Loading Existing FAISS Vector Store...")

        db = FAISS.load_local(
            folder_path=DB_FAISS_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )

        logger.info("✅ Vector Store Loaded Successfully.")

        return db

    except Exception as e:

        logger.error("Failed to Load Vector Store")
        traceback.print_exc()

        print("\n========== VECTOR STORE ERROR ==========")
        print(e)
        print("========================================\n")

        return None