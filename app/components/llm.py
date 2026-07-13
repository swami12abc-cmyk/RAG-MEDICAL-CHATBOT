from langchain_groq import ChatGroq
from app.config.config import GROQ_API_KEY
from app.common.logger import get_logger
import traceback

logger = get_logger(__name__)

def load_llm(
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY
):
    try:
        print("=" * 50)
        print("Groq API Key:", groq_api_key)
        print("Model:", model_name)

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=256,
        )

        print("✅ LLM Loaded Successfully")

        return llm

    except Exception as e:
        print("\n❌ REAL ERROR:")
        traceback.print_exc()
        print("\nError:", e)
        return None