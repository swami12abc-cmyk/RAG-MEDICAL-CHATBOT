from app.components.pdf_loader import load_pdf_files, create_text_chunks
from app.components.vector_store import save_vector_store


def main():

    print("=" * 60)
    print("Loading PDF Files...")
    print("=" * 60)

    documents = load_pdf_files()

    if not documents:
        print("No PDF files found.")
        return

    print(f"Loaded {len(documents)} pages.")

    print("=" * 60)
    print("Creating Chunks...")
    print("=" * 60)

    text_chunks = create_text_chunks(documents)

    print(f"Generated {len(text_chunks)} chunks.")

    print("=" * 60)
    print("Creating FAISS Database...")
    print("=" * 60)

    save_vector_store(text_chunks)

    print("\n✅ Vector Database Created Successfully!")


if __name__ == "__main__":
    main()