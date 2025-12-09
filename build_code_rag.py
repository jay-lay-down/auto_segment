"""Utility to build a Chroma-backed RAG index for the AutoSegmentTool codebase.

The script reads Python files, chunks them with language-aware boundaries, stores them
in a local Chroma vector store, and exposes a small CLI chat loop that can answer
questions about the code with cited source paths.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

DEFAULT_EXCLUDE_DIRS = {"__pycache__", ".git", "chroma_db", ".idea", ".vscode"}
DEFAULT_EXTENSIONS = {".py"}


def normalize_extensions(extensions: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for ext in extensions:
        suffix = ext.lower()
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        normalized.append(suffix)
    return normalized


def iter_source_files(project_root: Path, extensions: Sequence[str]) -> Iterable[Path]:
    """Yield source files under project_root that match the provided extensions."""

    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        if any(part in DEFAULT_EXCLUDE_DIRS or part.startswith(".") for part in path.parts):
            continue
        yield path


def load_documents(project_root: Path, extensions: Sequence[str]) -> List[Document]:
    """Load project files into LangChain Documents with relative source metadata."""

    documents: List[Document] = []
    for path in iter_source_files(project_root, extensions):
        text = path.read_text(encoding="utf-8")
        metadata = {"source": str(path.relative_to(project_root))}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def chunk_documents(documents: Sequence[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    return splitter.split_documents(documents)


def build_vector_store(documents: Sequence[Document], persist_dir: Path) -> Chroma:
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=str(persist_dir)
    )


def ensure_persist_dir(persist_dir: Path, rebuild: bool) -> None:
    if persist_dir.exists() and rebuild:
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)


def create_qa_chain(vector_store: Chroma, model: str, temperature: float, k: int):
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "fetch_k": max(k * 3, 10)}
    )

    system_prompt = (
        "You are a helpful assistant for the AutoSegmentTool codebase. "
        "Use the provided code context to answer questions and cite the source paths."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}\n\nContext:\n{context}"),
        ]
    )

    llm = ChatOpenAI(model=model, temperature=temperature)
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


def chat_loop(chain, prompt: str) -> None:
    print(prompt)
    while True:
        user_input = input("\nAsk about the code (or type 'exit'): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        result = chain.invoke({"input": user_input})
        print("\nAnswer:\n" + result["answer"].strip())
        sources = result.get("context", [])
        if sources:
            print("\nSources:")
            for doc in sources:
                print(f"- {doc.metadata.get('source')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or reuse a local Chroma index for the AutoSegmentTool codebase and "
            "ask questions via a simple CLI."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory containing the source code to index.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "chroma_db",
        help="Directory to store the Chroma index.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector store from scratch, replacing any existing index.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="Chat model name for answering questions.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="LLM temperature setting."
    )
    parser.add_argument(
        "--k", type=int, default=6, help="Number of documents to retrieve for answers."
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=sorted(DEFAULT_EXTENSIONS),
        help="File extensions to include (default: .py).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_persist_dir(args.persist_dir, args.rebuild)
    extensions = normalize_extensions(args.extensions)
    documents = load_documents(args.project_root, extensions)
    chunked_documents = chunk_documents(documents)
    vector_store = build_vector_store(chunked_documents, args.persist_dir)

    chain = create_qa_chain(vector_store, args.model, args.temperature, args.k)
    chat_loop(
        chain,
        prompt=(
            "Ready. Ensure OPENAI_API_KEY is set in your environment before running.\n"
            f"Indexed {len(chunked_documents)} chunks from {args.project_root}"
        ),
    )


if __name__ == "__main__":
    main()
