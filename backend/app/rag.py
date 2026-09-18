from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

class RAGService:
    def __init__(self, persist_directory: str, model: str) -> None:
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.store = Chroma(collection_name="enterprise_knowledge", embedding_function=self.embeddings, persist_directory=persist_directory)
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.history: dict[str, list[tuple[str, str]]] = {}

    def _load(self, path: Path, filename: str) -> list[Document]:
        if path.suffix == ".pdf":
            reader = PdfReader(str(path))
            return [Document(page_content=page.extract_text() or "", metadata={"filename": filename, "page": index + 1}) for index, page in enumerate(reader.pages)]
        return [Document(page_content=path.read_text(errors="ignore"), metadata={"filename": filename})]

    def ingest(self, path: Path, filename: str) -> int:
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=140)
        chunks = [c for c in splitter.split_documents(self._load(path, filename)) if c.page_content.strip()]
        if chunks:
            self.store.add_documents(chunks)
        return len(chunks)

    def answer(self, question: str, session_id: str) -> tuple[str, list[dict]]:
        results = self.store.similarity_search_with_relevance_scores(question, k=5)
        context = "\n\n".join(f"[Source {i}] {doc.metadata.get('filename')} page {doc.metadata.get('page', 'n/a')}\n{doc.page_content}" for i, (doc, _) in enumerate(results, 1))
        recent = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in self.history.get(session_id, [])[-4:])
        prompt = f"""Conversation:
{recent or "No earlier messages."}

Retrieved sources:
{context or "No sources were retrieved."}

Question: {question}

Answer only from retrieved sources. If unsupported, say so. Cite claims with [Source N]."""
        response = self.llm.invoke([SystemMessage(content="You are a careful enterprise knowledge assistant. Never invent sources."), HumanMessage(content=prompt)])
        answer = str(response.content)
        self.history.setdefault(session_id, []).append((question, answer))
        sources = [{"filename": d.metadata.get("filename", "unknown"), "page": d.metadata.get("page"), "excerpt": d.page_content[:260].strip(), "score": round(float(s), 4)} for d, s in results]
        return answer, sources

    def clear(self) -> None:
        self.store.delete_collection()
        self.store = Chroma(collection_name="enterprise_knowledge", embedding_function=self.embeddings, persist_directory=self.persist_directory)
