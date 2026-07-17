"""The end-to-end RAG loop: route -> retrieve -> generate.

The router (src/router.py) picks a path per query. Most questions take the
semantic-search 'qa' path; structural/section/summary queries take a direct
path so they aren't lost to content-only similarity search.
"""
from src.retrieve import Retriever
from src.generate import Generator
from src.router import route
from config import TOP_K


class RAGPipeline:
    def __init__(self, index_name="paper"):
        self.retriever = Retriever(index_name)
        self.generator = Generator()

    def ask(self, question, k=TOP_K):
        kind, arg = route(question)

        if kind == "metadata":
            return self._result(question, self._metadata_answer(question), [], "metadata")

        if kind == "section":
            chunks = self.retriever.by_section(arg)
            if chunks:  # fall through to qa if the section wasn't tagged
                return self._result(question, self.generator.answer(question, chunks),
                                    chunks, "section")

        if kind == "summary":
            chunks = self.retriever.summary_chunks()
            return self._result(question, self.generator.summarize(question, chunks),
                                chunks, "summary")

        chunks = self.retriever.retrieve(question, k=k)
        return self._result(question, self.generator.answer(question, chunks), chunks, "qa")

    def _metadata_answer(self, question):
        meta = self.retriever.store.meta
        q = question.lower()
        if "page" in q or "how long" in q:
            n = meta.get("page_count")
            return f"This paper is {n} pages long." if n else "Page count is unavailable."
        if "author" in q or "who wrote" in q:
            a = meta.get("author")
            return f"Author(s): {a}" if a else "The PDF does not embed author metadata."
        if "title" in q:
            t = meta.get("title")
            return f"Title: {t}" if t else "The PDF does not embed a title in its metadata."
        return "That document-level detail is unavailable."

    @staticmethod
    def _result(question, answer, chunks, route_name):
        return {"question": question, "answer": answer, "chunks": chunks, "route": route_name}
