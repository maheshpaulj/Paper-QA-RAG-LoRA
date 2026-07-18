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
            answer = self._metadata_answer(question)
            if answer:
                return self._result(question, answer, [], "metadata")
            # Not in the PDF's metadata -- but authors/title are printed on the
            # title page, so read them out of the front matter instead of giving
            # up. Only page count is metadata-only, and that path always answers.
            chunks = self.retriever.front_matter()
            return self._result(question, self.generator.answer(question, chunks),
                                chunks, "metadata+text")

        if kind == "section":
            chunks = self.retriever.by_section(arg)
            if chunks:  # fall through to qa if the section wasn't tagged
                return self._result(question, self.generator.answer(question, chunks),
                                    chunks, "section")

        if kind == "figure":
            chunks = self.retriever.by_figure(*arg)
            if chunks:  # unknown number -> fall through to semantic search
                return self._result(question, self.generator.answer(question, chunks),
                                    chunks, "figure")

        if kind == "summary":
            chunks = self.retriever.summary_chunks()
            return self._result(question, self.generator.summarize(question, chunks),
                                chunks, "summary")

        chunks = self.retriever.retrieve(question, k=k)
        return self._result(question, self.generator.answer(question, chunks), chunks, "qa")

    def _metadata_answer(self, question):
        """Answer from the PDF's embedded metadata, or None to read the paper's
        own front matter instead.

        Only page_count is trustworthy here -- PyMuPDF computes it. The author
        and title fields are whatever the authoring tool wrote and are routinely
        absent, truncated or plain wrong: GPT-3's PDF lists two of its ~31
        authors, and not the first one. The title page is authoritative, so
        author/title always go to the text.
        """
        q = question.lower()
        if "page" in q or "how long" in q:
            n = self.retriever.store.meta.get("page_count")
            return f"This paper is {n} pages long." if n else None
        return None

    @staticmethod
    def _result(question, answer, chunks, route_name):
        return {"question": question, "answer": answer, "chunks": chunks, "route": route_name}
