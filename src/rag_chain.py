import os

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


class RAGChain:
    """
    A class for Retrieval-Augmented Generation (RAG) that connects prompt templates,
    a language model (OpenAI GPT or Google Gemini), and an output parser to generate answers based on context.

    This class covers the "generation" half of RAG. Retrieval (finding the
    relevant documents) is done upstream by CustomIndex in src/indexer.py; the
    retrieved text is passed in to `query()` as `documents`.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0,
        api_key: str | None = None,
        provider: str = "openai",
    ):
        """
        Initialize the RAGChain with the specified model, temperature, and LLM provider.

        Args:
            model_name (str): The model identifier to use (e.g. 'gpt-4o', 'gemini-2.0-flash').
            temperature (float): Sampling temperature (0 = deterministic).
            api_key (str, optional): API key for the chosen provider. If omitted, reads from
                                     OPENAI_API_KEY (for OpenAI) or GEMINI_API_KEY/GOOGLE_API_KEY (for Gemini).
            provider (str): LLM provider to use: 'openai' (default) or 'gemini'.
        """
        self.provider = provider.lower() if provider else "openai"
        self.model_name = model_name
        self.temperature = temperature
        self.chain = None
        self._prompt = None

        if self.provider == "gemini" or "gemini" in model_name.lower():
            if ChatGoogleGenerativeAI is None:
                raise ImportError(
                    "langchain-google-genai is required for Gemini models. "
                    "Install it via `pip install langchain-google-genai`."
                )
            self.api_key = (
                api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or "placeholder-gemini-key"
            )
            gemini_model = model_name if "gemini" in model_name.lower() else "gemini-2.0-flash"
            self.llm = ChatGoogleGenerativeAI(
                model=gemini_model,
                temperature=temperature,
                api_key=self.api_key,
            )
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "sk-placeholder-key-for-initialization"
            self.llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=self.api_key)

    @staticmethod
    def _standardize_inputs(inputs: dict) -> dict:
        """Normalize the input dict into exactly the keys the prompt expects."""
        context = inputs.get("context", "")
        # Support both 'question' and 'query' keys for flexible invocation
        question = inputs.get("question") or inputs.get("query", "")
        return {"context": context, "question": question}

    def create_chain(self):
        """
        Create a chain: prompt → LLM → output parser.

        Attribute:
            self.chain (RunnableSequence): A LangChain pipeline.
        """
        template = (
            "You are a helpful customer support assistant for a B2B SaaS platform.\n"
            "Answer the question concisely and accurately using ONLY the provided context.\n"
            "If the answer cannot be determined from the context, or if no relevant context is found, "
            "respond with: 'I am sorry, but I do not have enough information to answer your question.'\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )
        self._prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.chain = (
            RunnableLambda(self._standardize_inputs)
            | self._prompt
            | self.llm
            | StrOutputParser()
        )
        return self.chain

    def query(self, query: str, documents: str | list) -> str:
        """
        Generate an answer for the given query using provided context documents.

        Args:
            query (str): The question to answer.
            documents (str | list): Supporting context for the query, either as a single string
                                    or a list of strings.

        Returns:
            str: Generated answer.

        Raises:
            ValueError: If inputs are invalid.
        """
        # --- Input validation ---
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(documents, (str, list)):
            raise ValueError("Documents must be a string or a list of strings.")

        # --- Flatten the documents into a single context string for the prompt ---
        if isinstance(documents, list):
            if not all(isinstance(doc, str) for doc in documents):
                raise ValueError("All elements in documents list must be strings.")
            context = "\n".join(doc.strip() for doc in documents if doc.strip())
        else:
            context = documents.strip()

        # Guardrail: If no relevant context is found, return the default fallback answer immediately
        if not context:
            return "I am sorry, but I do not have enough information to answer your question."

        if self.chain is None:
            self.create_chain()

        try:
            response = self.chain.invoke({"context": context, "question": query})
            return response.strip()
        except Exception as e:
            err_msg = str(e)
            # If a Gemini model name returned 404 NOT_FOUND on v1beta, try supported alternatives automatically
            if self.provider == "gemini" and ("404" in err_msg or "NOT_FOUND" in err_msg):
                candidate_fallbacks = [
                    "gemini-2.0-flash",
                    "gemini-2.5-flash",
                    "gemini-1.5-flash-latest",
                    "gemini-1.5-pro-latest",
                ]
                for alt_model in candidate_fallbacks:
                    if alt_model == self.model_name:
                        continue
                    try:
                        alt_llm = ChatGoogleGenerativeAI(
                            model=alt_model,
                            temperature=self.temperature,
                            api_key=self.api_key,
                        )
                        alt_chain = (
                            RunnableLambda(self._standardize_inputs)
                            | self._prompt
                            | alt_llm
                            | StrOutputParser()
                        )
                        alt_response = alt_chain.invoke({"context": context, "question": query})
                        # Cache the successful model
                        self.llm = alt_llm
                        self.chain = alt_chain
                        self.model_name = alt_model
                        return alt_response.strip()
                    except Exception:  # noqa: BLE001, S112
                        continue
            raise
