# utils/document_intent_classifier.py

from custom_logging.my_logger import logger
from utils.model_loaders import ModelLoader


class DocumentIntentClassifier:
    """
    Hybrid intent classifier for document chat.
    """

    HARDCODE_RULES = {
        "CONVERSATION_HISTORY": [
            "show chat history", "conversation history", "what did we talk",
            "previous messages", "show last questions", "show the chat",
            "last conversation", "show previous chat",
        ],
        "CONVERSATION_SUMMARY": [
            "summarize our conversation", "chat summary", "summarize our chat",
            "what have we discussed",
        ],
        "DOCUMENT_SUMMARY": [
            "summarize the document", "document summary", "what is this document about",
            "overview of document", "give me summary of this file", "summarize file",
            "explain the document", "summarize content",
        ],
        "GREETING": ["hi", "hello", "hey", "good morning", "good evening"],
    }

    NON_DOCUMENT_KEYWORDS = [
        "bitcoin price", "current weather", "tell me a joke", "who is the president",
        "internet search", "stock price", "news today", "what is happening in the world",
        "outside the document", "world news"
    ]

    SYSTEM_PROMPT = """
You are a classifier. Your ONLY job is to output one label.

Labels:
1. DOCUMENT_QUERY
2. DOCUMENT_SUMMARY
3. CONVERSATION_SUMMARY
4. CONVERSATION_HISTORY
5. GREETING
6. NOT_RELATED_TO_DOCUMENT

Rules:
- Output ONLY the label.
- No explanation.
"""

    def __init__(self):
        self.llm = ModelLoader().load_llm()

    def _keyword_classify(self, text: str):
        text_low = text.lower()

        # Hardcoded rules
        for label, patterns in self.HARDCODE_RULES.items():
            for p in patterns:
                if p in text_low:
                    return label

        # Detect non-document queries
        for p in self.NON_DOCUMENT_KEYWORDS:
            if p in text_low:
                return "NOT_RELATED_TO_DOCUMENT"

        return None

    def _llm_classify(self, text: str):
        try:
            resp = self.llm.invoke([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ])
            label = resp.content.strip().upper()

            allowed = {
                "DOCUMENT_QUERY",
                "DOCUMENT_SUMMARY",
                "CONVERSATION_SUMMARY",
                "CONVERSATION_HISTORY",
                "GREETING",
                "NOT_RELATED_TO_DOCUMENT"
            }

            return label if label in allowed else "DOCUMENT_QUERY"

        except Exception as e:
            logger.error(f"[INTENT ERROR] LLM classify failed: {e}")
            return "DOCUMENT_QUERY"

    def classify(self, user_input: str) -> str:
        rule_match = self._keyword_classify(user_input)
        if rule_match:
            logger.info(f"[INTENT] Matched fast rule → {rule_match}")
            return rule_match

        label = self._llm_classify(user_input)
        logger.info(f"[INTENT] LLM classified → {label}")
        return label
