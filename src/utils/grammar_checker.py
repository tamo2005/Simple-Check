# src/utils/grammar_checker.py
from transformers import pipeline

class GrammarChecker:
    def __init__(self):
        self.grammar_pipeline = pipeline("text2text-generation", model="t5-base")

    def check_grammar(self, text):
        """Refine grammar and structure of the text."""
        refined_text = self.grammar_pipeline(f"fix grammar: {text}", max_length=512, truncation=True)
        return refined_text[0]['generated_text']
