# src/utils/spelling_corrector.py
from spellchecker import SpellChecker

class SpellingCorrector:
    def __init__(self):
        self.spell = SpellChecker()

    def correct_spelling(self, text):
        """Correct spelling errors in the text."""
        words = text.split()
        corrected_words = [self.spell.correction(word) if word not in self.spell else word for word in words]
        return " ".join(corrected_words)
