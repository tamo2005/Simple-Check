from google.cloud import language_v1
from google.cloud.language_v1 import Document
from src.utils.logging import get_logger

logger = get_logger("nlp_handler_logs")

class NLPHandler:
    def __init__(self, credentials_path="config/nlp_key.json"):
        """Initialize the NLP Handler with Google Cloud credentials"""
        self.client = language_v1.LanguageServiceClient.from_service_account_json(credentials_path)
        logger.info("Successfully initialized Language API client")

    def analyze_text(self, text, reference_text):
        """
        Analyze text against a reference text using various NLP metrics
        """
        try:
            doc = Document(content=text, type_=Document.Type.PLAIN_TEXT)
            ref_doc = Document(content=reference_text, type_=Document.Type.PLAIN_TEXT)

            sentiment = self.client.analyze_sentiment(document=doc).document_sentiment
            ref_sentiment = self.client.analyze_sentiment(document=ref_doc).document_sentiment

            entities = self.client.analyze_entities(document=doc).entities
            ref_entities = self.client.analyze_entities(document=ref_doc).entities

            syntax = self.client.analyze_syntax(document=doc)
            ref_syntax = self.client.analyze_syntax(document=ref_doc)

            keyword_weights, keyword_score = self.extract_keyword_weightage(text, reference_text)

            similarity_scores = self.calculate_similarity(
                entities, ref_entities, syntax, ref_syntax, sentiment, ref_sentiment, keyword_weights, keyword_score
            )

            return similarity_scores

        except Exception as e:
            logger.error(f"Error in analyzing text: {e}")
            raise RuntimeError(f"Error in NLPHandler: {e}")

    def extract_keyword_weightage(self, text, reference_text):
        try:
            doc = Document(content=reference_text, type_=Document.Type.PLAIN_TEXT)
            response = self.client.analyze_entities(document=doc)
            
            keyword_weights = {entity.name.lower(): entity.salience for entity in response.entities}
            
            student_doc = Document(content=text, type_=Document.Type.PLAIN_TEXT)
            student_response = self.client.analyze_entities(document=student_doc)
            student_keywords = {entity.name.lower() for entity in student_response.entities}

            matched_weight = sum(keyword_weights.get(word, 0) for word in student_keywords)
            total_weight = sum(keyword_weights.values()) or 1

            keyword_score = (matched_weight / total_weight) * 100
            return keyword_weights, round(keyword_score, 2)

        except Exception as e:
            logger.error(f"Error in extracting keyword weightage: {e}")
            raise RuntimeError(f"Error in keyword weightage extraction: {e}")

    def calculate_similarity(self, entities, ref_entities, syntax, ref_syntax, sentiment, ref_sentiment, keyword_weights, keyword_score):
        try:
            
            text_entities = {entity.name.lower() for entity in entities}
            ref_text_entities = {entity.name.lower() for entity in ref_entities}
            common_entities = text_entities.intersection(ref_text_entities)
            missing_entities = ref_text_entities - text_entities

            entity_score = sum(keyword_weights.get(e, 0) for e in common_entities) / (sum(keyword_weights.values()) or 1) * 100
            
            sentiment_diff = abs(sentiment.score - ref_sentiment.score)
            sentiment_score = max(0, (1 - sentiment_diff) * 100)
            
            text_pos = [token.part_of_speech.tag for token in syntax.tokens]
            ref_pos = [token.part_of_speech.tag for token in ref_syntax.tokens]
            common_pos = sum(1 for a, b in zip(text_pos, ref_pos) if a == b)
            syntax_score = (common_pos / max(len(ref_pos), 1)) * 100
            
            plagiarism_score = (entity_score * 0.5 + sentiment_score * 0.2 + syntax_score * 0.3)
            
            return {
                'entity_similarity': round(entity_score, 2),
                'sentiment_similarity': round(sentiment_score, 2),
                'syntax_similarity': round(syntax_score, 2),
                'plagiarism_probability': round(plagiarism_score, 2),
                'common_entities': list(common_entities),
                'missing_entities': list(missing_entities),
                'keyword_score': keyword_score  
            }
        
        except Exception as e:
            logger.error(f"Error in calculating similarity: {e}")
            raise RuntimeError(f"Error in similarity calculation: {e}")