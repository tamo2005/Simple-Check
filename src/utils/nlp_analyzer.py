from google.cloud import language_v1
from google.cloud.language_v1 import Document
from src.utils.logging import get_logger
from difflib import SequenceMatcher

logger = get_logger("nlp_handler_logs")

class NLPHandler:
    def __init__(self, credentials_path="config/nlp_key.json"):
        """Initialize the NLP Handler with Google Cloud credentials"""
        self.client = language_v1.LanguageServiceClient.from_service_account_json(credentials_path)
        self.RELEVANCE_THRESHOLD = 0.10  # Lowered from 0.15
        self.KEYWORD_MATCH_THRESHOLD = 0.80
        logger.info("Successfully initialized Language API client")
    
    def analyze_text(self, text, reference_text):
        """
        Analyze text against a reference text using various NLP metrics with more lenient scoring.
        """
        try:
            # Clean and normalize texts
            text = text.lower().strip()
            reference_text = reference_text.lower().strip()

            # Create documents
            doc = Document(content=text, type_=Document.Type.PLAIN_TEXT)
            ref_doc = Document(content=reference_text, type_=Document.Type.PLAIN_TEXT)

            # Get basic NLP analysis
            sentiment = self.client.analyze_sentiment(document=doc).document_sentiment
            ref_sentiment = self.client.analyze_sentiment(document=ref_doc).document_sentiment
            
            entities = self.client.analyze_entities(document=doc).entities
            ref_entities = self.client.analyze_entities(document=ref_doc).entities
            
            syntax = self.client.analyze_syntax(document=doc)
            ref_syntax = self.client.analyze_syntax(document=ref_doc)

            # Calculate relevance with multiple metrics
            sequence_similarity = SequenceMatcher(None, text, reference_text).ratio()
            
            # Get word overlap
            text_words = set(text.split())
            ref_words = set(reference_text.split())
            word_overlap = len(text_words.intersection(ref_words)) / len(ref_words) if ref_words else 0.0

            # Calculate entity overlap with partial matching
            text_entities = {(e.name.lower(), e.type_) for e in entities}
            ref_entities_set = {(e.name.lower(), e.type_) for e in ref_entities}
            
            # Find exact and partial matches
            exact_matches = text_entities.intersection(ref_entities_set)
            partial_matches = set()
            
            for t_name, t_type in text_entities:
                for r_name, r_type in ref_entities_set:
                    if t_type == r_type and (
                        t_name in r_name or 
                        r_name in t_name or 
                        SequenceMatcher(None, t_name, r_name).ratio() > 0.8
                    ):
                        partial_matches.add((t_name, t_type))

            all_matches = exact_matches.union(partial_matches)
            entity_similarity = len(all_matches) / len(ref_entities_set) if ref_entities_set else 0.0

            # Calculate final relevance score
            relevance_score = (
                0.4 * entity_similarity +
                0.3 * sequence_similarity +
                0.3 * word_overlap
            )

            # Only consider irrelevant if score is very low
            if relevance_score < self.RELEVANCE_THRESHOLD:
                return {
                    'entity_similarity': 0.0,
                    'sentiment_similarity': 0.0,
                    'syntax_similarity': 0.0,
                    'relevance_score': round(relevance_score * 100, 2),
                    'is_relevant': False
                }

            # Calculate sentiment similarity with tolerance
            sentiment_diff = abs(sentiment.score - ref_sentiment.score)
            sentiment_similarity = max(0, (1 - sentiment_diff) * 100)

            # Calculate syntax similarity with partial credit
            text_pos = [token.part_of_speech.tag for token in syntax.tokens]
            ref_pos = [token.part_of_speech.tag for token in ref_syntax.tokens]
            
            # Use sliding window for syntax matching
            max_syntax_match = 0
            window_size = min(len(text_pos), len(ref_pos))
            
            for i in range(len(text_pos) - window_size + 1):
                window_match = sum(1 for a, b in zip(text_pos[i:i+window_size], ref_pos) if a == b)
                max_syntax_match = max(max_syntax_match, window_match)
            
            syntax_similarity = (max_syntax_match / len(ref_pos)) * 100 if ref_pos else 0

            # Get common and missing entities for feedback
            common_entities = [name for name, _ in all_matches]
            missing_entities = [name for name, _ in ref_entities_set if name not in [n for n, _ in all_matches]]

            return {
                'entity_similarity': round(entity_similarity * 100, 2),
                'sentiment_similarity': round(sentiment_similarity, 2),
                'syntax_similarity': round(syntax_similarity, 2),
                'relevance_score': round(relevance_score * 100, 2),
                'common_entities': common_entities,
                'missing_entities': missing_entities,
                'is_relevant': True
            }

        except Exception as e:
            logger.error(f"Error in analyzing text: {e}")
            raise RuntimeError(f"Error in NLPHandler: {e}")
    
    def extract_keyword_weightage(self, text, reference_text):
        """
        Extract and weight keywords from text with more lenient matching.
        Returns both the keyword scores and extracted keywords.
        """
        try:
            # Create documents
            ref_doc = Document(content=reference_text, type_=Document.Type.PLAIN_TEXT)
            student_doc = Document(content=text, type_=Document.Type.PLAIN_TEXT)

            # Get reference keywords and their salience
            ref_response = self.client.analyze_entities(document=ref_doc)
            ref_keywords = {
                entity.name.lower(): {
                    'salience': entity.salience,
                    'type': entity.type_,
                    'mentions': len(entity.mentions)
                }
            for entity in ref_response.entities}

            # Get student keywords
            student_response = self.client.analyze_entities(document=student_doc)
            student_keywords = {
                entity.name.lower(): {
                    'type': entity.type_,
                    'mentions': len(entity.mentions)
                }
            for entity in student_response.entities}

            # Initialize keyword scores
            keyword_scores = {k: 0.0 for k in ref_keywords.keys()}
            extracted_keywords = []

            # Process exact and partial matches
            for ref_key, ref_data in ref_keywords.items():
                best_match_score = 0.0
                best_match_key = None

                # Check for exact matches first
                if ref_key in student_keywords:
                    # Calculate score based on exact match
                    score = ref_data['salience']
                    # Bonus for correct type and multiple mentions
                    if student_keywords[ref_key]['type'] == ref_data['type']:
                        score *= 1.1
                    mention_ratio = min(student_keywords[ref_key]['mentions'] / ref_data['mentions'], 1.5)
                    score *= mention_ratio
                    keyword_scores[ref_key] = score
                    extracted_keywords.append(ref_key)
                    continue

                # Check for partial matches
                for student_key in student_keywords:
                    # Skip if already matched exactly
                    if student_key in extracted_keywords:
                        continue

                    # Calculate string similarity
                    similarity = SequenceMatcher(None, ref_key, student_key).ratio()
                    
                    # Check if words are substrings of each other
                    if ref_key in student_key or student_key in ref_key:
                        similarity = max(similarity, 0.85)

                    # Consider type matching for technical terms
                    if (similarity > self.KEYWORD_MATCH_THRESHOLD and 
                        student_keywords[student_key]['type'] == ref_data['type']):
                        
                        # Calculate partial match score
                        score = ref_data['salience'] * similarity
                        if score > best_match_score:
                            best_match_score = score
                            best_match_key = ref_key

                # Update scores for best partial match
                if best_match_key and best_match_score > keyword_scores[best_match_key]:
                    keyword_scores[best_match_key] = best_match_score
                    if best_match_key not in extracted_keywords:
                        extracted_keywords.append(best_match_key)

            # Normalize scores
            total_salience = sum(kw['salience'] for kw in ref_keywords.values())
            if total_salience > 0:
                for key in keyword_scores:
                    keyword_scores[key] = keyword_scores[key] / total_salience

            return keyword_scores, extracted_keywords

        except Exception as e:
            logger.error(f"Error in extracting keyword weightage: {e}")
            raise RuntimeError(f"Error in keyword extraction: {e}")

    def get_document_similarity(self, text, reference_text):
        """Calculate overall document similarity using multiple metrics."""
        try:
            # Clean and normalize texts
            text = text.lower().strip()
            reference_text = reference_text.lower().strip()

            # Get entity-based similarity
            doc = Document(content=text, type_=Document.Type.PLAIN_TEXT)
            ref_doc = Document(content=reference_text, type_=Document.Type.PLAIN_TEXT)

            entities = self.client.analyze_entities(document=doc).entities
            ref_entities = self.client.analyze_entities(document=ref_doc).entities

            text_entities = set((e.name.lower(), e.type_) for e in entities)
            ref_entities = set((e.name.lower(), e.type_) for e in ref_entities)

            # Calculate Jaccard similarity for entities
            if ref_entities:
                intersection = len(text_entities.intersection(ref_entities))
                union = len(text_entities.union(ref_entities))
                entity_similarity = intersection / union if union > 0 else 0.0
            else:
                entity_similarity = 0.0

            # Calculate sequence similarity using difflib
            sequence_similarity = SequenceMatcher(None, text, reference_text).ratio()

            # Calculate word overlap
            text_words = set(text.split())
            ref_words = set(reference_text.split())
            word_overlap = len(text_words.intersection(ref_words)) / len(ref_words) if ref_words else 0.0

            # Combine similarities with weights
            combined_similarity = (
                0.4 * entity_similarity +  # Entity matching
                0.3 * sequence_similarity +  # Overall text similarity
                0.3 * word_overlap  # Word overlap
            )

            return combined_similarity

        except Exception as e:
            logger.error(f"Error in calculating document similarity: {e}")
            return 0.0

    # ... (rest of the class methods remain the same)