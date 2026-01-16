from src.utils.logging import get_logger

logger = get_logger("scoring_utils")

def adjust_keyword_weightage(keyword_scores):
    """
    Reassign weightage from missing keywords to the next available keyword with a more lenient approach.
    """
    total_weight = sum(keyword_scores.values())
    missing_keywords = [k for k, v in keyword_scores.items() if v < 0.1]  # More lenient threshold
    
    if missing_keywords and total_weight > 0:
        remaining_keywords = {k: v for k, v in keyword_scores.items() if v >= 0.1}
        weight_to_redistribute = sum(keyword_scores[k] for k in missing_keywords)
        
        if remaining_keywords:
            weight_per_remaining = weight_to_redistribute / len(remaining_keywords)
            for k in remaining_keywords:
                remaining_keywords[k] += weight_per_remaining
            return remaining_keywords
        return keyword_scores
    
    return keyword_scores

def calculate_final_score(nlp_scores, keyword_scores, max_marks):
    """
    Calculate final score with more lenient scoring criteria.
    """
    try:
        # Adjusted weights to emphasize concept understanding over exact matches
        weights = {
            'semantic': 0.35,  # Increased from 0.3
            'keyword': 0.4,    # Decreased from 0.5
            'structure': 0.25  # Increased from 0.2
        }

        # Adjust keyword weightage with more leniency
        adjusted_keyword_scores = adjust_keyword_weightage(keyword_scores)
        
        # Calculate keyword score with partial matching
        total_possible_keyword_score = sum(keyword_scores.values())
        if total_possible_keyword_score == 0:
            keyword_score = 0
        else:
            # Add bonus for partial matches
            partial_matches = sum(v for v in adjusted_keyword_scores.values() if 0.1 <= v < 0.8)
            keyword_score = (sum(adjusted_keyword_scores.values()) + (partial_matches * 0.2)) / total_possible_keyword_score * 100

        # Calculate semantic score with more emphasis on entity similarity
        semantic_score = (
            nlp_scores['entity_similarity'] * 0.7 +  # Increased from 0.6
            nlp_scores['syntax_similarity'] * 0.3    # Decreased from 0.4
        )

        # Calculate structure score with less emphasis on exact syntax
        structure_score = (
            nlp_scores['syntax_similarity'] * 0.6 +     # Decreased from 0.7
            nlp_scores['sentiment_similarity'] * 0.4    # Increased from 0.3
        )

        # Apply minimum score threshold for partially correct answers
        if keyword_score > 30 or semantic_score > 30:  # If either score shows partial understanding
            semantic_score = max(semantic_score, 25)    # Ensure minimum semantic score
            keyword_score = max(keyword_score, 25)      # Ensure minimum keyword score

        # Calculate final percentage
        final_percentage = (
            semantic_score * weights['semantic'] +
            keyword_score * weights['keyword'] +
            structure_score * weights['structure']
        )

        # Apply minimum threshold for final percentage
        if final_percentage > 20:  # If answer shows some understanding
            final_percentage = max(final_percentage, 25)  # Ensure minimum score

        # Convert to marks
        final_score = (final_percentage / 100) * max_marks

        return round(final_score, 2), {
            'semantic_score': round(semantic_score, 2),
            'keyword_score': round(keyword_score, 2),
            'structure_score': round(structure_score, 2),
            'final_percentage': round(final_percentage, 2),
            'relevance_score': nlp_scores.get('relevance_score', 0.0)
        }

    except Exception as e:
        logger.error(f"Error in calculating final score: {e}")
        raise RuntimeError(f"Error in score calculation: {e}")