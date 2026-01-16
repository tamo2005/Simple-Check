import argparse
import sys
from pathlib import Path
from src.utils.logging import get_logger
from nlp_analyzer import NLPHandler

logger = get_logger("grader_cli")

def adjust_keyword_weightage(keyword_scores):
    """
    Reassign weightage from missing keywords to the next available keyword.
    """
    total_weight = sum(keyword_scores.values())
    missing_keywords = [k for k, v in keyword_scores.items() if v == 0]
    
    if missing_keywords and total_weight > 0:
        remaining_keywords = {k: v for k, v in keyword_scores.items() if v > 0}
        weight_to_redistribute = sum(keyword_scores[k] for k in missing_keywords)
        
        # Redistribute weight to existing keywords proportionally
        for k in remaining_keywords:
            remaining_keywords[k] += (remaining_keywords[k] / total_weight) * weight_to_redistribute
        
        return remaining_keywords
    
    return keyword_scores

def calculate_final_score(nlp_scores, keyword_scores, max_marks):
    """
    Calculate final score based on semantic similarity, keyword weightage, and structure.
    """
    # Weights for different components
    weights = {
        'semantic': 0.3,  # Semantic similarity (entities & syntax)
        'keyword': 0.5,  # Keyword weightage-based scoring (Increased importance)
        'structure': 0.2  # Sentence structure & sentiment
    }
    
    # Adjust keyword weightage to redistribute marks from missing keywords
    adjusted_keyword_scores = adjust_keyword_weightage(keyword_scores)
    
    # Compute weighted keyword score
    keyword_score = sum(adjusted_keyword_scores.values()) / (sum(keyword_scores.values()) or 1) * 100
    
    # Compute semantic similarity score
    semantic_score = (
        nlp_scores['entity_similarity'] * 0.6 +  # Entity match
        nlp_scores['syntax_similarity'] * 0.4    # Syntax match
    )
    
    # Compute structure score
    structure_score = (
        nlp_scores['syntax_similarity'] * 0.7 +
        nlp_scores['sentiment_similarity'] * 0.3
    )
    
    # Final weighted score
    final_percentage = (
        semantic_score * weights['semantic'] +
        keyword_score * weights['keyword'] +
        structure_score * weights['structure']
    )
    
    # Convert percentage to marks
    final_score = (final_percentage / 100) * max_marks
    
    return round(final_score, 2), {
        'semantic_score': round(semantic_score, 2),
        'keyword_score': round(keyword_score, 2),
        'structure_score': round(structure_score, 2),
        'final_percentage': round(final_percentage, 2)
    }

def print_detailed_analysis(scores, keyword_scores, max_marks, final_score, detailed_scores):
    """Print detailed analysis of the grading"""
    print("\nAnswer Analysis Report")
    print("=" * 50)
    
    print(f"\nFinal Score: {final_score}/{max_marks} ({detailed_scores['final_percentage']}%)")
    
    print("\nDetailed Scoring Breakdown:")
    print(f"1. Semantic Analysis: {detailed_scores['semantic_score']}%")
    print(f"   - Entity Similarity: {scores['entity_similarity']}%")
    print(f"   - Syntax Similarity: {scores['syntax_similarity']}%")
    
    print(f"\n2. Keyword Matching: {detailed_scores['keyword_score']}%")
    print("   Keyword Weightage Breakdown:")
    for keyword, weight in keyword_scores.items():
        print(f"   ✓ {keyword}: {round(weight, 2)}")
    
    print(f"\n3. Structural Analysis: {detailed_scores['structure_score']}%")
    print(f"   - Sentence Structure: {scores['syntax_similarity']}%")
    print(f"   - Writing Style Match: {scores['sentiment_similarity']}%")
    
    print("\nPlagiarism Check:")
    plag_prob = scores['plagiarism_probability']
    if plag_prob > 80:
        print(f"⚠️  High similarity detected ({plag_prob}%) - Please review")
    else:
        print(f"✓ Acceptable similarity level ({plag_prob}%)")

def main():
    parser = argparse.ArgumentParser(description='Grade exam answers using NLP analysis')
    parser.add_argument('--model', required=True, help='Model answer text')
    parser.add_argument('--answer', required=True, help='Student\'s answer text')
    parser.add_argument('--marks', type=float, required=True, help='Maximum marks for the question')
    parser.add_argument('--details', action='store_true', help='Show detailed analysis')
    
    args = parser.parse_args()
    
    try:
        # Initialize NLP Handler
        handler = NLPHandler()
        
        # Analyze answers
        nlp_scores = handler.analyze_text(args.answer, args.model)
        
        # Generate keyword weightage from model answer
        keyword_scores, _ = handler.extract_keyword_weightage(args.model, args.answer)
        
        # Calculate final score
        final_score, detailed_scores = calculate_final_score(nlp_scores, keyword_scores, args.marks)
        
        # Output results
        if args.details:
            print_detailed_analysis(nlp_scores, keyword_scores, args.marks, final_score, detailed_scores)
        else:
            print(f"\nFinal Score: {final_score}/{args.marks}")
            print(f"Percentage: {detailed_scores['final_percentage']}%")
            
    except Exception as e:
        logger.error(f"Error in grading: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())