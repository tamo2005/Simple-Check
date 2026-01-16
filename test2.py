from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example answers
model_answer = "A stack is a linear data structure that follows the LIFO (Last In, First Out) principle, where elements are added using push and removed using pop. It is used in function calls, undo/redo, and expression evaluation."
student_answer = "A FIFO follows stack principle, where elements are queued and dequeued. It is used in function calls."

def evaluate_semantic_similarity(model_answer, student_answer, full_marks=5, threshold_high=0.8, threshold_mid=0.6):
    """Evaluate semantic similarity using Sentence-BERT."""
    model_embedding = model.encode(model_answer, convert_to_numpy=True)
    student_embedding = model.encode(student_answer, convert_to_numpy=True)
    
    similarity_score = cosine_similarity([model_embedding], [student_embedding])[0][0]
    
    # Assign scores based on thresholds
    if similarity_score > threshold_high:
        score = full_marks
    elif similarity_score > threshold_mid:
        score = full_marks * (similarity_score - threshold_mid) / (threshold_high - threshold_mid)
    else:
        score = 0
    
    return round(score, 2), round(similarity_score, 2)

def extract_keywords(text):
    """Extract keywords from a text using CountVectorizer."""
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit([text])
    return set(vectorizer.get_feature_names_out())

def evaluate_keyword_matching(model_answer, student_answer, full_marks=5):
    """Evaluate keyword matching by comparing common keywords."""
    model_keywords = extract_keywords(model_answer)
    print("model:", model_keywords)
    student_keywords = extract_keywords(student_answer)
    print("student:", student_keywords)
    
    # Find common keywords
    common_keywords = model_keywords.intersection(student_keywords)
    
    # Calculate score based on keyword overlap
    total_keywords = len(model_keywords)
    matched_keywords = len(common_keywords)
    
    if total_keywords == 0:
        return 0, []
    
    score = (matched_keywords / total_keywords) * full_marks
    return round(score, 2), list(common_keywords)

def combined_evaluation(model_answer, student_answer, weight_semantic=0.5, weight_keywords=0.5):
    """Combine semantic similarity and keyword matching scores."""
    # Evaluate semantic similarity
    semantic_score, semantic_similarity = evaluate_semantic_similarity(model_answer, student_answer)
    
    # Evaluate keyword matching
    keyword_score, matched_keywords = evaluate_keyword_matching(model_answer, student_answer)
    
    # Combine scores with weights
    combined_score = (weight_semantic * semantic_score) + (weight_keywords * keyword_score)
    
    return round(combined_score, 2), semantic_similarity, matched_keywords

# Test the combined evaluation
final_score, semantic_sim, matched_keywords = combined_evaluation(model_answer, student_answer)
print(f"Final Score: {final_score}")
print(f"Semantic Similarity: {semantic_sim}")
print(f"Matched Keywords: {matched_keywords}")
