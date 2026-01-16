from sklearn.feature_extraction.text import CountVectorizer

# Example answers
model_answer = "A stack is a linear data structure that follows the LIFO (Last In, First Out) principle, where elements are added using push and removed using pop. It is used in function calls, undo/redo, and expression evaluation."
student_answer = "A stack follows FIFO principle, where elements are queued and dequeued. It is used in function calls."

def extract_keywords(text):
    """Extract keywords from a text using CountVectorizer."""
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit([text])
    return set(vectorizer.get_feature_names_out())

def evaluate_keywords(model_answer, student_answer, full_marks=5):
    # Extract keywords from both answers
    model_keywords = extract_keywords(model_answer)
    print("model :", model_keywords)
    student_keywords = extract_keywords(student_answer)
    print("student :", student_keywords)
    
    # Find matching keywords
    common_keywords = model_keywords.intersection(student_keywords)
    
    # Calculate score based on keyword overlap
    total_keywords = len(model_keywords)
    matched_keywords = len(common_keywords)
    
    # Avoid division by zero
    if total_keywords == 0:
        return 0, []
    
    score = (matched_keywords / total_keywords) * full_marks
    return round(score, 2), list(common_keywords)

# Test the function
score, matched_keywords = evaluate_keywords(model_answer, student_answer)
print(f"Score: {score}")
print(f"Matched Keywords: {matched_keywords}")
