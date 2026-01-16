from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient model

# Example answers
model_answer = "A stack is a linear data structure that follows the LIFO (Last In, First Out) principle, where elements are added using push and removed using pop. It is used in function calls, undo/redo, and expression evaluation."
student_answer = "A stack follows FIFO principle, where elements are queued and dequeued.It is used in function calls"

# Generate embeddings
model_embedding = model.encode(model_answer, convert_to_numpy=True)
student_embedding = model.encode(student_answer, convert_to_numpy=True)

from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity
similarity_score = cosine_similarity([model_embedding], [student_embedding])[0][0]

print(f"Similarity Score: {similarity_score:.2f}")

def evaluate_answer(model_answer, student_answer, full_marks=5, threshold_high=0.8, threshold_mid=0.6):
    # Generate embeddings
    model_embedding = model.encode(model_answer, convert_to_numpy=True)
    student_embedding = model.encode(student_answer, convert_to_numpy=True)
    
    # Compute cosine similarity
    similarity_score = cosine_similarity([model_embedding], [student_embedding])[0][0]
    
    # Assign scores based on thresholds
    if similarity_score > threshold_high:
        score = full_marks
    elif similarity_score > threshold_mid:
        score = full_marks * (similarity_score - threshold_mid) / (threshold_high - threshold_mid)
    else:
        score = 0  # Minimal marks for low similarity
    
    return round(score, 2), round(similarity_score, 2)

# Test the function
score, sim = evaluate_answer(model_answer, student_answer)
print(f"Score: {score}, Similarity: {sim}")
