import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize Sentence-BERT model and lemmatizer
model = SentenceTransformer('all-MiniLM-L6-v2')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess text by lowercasing, removing special characters, and lemmatizing."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation and special characters
    words = word_tokenize(text)  # Tokenize text
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(words)

def evaluate_semantic_similarity(model_answer, student_answer, full_marks, threshold_high=0.8, threshold_mid=0.6):
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

def evaluate_keyword_matching(model_answer, student_answer, full_marks):
    """Evaluate keyword matching by comparing common keywords."""
    model_keywords = extract_keywords(model_answer)
    student_keywords = extract_keywords(student_answer)
    
    # Find common keywords
    common_keywords = model_keywords.intersection(student_keywords)
    
    # Calculate score based on keyword overlap
    total_keywords = len(model_keywords)
    matched_keywords = len(common_keywords)
    
    if total_keywords == 0:
        return 0, []
    
    score = (matched_keywords / total_keywords) * full_marks
    return round(score, 2), list(common_keywords)

def combined_evaluation(model_answer, student_answer, full_marks):
    """Combine semantic similarity and keyword matching scores with dynamic weights."""
    # Preprocess texts
    model_answer_clean = preprocess_text(model_answer)
    student_answer_clean = preprocess_text(student_answer)
    
    # Adjust weights based on full_marks
    if full_marks <= 3:
        weight_keywords = 0.6  # Higher weight for keywords for low marks
        weight_semantic = 0.4
    elif 3 < full_marks <= 10:
        weight_keywords = 0.5  # Equal weight for medium marks
        weight_semantic = 0.5
    else:
        weight_keywords = 0.3  # Higher weight for semantic analysis for high marks
        weight_semantic = 0.7
    
    # Evaluate semantic similarity
    semantic_score, semantic_similarity = evaluate_semantic_similarity(model_answer_clean, student_answer_clean, full_marks)
    
    # Evaluate keyword matching
    keyword_score, matched_keywords = evaluate_keyword_matching(model_answer_clean, student_answer_clean, full_marks)
    
    # Combine scores with dynamic weights
    combined_score = (weight_semantic * semantic_score) + (weight_keywords * keyword_score)
    
    return round(combined_score, 2), semantic_similarity, matched_keywords

# Example answers
model_answer = "Graphs represent complex relationships by connecting nodes with edges, where nodes represent entities and edges represent the connections between them. This structure allows for the modeling of various networks, such as social networks, computer networks, and biological networks, where the entities and their interconnections are crucial for understanding the system's behavior."
student_answer = "Graphs are a fundamental data structure used to represent complex relationships between entities. In a graph, nodes (also known as vertices) represent individual entities, while edges (also called arcs) represent the connections or relationships between these entities. The beauty of graphs lies in their ability to model a wide variety of systems where connections between entities are critical for understanding the overall structure and behavior of the system.One of the most common applications of graphs is in social networks. In this context, nodes represent individuals or groups, and edges represent relationships such as friendships, following, or messaging. Social media platforms like Facebook, Twitter, and LinkedIn use graph structures to model how people are connected. For example, in Facebook, each user is a node, and each friendship is represented by an edge between two nodes. Understanding the relationships between individuals can help platforms recommend friends, identify influential users, or detect communities of like-minded individuals.Graphs are also widely used in computer networks. Here, the nodes represent devices such as computers, routers, or switches, and the edges represent communication links between them, such as wired or wireless connections. In computer networking, graphs help model how data flows from one device to another, enabling network administrators to optimize routing paths, detect bottlenecks, and ensure that the network is both efficient and resilient.Another important area where graphs are utilized is in biological networks. In this domain, graphs are used to model complex interactions within biological systems. For instance, in a protein-protein interaction network, each protein is represented by a node, and the edges represent interactions between proteins. These graphs are valuable in studying disease mechanisms, drug discovery, and understanding cellular processes. By analyzing these graphs, scientists can identify key proteins that play crucial roles in specific biological functions or diseases.Graphs can also be classified into different types based on how the edges are structured. For example, a directed graph has edges that have a direction, indicating that the relationship between two nodes is one-way, while an undirected graph represents two-way relationships. A weighted graph assigns a weight or cost to each edge, representing the strength or distance of the connection, which is particularly useful in pathfinding algorithms like Dijkstra's algorithm. In summary, graphs are a powerful tool for representing complex systems with interconnected entities. From social and computer networks to biological systems, graphs help us model, analyze, and understand the underlying structure of various domains, making them an essential concept in computer science, mathematics, and engineering."

# Test the combined evaluation for different full_marks values
low_marks = 2
high_marks = 10

# Low marks example
print("For low marks:")
low_score, low_sim, low_keywords = combined_evaluation(model_answer, student_answer, full_marks=low_marks)
print(f"Final Score: {low_score}")
print(f"Semantic Similarity: {low_sim}")
print(f"Matched Keywords: {low_keywords}\n")

# High marks example
print("For high marks:")
high_score, high_sim, high_keywords = combined_evaluation(model_answer, student_answer, full_marks=high_marks)
print(f"Final Score: {high_score}")
print(f"Semantic Similarity: {high_sim}")
print(f"Matched Keywords: {high_keywords}")
