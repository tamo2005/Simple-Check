import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np

class AdvancedRAGModel:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        # Initialize embedding model and FAISS index
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        self.document_embeddings = None

    def add_documents(self, documents):
        """
        Add documents to the semantic search index
        
        Args:
            documents (list): List of text documents to index
        """
        # Encode documents
        embeddings = self.embedding_model.encode(documents)
        
        # Add to FAISS index
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.document_embeddings = embeddings

    def retrieve_context(self, query, k=3):
        """
        Retrieve most semantically similar documents
        
        Args:
            query (str): Input query
            k (int): Number of top documents to retrieve
        
        Returns:
            list: Most relevant document contexts
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return top k documents
        return [self.documents[i] for i in indices[0]]

    def generate_answer(self, question, api_key):
        """
        Generate a comprehensive answer using RAG approach
        
        Args:
            question (str): Input question
            api_key (str): Gemini API key
        
        Returns:
            str: Generated answer based on retrieved context
        """
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Retrieve relevant context
        contexts = self.retrieve_context(question)
        combined_context = " ".join(contexts)
        
        # Construct enhanced prompt
        prompt = f"""You are an expert technical assistant designed to provide detailed answers based on given context. The answer should not contain bullet points and line breaks. Generate an answer for 10 marks question.

Context:
{combined_context}

Question: {question}

Instructions:
- Provide a straightforward answer without any formatting, new lines, or bold text.
- Ensure the answer is clear and concise.

Answer:"""
        
        try:
            # Use Gemini Pro for generation
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt, 
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500
                )
            )
            return response.text.strip()  # Strip any leading/trailing whitespace
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"

def main():
    # Initialize RAG model
    rag_model = AdvancedRAGModel()
    
    # Comprehensive technical documents
    documents = [
    # Data Structures Deep Dive
    "A stack is a linear data structure that follows the LIFO (Last In, First Out) principle. This means that the last element added to the stack is the first one to be removed. The stack supports two primary operations: push and pop. Push adds an element to the top of the stack, while pop removes the element from the top. Stacks are commonly implemented using arrays or linked lists. In a stack, elements are processed in a strict order—first in, last out. The LIFO behavior is particularly useful in scenarios such as function call management, where the most recent function call should be processed first. Another notable application of stacks is in expression evaluation, such as converting infix expressions to postfix or evaluating arithmetic expressions. The backtracking algorithm also relies on the stack to store intermediate states while searching for solutions, making it a critical component in algorithms like depth-first search (DFS). In addition to these applications, stacks are also used in parsing syntax trees, undo operations in software, and in certain networking protocols like TCP for managing data buffers.",
    
    "A queue is a linear data structure that follows the FIFO (First In, First Out) principle. This means the first element added to the queue is the first one to be removed, unlike a stack where the last element added is the first to be removed. The queue supports two main operations: enqueue (add) and dequeue (remove). Enqueue adds an element to the back of the queue, while dequeue removes the element from the front of the queue. This structure is used extensively in task scheduling, where tasks are handled in the order they are received, ensuring fairness and order. In graph algorithms like breadth-first search (BFS), queues are crucial for exploring nodes level by level, ensuring that all nodes at a particular depth are processed before moving deeper into the graph. Queues are also used in managing asynchronous data processing, such as in messaging systems or event-driven architectures, where tasks or data messages are processed in the order they arrive. Other applications include job scheduling in operating systems, printer task management, and implementing buffers in various communication protocols.",
    
    "Graphs are a versatile and complex data structure consisting of nodes (also called vertices) connected by edges. Each edge represents a relationship or connection between two nodes. Graphs can be categorized into several types, including directed and undirected graphs, weighted and unweighted graphs, and cyclic and acyclic graphs. In a directed graph, edges have a direction, meaning they go from one node to another, whereas in an undirected graph, the relationship between nodes is bidirectional. In weighted graphs, each edge has a cost or weight associated with it, which is important in applications like finding the shortest path between two nodes. Graphs are essential in network modeling, as they represent the relationships between various entities such as computers, routers, or individuals in a social network. Pathfinding algorithms, such as Dijkstra’s algorithm and A* search, rely on graphs to find the most efficient routes between points. Additionally, graphs are used in representing complex systems like transportation networks, communication systems, and even biological networks such as protein interactions.",
    
    # Algorithms and Computational Techniques
    "Binary search is one of the most efficient algorithms for searching an element in a sorted list or array. It operates by repeatedly dividing the search space in half, making it much faster than a linear search, especially for large datasets. The binary search algorithm works by comparing the middle element of the sorted list with the target value. If the middle element is equal to the target, the search ends. If the target is less than the middle element, the search continues in the left half of the array, and if the target is greater, the search continues in the right half. This division process continues until the target is found or the search space is exhausted. The time complexity of binary search is O(log n), where n is the number of elements in the list, making it extremely efficient for large datasets. Binary search is used in various applications, such as searching for an element in a large database, in finding the position to insert an element in a sorted array, and in more complex algorithms like finding the first or last occurrence of a number in a sorted list.",
    
    "Depth-first search (DFS) is a graph traversal algorithm that explores as far as possible along each branch before backtracking. It starts at a root node and explores each branch of the graph as deeply as possible before moving on to the next branch. This behavior makes DFS particularly useful in tasks such as topological sorting, solving mazes, and finding strongly connected components in directed graphs. DFS can be implemented using a stack, either explicitly or via recursion. In DFS, once a node is visited, its adjacent nodes are visited in a depth-first manner, and if no unvisited adjacent nodes are found, the algorithm backtracks to explore other branches. DFS is also used in algorithms that require exploration of all possible configurations, such as puzzle-solving algorithms like the eight-puzzle problem or in game tree exploration for decision-making in AI. However, DFS is not guaranteed to find the shortest path in a graph, unlike breadth-first search (BFS), making it more suited for problems where path length is not a priority.",
    
    "Breadth-first search (BFS) is a graph traversal algorithm that explores all neighboring nodes at the present depth level before moving on to nodes at the next depth level. This level-by-level exploration makes BFS optimal for finding the shortest path between two nodes in an unweighted graph. BFS starts at a root node and explores all adjacent nodes (or neighbors), then explores the neighbors' neighbors, and so on. BFS can be implemented using a queue, where nodes are added to the queue and processed in the order they are dequeued. One of the key applications of BFS is in the shortest path problem, where it is used to determine the minimum number of edges between two nodes in an unweighted graph. BFS is also crucial in network broadcasting, where data needs to be sent to all reachable nodes in a network. Other applications of BFS include finding connected components in a graph, solving puzzles, and analyzing social networks to find clusters of interconnected individuals.",
    
    # Software Design Principles
    "Object-oriented programming (OOP) is a programming paradigm that organizes software design around objects, which are instances of classes. Each object contains both data, in the form of attributes or properties, and behavior, in the form of methods or functions that can manipulate the data. The primary principles of OOP include encapsulation, inheritance, and polymorphism. Encapsulation refers to the bundling of data and methods that operate on the data within a single unit, typically a class, and restricting access to the internal state of the object through access modifiers (like public, private, and protected). Inheritance allows one class to inherit the attributes and methods of another, enabling code reuse and the creation of hierarchies of classes. Polymorphism enables objects of different classes to be treated as instances of the same class through shared interfaces, allowing for more flexible and maintainable code. OOP helps in structuring complex systems, making the software more modular, easier to maintain, and reusable. It is used in many programming languages, including Java, Python, C++, and C#, and forms the foundation of most modern software development practices.",
    
    "Microservices architecture is a design pattern that breaks down applications into small, independent services that can be developed, deployed, and scaled independently. Each microservice is responsible for a specific functionality or domain and communicates with other microservices through APIs or messaging protocols. This architecture offers several benefits, including improved scalability, flexibility, and maintainability. Because each microservice operates independently, teams can develop and deploy them in isolation, enabling faster development cycles and more efficient use of resources. Microservices also enable organizations to adopt different technologies for different services, allowing for more flexibility in choosing the best tools for each task. However, microservices come with their own set of challenges, such as increased complexity in managing inter-service communication, data consistency, and distributed transactions. Despite these challenges, microservices have become the preferred architecture for many large-scale applications, particularly in cloud computing environments, where scalability and reliability are crucial.",
    
    "Design patterns are general, reusable solutions to common problems that occur in software design. These patterns represent best practices that have been developed and refined by experienced software engineers over time. They provide a proven approach to solving specific types of problems, making development faster and more efficient. Design patterns can be categorized into three main types: creational, structural, and behavioral. Creational patterns deal with the process of object creation, such as the Singleton, Factory, and Abstract Factory patterns. Structural patterns focus on the composition of classes and objects, like the Adapter, Composite, and Proxy patterns. Behavioral patterns are concerned with communication between objects and include patterns like Observer, Strategy, and Command. By using design patterns, developers can avoid reinventing the wheel and instead focus on implementing the unique aspects of their applications. Design patterns have been widely adopted in object-oriented programming and are essential for developing flexible, maintainable, and scalable software systems."
]

    
    # Add documents to RAG model
    rag_model.add_documents(documents)
    
    # Questions covering various technical domains
    questions = [
        "What is a stack in computer science?",
        "Explain how breadth-first search works",
        "Describe object-oriented programming principles",
        "What are microservices in software architecture?",
        "How do graphs represent complex relationships?"
    ]
    
    # Gemini API key (replace with your actual key)
    api_key = "AIzaSyAN6hW728nRG6ITJWMSJYQzSrkaEDeseOk"
    
    # Generate answers
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = rag_model.generate_answer(question, api_key)
        print("Answer:", answer)

if __name__ == "__main__":
    main()