import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load FAQ dataset
faq_data = pd.read_csv("faq_dataset.csv")  # Ensure your CSV has 'Question' and 'Answer' columns

# Preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

faq_data['Processed_Question'] = faq_data['Question'].apply(preprocess_text)

# Vectorize questions using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faq_data['Processed_Question'])

# Function to find the best matching FAQ
def get_best_match(user_query):
    user_query_processed = preprocess_text(user_query)
    user_query_vector = vectorizer.transform([user_query_processed])
    similarity_scores = cosine_similarity(user_query_vector, tfidf_matrix)
    best_match_idx = similarity_scores.argmax()
    best_match_score = similarity_scores[0, best_match_idx]
    
    if best_match_score > 0.3:  # Confidence threshold
        return faq_data.iloc[best_match_idx]['Answer']
    else:
        return "I'm sorry, I couldn't find a relevant answer. Could you rephrase your question?"

# Chatbot interaction
def chatbot():
    print("Welcome to the FAQ Chatbot! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = get_best_match(user_input)
        print(f"Chatbot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
