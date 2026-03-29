import pickle
from utils import preprocess_text

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_sentiment(text):
    text = preprocess_text(text)
    
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

# Test
while True:
    text = input("Enter text: ")
    print("Sentiment:", predict_sentiment(text))
