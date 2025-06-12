import joblib
import re

model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('tfdif_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_emotion(raw_text):
    cleaned = clean_text(raw_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction


# Interactive input from user
print("\nEmotion Predictor — type a sentence and I'll tell you the emotion.")
print("Type 'bye' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ['exit', 'quit', 'bye', 'thanks']:
        print("Exiting the Emotion Predictor. See you next time!")
        break

    emotion = predict_emotion(user_input)
    print(f"Predicted Emotion: {emotion}\n")
