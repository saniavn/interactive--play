import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import numpy as np

# Load dataset
df = pd.read_csv('mergedata.csv')  # Replace 'path_to_your_file.csv' with the actual path to your CSV file

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Ingredients'], df['Dish'], test_size=0.25, random_state=42)

# Creating a text classification pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Training the model
model.fit(X_train, y_train)

# Save the trained model and TF-IDF Vectorizer
#joblib.dump(model, 'text_classification_model.joblib')
#joblib.dump(model.named_steps['tfidfvectorizer'], 'tfidf_vectorizer.joblib')

# Function to predict the dish names and probabilities based on ingredients
# def predict_dish_with_probabilities(ingredients, top_n=3):
#     predictions = model.predict_proba([ingredients])[0]
#     classes = model.classes_
#     top_indices = predictions.argsort()[-top_n:][::-1]
#     top_dishes_with_probabilities = [(classes[i], round(predictions[i]*100, 2)) for i in top_indices]
#     return top_dishes_with_probabilities
# Function to predict the dish names and probabilities based on ingredients
def predict_dish_with_probabilities(ingredients, top_n=3):
    predictions = model.predict_proba([ingredients])[0]
    classes = model.classes_
    top_indices = predictions.argsort()[-top_n:][::-1]

    # Use softmax to normalize the probabilities
    exp_scores = np.exp(predictions[top_indices])
    probabilities = exp_scores / np.sum(exp_scores)

    top_dishes_with_probabilities = [(classes[top_indices[i]], round(probabilities[i] * 100, 2)) for i in
                                     range(len(top_indices))]
    return top_dishes_with_probabilities

# Function to take user input and predict dish names with probabilities
def predict_from_user_input():
    while True:
        user_input = input("Enter a list of ingredients (separated by commas), or 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        else:
            top_dishes_with_probabilities = predict_dish_with_probabilities(user_input)
            print("Top predicted dishes with probabilities:")
            for dish, probability in top_dishes_with_probabilities:
                print(f"Dish: {dish}, Probability: {probability}%")
            print()

# Call the function to predict from user input
predict_from_user_input()
