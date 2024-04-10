import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Load your dataset
df = pd.read_csv('mergedata.csv')  # Replace 'path_to_your_file.csv' with the actual path to your CSV file

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Ingredients'], df['Dish'], test_size=0.25, random_state=42)

# Creating a text classification pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Training the model
model.fit(X_train, y_train)

# Function to predict the dish name based on ingredients
def predict_dish(ingredients):
    prediction = model.predict([ingredients])
    return prediction[0]

# Function to take user input and predict dish name
def predict_from_user_input():
    while True:
        user_input = input("Enter a list of ingredients (separated by commas), or 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        else:
            predicted_dish = predict_dish(user_input)
            print(f'Predicted Dish: {predicted_dish}\n')

# Call the function to predict from user input
predict_from_user_input()


joblib.dump(model, 'text_classification_model.joblib')
joblib.dump(model.named_steps['tfidfvectorizer'], 'tfidf_vectorizer.joblib')