# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# import joblib
# import numpy as np
#
# # Load your dataset
# df = pd.read_csv('mergedata.csv')  # Replace 'path_to_your_file.csv' with the actual path to your CSV file
#
# # Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df['Ingredients'], df['Dish'], test_size=0.25, random_state=42)
#
# # Creating a TF-IDF Vectorizer to convert text data to numerical features
# vectorizer = TfidfVectorizer()
# X_train_vectorized = vectorizer.fit_transform(X_train)
# X_test_vectorized = vectorizer.transform(X_test)
#
# # Creating and training the Decision Tree classifier
# dt_classifier = DecisionTreeClassifier()
# dt_classifier.fit(X_train_vectorized, y_train)
#
# # Save the trained model and TF-IDF Vectorizer
# joblib.dump(dt_classifier, 'decision_tree_modelm24.joblib')
# joblib.dump(vectorizer, 'tfidf_vectorizerm24.joblib')
#
# # Function to predict the dish name based on ingredients
# def predict_dish(ingredients):
#     ingredients_vectorized = vectorizer.transform([ingredients])
#     prediction = dt_classifier.predict(ingredients_vectorized)
#     return prediction[0]
#
# # Function to take user input and predict dish name
# def predict_from_user_input():
#     while True:
#         user_input = input("Enter a list of ingredients (separated by commas), or 'exit' to quit: ")
#         if user_input.lower() == 'exit':
#             break
#         else:
#             predicted_dish = predict_dish(user_input)
#             print(f'Predicted Dish: {predicted_dish}\n')
#
# # Call the function to predict from user input
# predict_from_user_input()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load your dataset
df = pd.read_csv('mergedata.csv')  # Replace 'path_to_your_file.csv' with the actual path to your CSV file

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Ingredients'], df['Dish'], test_size=0.25, random_state=42)

# Creating a TF-IDF Vectorizer to convert text data to numerical features
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Creating and training the Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_vectorized, y_train)

# Save the trained model and TF-IDF Vectorizer
joblib.dump(dt_classifier, 'decision_tree_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# Function to predict the dish name based on ingredients
def predict_dish(ingredients):
    ingredients_vectorized = vectorizer.transform([ingredients])
    prediction = dt_classifier.predict(ingredients_vectorized)
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


