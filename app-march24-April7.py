from flask import Flask, request, jsonify, render_template, redirect, url_for
from joblib import load
import os
import json
import numpy as np
from sklearn.preprocessing import normalize
from flask import jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.naive_bayes import MultinomialNB
import tempfile
import shutil
from flask import session
from sklearn.model_selection import train_test_split

import uuid
import os
import tempfile


app = Flask(__name__, static_url_path='/static')
print("Flask app root path:", app.root_path)

# Load the pre-trained model and the dish-to-recipe mapping
food_rec_model = load('trained_modelmarch14d5k.joblib')
with open('dish_to_recipe.json') as f:
    dish_to_recipe = json.load(f)


new_text_model = load('models/text_classification_model.joblib')
tfidf_vectorizer = load('models/tfidf_vectorizer.joblib')

new_text_model3 = load('models-RecipeNLG/text_classification_modelRecipeNLG.joblib')
tfidf_vectorizer3 = load('models-RecipeNLG/tfidf_vectorizerRecipeNLG.joblib')

# Load your dataset
df = pd.read_csv('mergedata.csv')

@app.route('/')
def home():
    return render_template('student_info.html')

@app.route('/submit_student_info', methods=['POST'])
def submit_student_info():
    # Extract and sanitize user input
    student_name = request.form.get('student_name', 'default').replace(' ', '_')
    student_age = request.form.get('student_age', 'default')
    student_grade = request.form.get('student_grade', 'default')
    user_file = f"{student_name}_{student_age}_{student_grade}.json"
    print(f"Received - Name: {student_name}, Age: {student_age}, Grade: {student_grade}")

    # Define the path for the user's file within the 'user_data' directory
    user_data_path = os.path.join(app.root_path, 'user_data', user_file)

    # Data to be written to the file
    user_data = {'name': student_name, 'age': student_age, 'grade': student_grade, 'interactions': []}

    # Write the data to a JSON file
    with open(user_data_path, 'w') as f:
        json.dump(user_data, f)

    # Redirect to another page after saving the data
    return redirect(url_for('ingredient_interaction', user_file=user_file))

@app.route('/ingredient_interaction', methods=['GET', 'POST'])
def ingredient_interaction():
    user_file = request.args.get('user_file', 'default_user.json')
    print("Received user file for ingredient interaction:", user_file)
    return render_template('ingredient_interaction.html', user_file=user_file)


### Random forest on large dataset I got from Kaggle 5000 rows
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_file = data.get('user_file')
    user_ingredients = data.get('ingredients', [])

    if not user_ingredients:
        return jsonify({'error': 'Missing required data: ingredients'}), 400

    # Prediction logic
    user_input = ' '.join(user_ingredients)
    user_input_transformed = food_rec_model['tfidfvectorizer'].transform([user_input])
    predictions = food_rec_model['randomforestclassifier'].predict_proba(user_input_transformed)
    class_labels = food_rec_model['randomforestclassifier'].classes_
    top_dishes = sorted(zip(class_labels, predictions[0]), key=lambda x: x[1], reverse=True)[:3]

    # Prepare response without recipes
    response_data = {
        'predictions': [{
            'dish': dish,
            'probability': round(prob * 100, 2)
        } for dish, prob in top_dishes],
        'model': 'Model A'
    }

    # Update user file with new interaction
    user_data_path = os.path.join(app.root_path, 'user_data', user_file)
    try:
        if os.path.exists(user_data_path):
            with open(user_data_path, 'r+') as f:
                user_data = json.load(f)
                user_data['interactions'].append({
                    'ingredients': user_ingredients,
                    'predictions': response_data['predictions']
                })
                f.seek(0)
                json.dump(user_data, f)
                f.truncate()
    except Exception as e:
        return jsonify({'error': f'Failed to update user file: {e}'}), 500

    return jsonify(response_data)





#####Worked!!! My dataset on naiive bays,
@app.route('/predict2', methods=['POST'])
def predict2():
    data = request.get_json()
    user_file = data.get('user_file')
    user_ingredients = data.get('ingredients', [])

    if not user_ingredients:
        return jsonify({'error': 'Missing required data: ingredients'}), 400

    # Combine the user ingredients into a single string
    user_input = ' '.join(user_ingredients)

    # Perform the prediction directly with the pipeline
    predictions = new_text_model.predict_proba([user_input])
    class_labels = new_text_model.classes_
    top_dishes = sorted(zip(class_labels, predictions[0]), key=lambda x: x[1], reverse=True)[:3]

    # Normalize probabilities
    total_probability = sum(prob for _, prob in top_dishes)
    normalized_top_dishes = [(dish, prob / total_probability * 100) for dish, prob in top_dishes]

    # Prepare the response
    response_data = {
        'predictions': [{
            'dish': dish,
            'probability': round(prob, 2)
        } for dish, prob in normalized_top_dishes],
        'model': 'Model B'
    }

    # Update user data file as before
    user_data_path = os.path.join(app.root_path, 'user_data', user_file)
    try:
        if os.path.exists(user_data_path):
            with open(user_data_path, 'r+') as f:
                user_data = json.load(f)
                user_data['interactions'].append({
                    'ingredients': user_ingredients,
                    'predictions': response_data['predictions']
                })
                f.seek(0)
                json.dump(user_data, f)
                f.truncate()
    except Exception as e:
        return jsonify({'error': f'Failed to update user file: {e}'}), 500

    return jsonify(response_data)


@app.route('/predict3', methods=['POST'])
def predict3():
    data = request.get_json()
    user_file = data.get('user_file')
    user_ingredients = data.get('ingredients', [])

    if not user_ingredients:
        return jsonify({'error': 'Missing required data: ingredients'}), 400

    # Combine the user ingredients into a single string
    user_input = ' '.join(user_ingredients)

    # Perform the prediction directly with the pipeline
    predictions = new_text_model3.predict_proba([user_input])
    class_labels = new_text_model3.classes_
    top_dishes = sorted(zip(class_labels, predictions[0]), key=lambda x: x[1], reverse=True)[:3]

    # Normalize probabilities
    total_probability = sum(prob for _, prob in top_dishes)
    normalized_top_dishes = [(dish, prob / total_probability * 100) for dish, prob in top_dishes]

    # Prepare the response
    response_data = {
        'predictions': [{
            'dish': dish,
            'probability': round(prob, 2)
        } for dish, prob in normalized_top_dishes],
        'model': 'Model C'
    }

    # Update user data file as before
    user_data_path = os.path.join(app.root_path, 'user_data', user_file)
    try:
        if os.path.exists(user_data_path):
            with open(user_data_path, 'r+') as f:
                user_data = json.load(f)
                user_data['interactions'].append({
                    'ingredients': user_ingredients,
                    'predictions': response_data['predictions']
                })
                f.seek(0)
                json.dump(user_data, f)
                f.truncate()
    except Exception as e:
        return jsonify({'error': f'Failed to update user file: {e}'}), 500

    return jsonify(response_data)

## predict 4 my dadatset DT


@app.route('/classification_game')
def classification_game():
    return render_template('classification_game.html')  # Ensure this HTML exists

@app.route('/classification_game1')
def classification_game1():
    return render_template('classification_game1.html')  # Ensure this HTML exists

@app.route('/add_data', methods=['POST'])
def add_data():
    """
    Add new training data (ingredients and dish) to existing dataset.
    """
    data = request.get_json()
    ingredients = data['ingredients']
    dish_name = data['dishName']
    user_file = data['userFile']

    # Path to the CSV file where data is appended
    dataset_path = os.path.join(app.root_path, 'data', 'mergedata.csv')

    # Append new data
    with open(dataset_path, 'a') as f:
        f.write(f"\n{','.join([ingredients, dish_name])}")

    # Update user interaction file
    user_data_path = os.path.join(app.root_path, 'user_data', user_file)
    with open(user_data_path, 'r+') as f:
        user_data = json.load(f)
        user_data['interactions'].append({
            'ingredients': ingredients,
            'dishName': dish_name
        })
        f.seek(0)
        json.dump(user_data, f, indent=4)
        f.truncate()

    return jsonify({'message': 'Data added successfully'})

###training
@app.route('/train', methods=['POST'])
def train_model():
    # Path to the existing dataset
    dataset_path = os.path.join(app.root_path, 'data', 'mergedata.csv')

    # Load and prepare the dataset
    df = pd.read_csv(dataset_path)
    X = df['Ingredients'].apply(lambda x: x.lower())  # Ensure consistency in casing
    y = df['Dish']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and train the model
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    # Save the trained model
    model_path = os.path.join(app.root_path, 'models', 'food_rec_model.joblib')
    joblib.dump(model, model_path)

    return jsonify({'message': 'Model trained successfully'})

# Ensure to adjust your routes for prediction to load the correct, newly trained model

# @app.route('/train', methods=['POST'])
# def train_model():
#     # Generate a unique ID for this session
#     unique_id = str(uuid.uuid4())
#         return jsonify({'message': 'No training data provided'}), 400
#
#     # Write the received training data to a JSON file
#     with open(user_data_path, 'w') as f:
#         json.dump(trainingData, f, indent=4)
#
#     # Assuming 'mergedata.csv' contains the initial dataset
#     existing_data_path = 'mergedata.csv'
#     # Convert existing CSV data to a DataFrame
#     existing_df = pd.read_csv(existing_data_path)
#
#     # Load the new user data from the JSON file
#     with open(user_data_path, 'r') as f:
#         new_user_data = json.load(f)
#     # Convert the new user data to a DataFrame
#     user_df = pd.DataFrame(new_user_data)
#
#     # Merge the existing dataset with the new user data
#     combined_df = pd.concat([existing_df, user_df])
#
#     # Convert ingredient strings to lowercase to ensure consistency
#     combined_df['Ingredients'] = combined_df['Ingredients'].str.lower()
#
#     # Train a new model using the combined dataset
#     X_train, X_test, y_train, y_test = train_test_split(combined_df['Ingredients'], combined_df['Dish'], test_size=0.25, random_state=42)
#     user_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
#     user_model.fit(X_train, y_train)
#
#     # Save the trained model specific to this session
#     user_model_path = os.path.join(user_data_dir, 'user_model.joblib')
#     joblib.dump(user_model, user_model_path)
#
#     return jsonify({'message': 'Model trained successfully with your data'})


@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    data = request.get_json()
    ingredients = data.get('ingredients', [])
    ingredients_str = ', '.join(ingredients).lower()

    model_path = os.path.join(app.root_path, 'models', 'food_rec_model.joblib')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        return jsonify({'error': 'Model not found.'}), 500

    # Predict probabilities
    probabilities = model.predict_proba([ingredients_str])[0]
    class_labels = model.classes_

    # Get top 3 predictions
    top_indices = probabilities.argsort()[-3:][::-1]  # Indices of top 3 probabilities
    top_predictions = [{'dish': class_labels[i], 'probability': probabilities[i]} for i in top_indices]

    return jsonify(top_predictions)




@app.route('/reset_session', methods=['POST'])
def reset_session():
    data = request.get_json()
    unique_id = data.get('uniqueId')

    if unique_id:
        # Path to the directory where user-specific files are stored
        user_data_dir = os.path.join(tempfile.gettempdir(), unique_id)

        # Remove the directory and its contents
        if os.path.exists(user_data_dir):
            shutil.rmtree(user_data_dir)

        return jsonify({'message': 'Your session has been reset successfully'})
    else:
        return jsonify({'error': 'Unique ID is missing'}), 400


@app.route('/compare_guess', methods=['POST'])
def compare_guess():
    data = request.get_json()
    user_guess = data.get('guess')
    ingredients = ' '.join(data.get('ingredients', []))
    # Implement model's prediction logic here for comparison
    ai_prediction = ...  # Add the prediction logic based on ingredients
    correct = user_guess == ai_prediction  # Define correctness based on your criteria
    return jsonify({'correct': correct, 'aiPrediction': ai_prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5007)
