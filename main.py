import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load data
df = pd.read_csv('./RecipesWithLocationGI.csv')

# Clean and prepare
df = df.drop_duplicates('title')
df = df.dropna(subset=['title', 'ingridients'])
df = df.reset_index(drop=True)

# Clean and split ingredients
df['ingridients'] = df['ingridients'].str.replace(r'[^\w\s,]', '', regex=True)
df['ingridients'] = df['ingridients'].str.split(',|\n')
df['ingridients'] = df['ingridients'].apply(lambda x: [i.strip().lower() for i in x if i.strip()])

# Join ingredients back into a string for CountVectorizer
df['ingridients_str'] = df['ingridients'].apply(lambda x: ' '.join(x))

# Create CountVectorizer and compute cosine similarity
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['ingridients_str'])
cosine_sim = cosine_similarity(count_matrix)

def get_recommendations_from_ingredients(ingredients, cosine_sim=cosine_sim, df=df, cv=cv):
    # Convert ingredients to lowercase
    ingredients = [ing.strip().lower() for ing in ingredients]
    
    # Create a string from input ingredients
    input_ingredients = ' '.join(ingredients)
    
    # Transform input ingredients using the fitted CountVectorizer
    input_vector = cv.transform([input_ingredients])
    
    # Compute cosine similarity between input and all recipes
    sim_scores = cosine_similarity(input_vector, count_matrix)[0]
    
    # Sort recipes by similarity score
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 10 similar recipes
    sim_scores = sim_scores[:10]
    recipe_indices = [i[0] for i in sim_scores]
    
    return df[['title', 'ingridients', 'preparation']].iloc[recipe_indices]

@app.route('/')
def index():
    return "welcome to the recipe recommender, start by sending a POST request to /recommend with a JSON body containing an array of ingredients"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    ingredients = data.get('ingredients', [])
    
    if not ingredients:
        return jsonify({"error": "Please provide at least one ingredient"}), 400
    
    recommendations = get_recommendations_from_ingredients(ingredients)
    
    recommendations_list = []
    for i, (title, ingredients, preparation) in enumerate(zip(recommendations['title'], recommendations['ingridients'], recommendations['preparation']), 1):
        recipe = {
            "id": i,
            "title": title,
            "ingredients": ingredients,
            "preparation": preparation
        }
        recommendations_list.append(recipe)
    
    return jsonify(recommendations_list)

if __name__ == '__main__':
    app.run(debug=True)