# innacurate, needs to be updated
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Step 1: Prepare the data
# Assume we have a CSV file with columns: index, title, ingredients

# Load the data
data = pd.read_csv('./RecipesWithLocationGI.csv')

# Create a MultiLabelBinarizer to convert ingredients to binary format
mlb = MultiLabelBinarizer()
ingredient_matrix = mlb.fit_transform(data['ingridients'].apply(lambda x: x.split(',')))

# Step 2: Create the model
input_dim = len(mlb.classes_)
encoding_dim = 32

input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = tf.keras.models.Model(input_layer, decoded)
encoder = tf.keras.models.Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Step 3: Train the model
X_train, X_test = train_test_split(ingredient_matrix, test_size=0.2, random_state=42)
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# Step 4: Create a function to recommend recipes
def recommend_recipes(ingredients, top_n=5):
    # Convert input ingredients to binary format
    input_ingredients = mlb.transform([ingredients])
    
    # Get the encoded representation of all recipes
    all_recipes_encoded = encoder.predict(ingredient_matrix)
    
    # Get the encoded representation of the input ingredients
    input_encoded = encoder.predict(input_ingredients)
    
    # Calculate cosine similarity between input and all recipes
    similarities = tf.keras.losses.cosine_similarity(all_recipes_encoded, input_encoded).numpy()
    
    # Get the indices of the top N most similar recipes
    top_indices = np.argsort(similarities)[:top_n]
    
    # Return the recommended recipes
    return data.iloc[top_indices][['title', 'ingridients']]

# Example usage
ingredients = ['chicken', 'tomato', 'onion']
recommendations = recommend_recipes(ingredients)
print(recommendations)