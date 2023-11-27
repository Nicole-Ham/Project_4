
import pickle
import os
from flask import Flask
import requests
from fuzzywuzzy import process


 

def load_computed_data():
    print(f"opening computed_data.pkl file")
    if os.path.exists('computed_data.pkl'):
        with open('computed_data.pkl', 'rb') as file:
            return pickle.load(file)
    else:
        return None
    
    
# we used the jupyter notebook to create the saved_cmputed_model_data.pkl file which contains our trained model
# then we used the model inside the flask app to make predictions
def get_similar_game_names(input_game, num_suggestions=2,saved_computed_model_data=None)->list:
    print(f"app.get_similar_game_names input_game: {input_game}")
    
    # Extract relevant data from computed_data
    df_ml = saved_computed_model_data['df_ml']
    selected_features = saved_computed_model_data['selected_features']
    scaler = saved_computed_model_data['scaler']
    kmeans = saved_computed_model_data['kmeans']
    
    input_game_lower = input_game.lower()  # Converts the input to lowercase

    try:
        # Find the row for the input game (case-insensitive)
        input_game_row = df_ml[df_ml['game'].str.lower() == input_game_lower].iloc[0]
    except IndexError:
        # If there's no exact match, suggest similar names
        all_game_names_lower = df_ml['game'].str.lower().tolist()
        similar_names = process.extract(input_game_lower, all_game_names_lower, limit=num_suggestions)
        
        suggestions = [name for name, _ in similar_names]
        
        print(f"No exact match found for the game '{input_game}'. You can try:")
        for suggestion in suggestions:
            print(f"- {suggestion.capitalize()}")  # Capitalizes the suggestions for better readability
        # return {'alternative_game_names':suggestions,'similar_game_names':None}
        return [None,suggestions]

    # Get features for the input game
    input_game_features = input_game_row[selected_features].values.reshape(1, -1)

    # Predict the cluster for the input game
    predicted_cluster = kmeans.predict(scaler.transform(input_game_features))[0]

    # Filter similar games in the same cluster (excluding the input game itself)
    similar_games = df_ml[(df_ml['Cluster'] == predicted_cluster) & (df_ml['game'] != input_game_row['game'])].head(num_suggestions)

    # Extract names of similar games
    similar_game_names = similar_games['game'].tolist()

    return [ similar_game_names,None]


def get_game_info(game_title,api_key=os.environ.get('RAWG_API_KEY')):
    # Set up the API endpoint and parameters
    base_url = "https://api.rawg.io/api/games"
    params = {
        'key': api_key,
        'search': game_title,
        'page_size': 1  # Get only one result for simplicity
    }
    print(f"API Key: {api_key}")


    try:
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        data = response.json()

        if 'results' in data and data['results']:
            # Extract relevant information from the API response
            game_data = data['results'][0]
            year_published = game_data.get('released', 'N/A')
            developer = game_data.get('developers', ['N/A'])[0]
            price = game_data.get('price', 'N/A')
            stores = game_data.get('stores', [])
            # copies_sold = game_data.get('ratings', {}).get('addicted', 'N/A')
            rating = game_data.get('rating', 'N/A')
            picture_url = game_data.get('background_image', 'N/A')

            # Format the information into a dictionary
            game_info = {
                'title': game_data.get('name', 'N/A'),
                'year_published': year_published,
                'developer': developer,
                'price': price,
                'stores': stores,
                # 'copies_sold': copies_sold,
                'rating': rating,
                'picture_url': picture_url,
                'game_data': game_data,
                "genres": game_data.get('genres', [{"name":'N/A'}]),
            }

            return game_info
        else:
            return None  # No results found

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None
