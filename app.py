from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func, inspect

from flask import Flask, jsonify, render_template, url_for, request #urlfor ask flask to find certain files and translate to website
from flask_cors import cross_origin
import json
import sqlite3 as sq
import os
# import geopandas as gpd
import pandas as pd
import pickle
from prediction_engine import load_computed_data,get_similar_game_names,get_game_info

#################################################
# Database Setup
#################################################

# make a flask app, 
# configure a templates engine and directory
# make a 3 demo endpoints

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# a QLALCHEMY_TRACK_MODIFICATIONS'] = False

# engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])

# configue public assets directory for the app

# app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'static')

#  configure the templates directory to be  in templates folder
app.config['TEMPLATES_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')



# make routes
# / index route: return index.html 
@app.route("/")
def index():
    query_title = request.args.get('game_title',None)
    print(f"app.home query_title: {query_title}")
    
    if query_title == None:
        return render_template("index.html",game_suggestions=None,alternative_game_names=None)

    # get the suggested titles 
    suggestions_or_alternatives = None if query_title is None else get_similar_game_names(input_game=query_title,saved_computed_model_data=app.config.get('computed_data'))
    # print(f"app.home game_suggestions: {count(suggested_titles)}")
    
    print(f"app.home suggestions_or_alternatives: {suggestions_or_alternatives}")
    suggested_titles ,alternative_game_names = suggestions_or_alternatives
    
    game_suggestions = None

    # get the game details
    if suggested_titles is not None:
        game_suggestions = [get_game_info(game_title=game_title) for game_title in suggested_titles]
     
       
    return render_template("index.html",game_suggestions = game_suggestions,alternative_game_names=alternative_game_names)


@app.route("/query")
def query():
    return render_template("query.html")

# /api/v1.0/precipitation route: return json of precipitation data


if __name__ == '__main__':
    # Load precomputed data from binary file
    # Store the loaded data in the Flask application context
    app.config['computed_data'] = load_computed_data()
    print("Loaded saved model")

    # if not computed_data:
    #     # If the data is not available or outdated, compute and save it
    #     compute_dependencies_and_save()
    #     computed_data = load_computed_data()


    # Run the Flask application
    app.run(debug=True)