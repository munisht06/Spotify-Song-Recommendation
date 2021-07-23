import streamlit as st 
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import re
import spotipy
import os
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import spotify_api as cred
from scipy.spatial.distance import cdist
from sklearn.metrics import euclidean_distances
import difflib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=cred.CLIENT_ID, client_secret=cred.CLIENT_SECRET))
df_spotify = pd.read_csv('spotify_data.csv')


# https://www.kaggle.com/artempozdniakov/spotify-data-eda-and-music-recommendation
def search_song(name, year):
    song_data = defaultdict()
    results = sp.search(q='track: {} year: {}'.format(name, year), limit=1)
    if results['tracks']['items'] == []:
        return None
    
    results = results['tracks']['items'][0]

    audio_features = sp.audio_features(results['id'])[0]
    
    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]
    
    for k,v in audio_features.items():
        song_data[k] = v
    
    return pd.DataFrame(song_data)

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = None
        try:
            song_data = spotify_data[(spotify_data['name'].str.lower() == song['name'].lower()) &
                                     (spotify_data['year'] == song['year'])].iloc[0]
        except IndexError:
            song_data = search_song(song['name'], song['year'])
        if song_data is None:
            print('Song not found')
            continue
        song_vectors.append(song_data[number_cols].values)

    return np.mean(np.array(song_vectors,dtype="object"), axis=0)

def combined_data_dict(dict_list):
    combined_data = {}
    for dictionary in dict_list:
        for key, value in dictionary.items():
            if key in combined_data.keys():
                combined_data[key].append(value)
            else:
                combined_data[key] = [value]
    return combined_data

def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = combined_data_dict(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = StandardScaler().fit(spotify_data[number_cols])
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1,-1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')


def main():
    st.title('Recommendation')

    html_temp2 = """
    <div style ="background-color:royalblue;padding:10px;border-radius:10px">
    <h2 style="color:white;text-align:center;">Spotify songsr </h2>
        <h1 style="color:white;text-align:center;">Recommendation</h1>
    </div>
    """
    components.html(html_temp2)

    components.html("""
                <img src="https://www.tech-recipes.com/wp-content/uploads/2016/02/Spotify.png" width="700" height="150">
                
                """)
    name = st.text_input("Name of the song", "Type Here")
    year = st.text_input("Year", "Type Here")

    result = ''
    result_year = ''
    result_artist = ''

    if st.button("Recommed"):
        input_data = [{'name': name, 'year': int(year)}]
        results = recommend_songs(input_data, df_spotify)
        print(results)
        for result in results:
            artists = result['artists'] 
            artists = re.sub(r'[\[\]\'\"]', '',artists)
            st.success('The recommendation song is {0} by {1} from {2}'.format(result['name'], artists, result['year']))
    

main()