import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import statistics
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import seaborn as sns; sns.set(style='white')
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')


def auth(cid, secret, username, redirect_uri):
    scope = 'user-library-read playlist-modify-public playlist-read-private user-top-read user-read-recently-played'
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    token = util.prompt_for_user_token(username,scope,client_id=cid,client_secret=secret,
                                       redirect_uri=redirect_uri)
    if token:
        sp = spotipy.Spotify(auth=token)
    else:
        print("Can't get token for", username)
    token = util.prompt_for_user_token(username, scope, cid, secret, redirect_uri)
    return sp



def track_info(sp, songs):
    track_ids = []
    track_names = []
    for i in range(0, len(songs)):
        for j in range(0, len(songs[i])):
            if songs[i][j]['track']['id'] != None:
                track_ids.append(songs[i][j]['track']['id'])
                track_names.append(songs[i][j]['track']['name'])
    features = []
    for i in range(0,len(track_ids)):
        audio_features = sp.audio_features(track_ids[i])
        for track in audio_features:
          if track is None:
            print(track)
            features.append({'danceability': 0, 'energy': 0, 'key': 0, 'loudness': 0, 'mode': 0, 'speechiness': 0, 'acousticness': 0, 'instrumentalness': 0, 'liveness': 0, 'valence': 0, 'tempo': 0, 'type': 'audio_features', 'id': '00000', 'uri': 'spotify:track:0', 'track_href': 'https://api.spotify.com/', 'analysis_url': 'https://api.spotify.com/', 'duration_ms': 0, 'time_signature': 0})
          else:
            features.append(track)
    return features, track_names



def listening_history(track_names, folder, file, months):
    count_tracks = dict.fromkeys(track_names, 0)
    with open(f'{folder}/{file}.json', encoding='utf-8') as f:
        data = json.load(f)
        six_months_ago = (str(datetime.now() - timedelta(days=30*months))).split('.')[0]
        reversed_data = data[::-1]
        for diz in reversed_data:
            endTime = diz['endTime']
            trackName = diz['trackName']
            msPlayed = diz['msPlayed']
            if endTime >= six_months_ago and trackName in track_names and msPlayed >= 10000:
                count_tracks[trackName] += 1
    return count_tracks



def ratings_(ratings, sorted_ratings):
    min_value = min(ratings)
    max_value = max(ratings)
    scale_range = 10 - 1
    normalized_data = []
    for value in ratings:
        if (value - min_value) == 0 or (max_value - min_value) == 0:
            normalized_value = 1
        else:
             normalized_value = ((value - min_value) / (max_value - min_value)) * scale_range + 1
        normalized_data.append(round(normalized_value, 2))
    print("\n")
    print(sorted(normalized_data, reverse=True))
    print("\n")
    
    unique_ratings = list(set(ratings))
    mean = statistics.mean(unique_ratings)
    mean_rounded_up = math.ceil(mean)
    if mean_rounded_up in ratings:
        mean_index = sorted_ratings.index(mean_rounded_up)
    else:
        while True:
            if mean_rounded_up+1 in ratings:
                mean_index = sorted_ratings.index(mean_rounded_up+1)
                break
    first = sorted_ratings[:mean_index]
    second = sorted_ratings[mean_index:]
    
    votes = {}
    
    max_value = max(first)
    min_value = min(first)
    scale_range = 10 - 8 
    normalized_data = []
    for value in first:
        if (value - min_value) == 0 or (max_value - min_value) == 0:
            normalized_value = 8
        else:
            normalized_value = ((value - min_value) / (max_value - min_value)) * scale_range + 8
        normalized_data.append(round(normalized_value, 2))
    a = sorted(normalized_data, reverse=True)
    
    unique_ratings = list(set(first))
    mean = statistics.mean(unique_ratings)
    mean_rounded_up = math.ceil(mean)
    if mean_rounded_up in first:
        mean_index = first.index(mean_rounded_up)
    else:
        while True:
            mean_rounded_up += 1
            if mean_rounded_up in first:
                mean_index = first.index(mean_rounded_up)
                break
    ten = first[:mean_index]
    aa = ten
    
    new_list = [10 for el in aa]
    for value in a[len(aa):]:
        if value >= 0.5:
            rounded_value = round(value)
        else:
            rounded_value = round(value - 0.5)
        new_list.append(rounded_value)
    new_votes = {key: value for key, value in zip(first, new_list)}
    votes.update(new_votes)
    
    max_value = max(second)
    min_value = min(second)
    normalized_data = []
    for value in second:
        if (value - min_value) == 0 or (max_value - min_value) == 0:
            normalized_value = 0
        else:
            normalized_value = ((value - min_value) / (max_value - min_value)) * 8
        normalized_data.append(round(normalized_value, 2))
    b = sorted(normalized_data, reverse=True)
    unique_ratings = list(set(second))
    mean = statistics.mean(unique_ratings)
    mean_rounded_up = math.ceil(mean)
    if mean_rounded_up in second:
        mean_index = second.index(mean_rounded_up)
    else:
        while True:
            mean_rounded_up += 1
            if mean_rounded_up in second:
                mean_index = second.index(mean_rounded_up)
                break
    eight = second[:mean_index]
    second2 = second[mean_index:]
    unique_ratings = list(set(second2))
    mean = statistics.mean(unique_ratings)
    mean_rounded_up = math.ceil(mean)
    if mean_rounded_up in second2:
        mean_index = second2.index(mean_rounded_up)
    else:
        while True:
            mean_rounded_up += 1
            if mean_rounded_up in second2:
                mean_index = second2.index(mean_rounded_up)
                break
    zero = second2[mean_index:]
    cc = zero
    
    aa = eight
    
    new_list = []
    new_list = [8 for el in aa]
    for value in b[len(aa):len(b)-len(cc)]:
        if value >= 0.5:
            rounded_value = round(value)
        else:
            rounded_value = round(value - 0.5)
        new_list.append(rounded_value)
    for value in zero: 
        new_list.append(0)
    new_votes = {key: value for key, value in zip(second, new_list)}
    votes.update(new_votes)
    print(votes)
    return votes
    
    
    
def feature_ranking(playlist_df):
    X_train = playlist_df.drop(['id', 'ratings'], axis=1)
    y_train = playlist_df['ratings']
    forest = RandomForestClassifier(random_state=42, max_depth=5, max_features=12) 
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(len(importances)):
        print("%d. %s %f " % (f + 1, 
                X_train.columns[f], 
                importances[indices[f]]))
    return X_train, y_train
    
    
    
def PCA_analysis(X_train):
    X_scaled = StandardScaler().fit_transform(X_train)
    pca = decomposition.PCA().fit(X_scaled)
    plt.figure(figsize=(10,7))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
    plt.xlabel('Number of components')
    plt.ylabel('Total explained variance')
    plt.xlim(0, 12)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axvline(8, c='b')
    plt.axhline(0.95, c='r')
    plt.show()
    pca1 = decomposition.PCA(n_components=8)
    X_pca = pca1.fit_transform(X_scaled)
    return X_pca, pca1



def model_evaluation(X_train_last, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tree = DecisionTreeClassifier()
    tree_params = {'max_depth': range(1,11), 'max_features': range(4,19)}
    tree_grid = GridSearchCV(tree, tree_params, cv=skf, n_jobs=-1, verbose=True)
    tree_grid.fit(X_train_last, y_train)
    print(tree_grid.best_estimator_, tree_grid.best_score_)
    
    parameters = {'max_features': [4, 7, 8, 10], 'min_samples_leaf': [1, 3, 5, 8], 
                  'max_depth': [3, 5, 8]}
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, 
                                 n_jobs=-1, oob_score=True)
    gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
    gcv.fit(X_train_last, y_train)
    print(gcv.best_estimator_, gcv.best_score_)
    
    knn_params = {'n_neighbors': range(1, 10)}
    knn = KNeighborsClassifier(n_jobs=-1)
    knn_grid = GridSearchCV(knn, knn_params, cv=skf, n_jobs=-1, verbose=True)
    knn_grid.fit(X_train_last, y_train)
    print(knn_grid.best_params_, knn_grid.best_score_)
    return tree_grid, gcv, knn_grid



def recommendations(sp, playlist_df, v, tree_grid, X_train_last, y_train, pca1):
    rec_tracks = []
    for i in playlist_df['id'].values.tolist():
        rec_tracks += sp.recommendations(seed_tracks=[i], limit=int(len(playlist_df)/3))['tracks']
    rec_track_ids = []
    rec_track_names = []
    for i in rec_tracks:
        rec_track_ids.append(i['id'])
        rec_track_names.append(i['name'])
    rec_features = []
    for i in range(0,len(rec_track_ids)):
        rec_audio_features = sp.audio_features(rec_track_ids[i])
        for track in rec_audio_features:
            rec_features.append(track)
            
    rec_playlist_df = pd.DataFrame(rec_features, index = rec_track_ids)
    X_test_names = v.transform(rec_track_names)
    rec_playlist_df=rec_playlist_df[["acousticness", "danceability", "duration_ms", 
                             "energy", "instrumentalness",  "key", "liveness",
                             "loudness", "mode", "speechiness", "tempo", "valence"]]
    
    tree_grid.best_estimator_.fit(X_train_last, y_train)
    rec_playlist_df_scaled = StandardScaler().fit_transform(rec_playlist_df)
    rec_playlist_df_pca = pca1.transform(rec_playlist_df_scaled)
    X_test_last = csr_matrix(hstack([rec_playlist_df_pca, X_test_names]))
    y_pred_class = tree_grid.best_estimator_.predict(X_test_last)
    
    rec_playlist_df['ratings']=y_pred_class
    rec_playlist_df = rec_playlist_df.sort_values('ratings', ascending = False)
    rec_playlist_df = rec_playlist_df.reset_index()
    recs_to_add = rec_playlist_df[rec_playlist_df['ratings']>=8]['index'].values.tolist()
    return recs_to_add



def new_playlist(sp, username, sourcePlaylist, recs_to_add):
    playlist_recs = sp.user_playlist_create(username, 
                                            name='Recommended Songs for Playlist by AI - {}'.format(sourcePlaylist['name']))
    for i in recs_to_add:
        sp.user_playlist_add_tracks(username, playlist_recs['id'], [i])



def main(cid, secret, username, redirect_uri, sourcePlaylistID, folder, file, months):
    sp = auth(cid, secret, username, redirect_uri)
    
    sourcePlaylist_simple = sp.user_playlist(username, sourcePlaylistID)
    
    limit = 100
    offset = 0
    tracks = []
    songs = []
    while True:
        sourcePlaylist = sp.user_playlist_tracks(username, sourcePlaylistID, limit=limit, offset=offset)
        tracks.append(sourcePlaylist)
        if sourcePlaylist['next']:
            offset += limit
        else:
            break
    for tr in tracks:
        songs.append(tr["items"])
        
    features, track_names = track_info(sp, songs)
          
    
    playlist_df = pd.DataFrame(features, index = track_names)
    playlist_df = playlist_df[["id", "acousticness", "danceability", "duration_ms", 
                             "energy", "instrumentalness",  "key", "liveness",
                             "loudness", "mode", "speechiness", "tempo", "valence"]]
    
    v=TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 6), max_features=10000)
    X_names_sparse = v.fit_transform(track_names)
    X_names_sparse.shape
    
    count_tracks = listening_history(track_names, folder, file, months)

    ratings = list(count_tracks.values())
    sorted_ratings = sorted(ratings, reverse=True)
    print(sorted_ratings)
    
    votes = ratings_(ratings, sorted_ratings)
    
    col = []
    for el in track_names:
        number_of_listens = count_tracks[el]
        rank = votes[number_of_listens]
        col.append(rank)

    playlist_df['ratings'] = col
    

    X_train, y_train = feature_ranking(playlist_df)
    
    X_pca, pca1 = PCA_analysis(X_train)
    
    
    X_train_last = csr_matrix(hstack([X_pca, X_names_sparse]))
    warnings.filterwarnings('ignore')


    tree_grid, gcv, knn_grid = model_evaluation(X_train_last, y_train)
    
    recs_to_add = recommendations(sp, playlist_df, v, tree_grid, X_train_last, y_train, pca1)

    new_playlist(sp, username, sourcePlaylist_simple, recs_to_add)




if __name__ == '__main__':
    #the weeknd
    main('insertcid', 'insertsecret', 
          'susannacifani', 'http://localhost:8000', 
          'https://open.spotify.com/playlist/2u11ymH4swrfdVmoELwbRb?si=549c0d6111bb4dc6', 
          'MyData', 'StreamingHistory0', 6)
    
    #disney
    # main('insertcid', 'insertsecret', 
    #       'susannacifani', 'http://localhost:8000', 
    #       'https://open.spotify.com/playlist/4PDyMIMOhIvlS2R73vconA?si=767a95f1a0b74408', 
    #       'MyData', 'StreamingHistory0', 6)
    
