from flask import Flask, jsonify, request, render_template, redirect
from werkzeug.utils import secure_filename
import os
import pandas as pd
import json
import math
import music_tag
import secret
import librosa
import spotipy
import statistics
import numpy as np
import keras
from itertools import chain
from pydub import AudioSegment
from keras.models import model_from_json
from spotipy.oauth2 import SpotifyClientCredentials

## Parent directories
MP3_PATH = "static/music/uploaded_music.mp3"
WAVFILES_PATH = "static\music\wav"
JSON_PATH = "static/infeed_data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
INFEED_DATA_PATH = "static/infeed_data.json"
MAPPING_DATA_PATH = "static/mappings.json"

## Load Model
def load_model ():
    with open("static/model/trained_model.json", "r") as tm:
        model = model_from_json(tm.read())
    model.load_weights("static/model/trained_model_weights.h5")

    return model


## Load External Music and save as .wav, then split.
def load_external_audio(mp3_path):
    mp3_file = mp3_path
    wav_file = "uploaded_music.wav"

    # Delete everything from folder path if any files exists previously
    folder_path = "static/music/wav"
    for the_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            pass

    # Load the mp3 from the current directory and save the wav to current directory
    loaded_file = AudioSegment.from_mp3(mp3_file)
    loaded_file.export(wav_file, format="wav")

    name = str(wav_file).split(".")
    loaded_file = AudioSegment.from_wav(wav_file)
    duration = int(loaded_file.duration_seconds * 1000) #In miliseconds because pydub works in miliseconds
    clip_length = 30000 # In miliseonds because we want 30 second clips
    expected_clips = int(duration/clip_length)
    clip_num = 1
    for i in range(0, duration, clip_length):
        #print(i) # for debugging
        #print(clip_length)
        clipped_song = loaded_file[i:i+clip_length]
        clipped_song.export("static/music/wav/" + name[0] + "_" + str(clip_num) + "." + name[1], format="wav")
        clip_num += 1
        #print(clip_num)
        if clip_num > expected_clips:
            break

    return expected_clips


# MFCC Generation
def generate_mfcc(wavfiles_path, json_path, expected_clips, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):

    # dictionary to store MFCCs
    data = {
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    num_model_iterations = expected_clips * num_segments

    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(wavfiles_path)):

        for f in filenames:

		# load audio file
            file_path = os.path.join(dirpath, f)
            signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

            # process all segments of audio file
            for d in range(num_segments):

                # calculate start and finish sample for current segment
                start = samples_per_segment * d
                finish = start + samples_per_segment

                # extract mfcc
                mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T

                # store only mfcc feature with expected number of vectors
                if len(mfcc) == num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    return num_model_iterations


## Genre Prediction
def predict_genre(infeed_data_path, mapping_data_path, model, num_model_iterations):

    outputs = list()

    with open(infeed_data_path, "r") as idp:
        data = json.load(idp)

    X = np.array(data["mfcc"])
    X = X.reshape(X.shape[1], X.shape[2], X.shape[0])
    X = X[np.newaxis, ...]

    for i in range(num_model_iterations):
        prediction = model.predict(X[..., i])
        predicted_index = np.argmax(prediction, axis=1)
        outputs.append(predicted_index.tolist())

    output_index = statistics.mode(list(chain.from_iterable(outputs)))

    with open(mapping_data_path, "r") as mdp:
        mp_data = json.load(mdp)
    
    predicted_genre = mp_data[output_index]

    return predicted_genre




## Genre recommendation
def recommend(genre):
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=secret.id, client_secret=secret.pswd))
    
    song = []
    song_url = {}
    seed_genre = list()
    seed_artist = list()
    seed_track = list()

    tags = music_tag.load_file(MP3_PATH)
    artist_name = tags["artist"].value if "artist" in tags else ""
    song_title = tags["title"].value if "title" in tags else ""

    if artist_name:
        search_result = sp.search(q=artist_name, limit = 1, type="artist")
        for idx, track in enumerate(search_result["artists"]["items"]):
            artist_id = search_result["artists"]["items"][idx]["id"]
            seed_artist.append(artist_id)

    if song_title:
        search_result = sp.search(q=song_title, limit = 1, type="track")
        for idx, track in enumerate(search_result["tracks"]["items"]):
            track_id = search_result["tracks"]["items"][idx]["id"]
            seed_track.append(track_id)

    if genre == "hiphop":
        seed_genre.append("hip-hop")
    else:
        seed_genre.append(genre)

    recommendation = sp.recommendations(seed_artists = seed_artist, seed_genres = seed_genre, seed_tracks = seed_track, limit=20)

    for track in recommendation["tracks"]:
        song.append({"name": track["name"],"Artist Name": track["artists"][0]["name"],  "url" : track["external_urls"]["spotify"]})

    return song  


## Flask App from below
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/music'
app.config['ALLOWED_EXTENSIONS'] = {'mp3'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def index():
    filename= ""
    result = ""
    recommendation= []
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = "uploaded_music."+ filename.split(".")[-1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        ## Model related code starts from here:           
        model = load_model()
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        num_clips = load_external_audio(MP3_PATH)
        num_iterations = generate_mfcc(WAVFILES_PATH, JSON_PATH, expected_clips=num_clips, num_segments=10)
        result = predict_genre(INFEED_DATA_PATH, MAPPING_DATA_PATH, model, num_model_iterations=num_iterations) 
        recommendation = recommend(genre = result)
    return render_template('index.html', music_file= filename, result=result, recommendation = recommendation  )


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
