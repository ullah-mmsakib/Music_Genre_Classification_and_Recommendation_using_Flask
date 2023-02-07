import os
from pydub import AudioSegment


mp3_file = "Closer.mp3"
wav_file = "Closer.wav"

# Load the mp3 from the current directory and save the wav to current directory
loaded_file = AudioSegment.from_mp3(mp3_file)
laoded_file.export(wav_file, format="wav")

wav_file = "Closer.wav"
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
    clipped_song.export(name[0] + "_" + str(clip_num) + "." + name[1], format="wav")
    clip_num += 1
    #print(clip_num)
    if clip_num > expected_clips:
        break
