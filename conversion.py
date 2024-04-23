import os
import subprocess

def convert_mp3_to_wav(mp3_dir, wav_dir):
    # Check if the WAV directory exists
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)

    # Loop through all files in the MP3 directory
    for filename in os.listdir(mp3_dir):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(mp3_dir, filename)
            wav_path = os.path.join(wav_dir, os.path.splitext(filename)[0] + ".wav")

            # Run ffmpeg command to convert MP3 to WAV
            subprocess.run(['ffmpeg', '-i', mp3_path, '-acodec', 'pcm_s16le', '-ar', '44100', wav_path], capture_output=True, text=True)

            print(f"Converted {filename} to WAV")

# Directory containing MP3 files
mp3_directory = "/Users/superdev/Desktop/Speech_Recognition/test_dataset"

# Directory to save WAV files
wav_directory = "/Users/superdev/Desktop/Speech_Recognition/converted_testdata"

convert_mp3_to_wav(mp3_directory, wav_directory)
