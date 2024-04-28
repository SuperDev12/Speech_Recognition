import os
import torchaudio
import speech_recognition as sr

# Directory containing audio files
audio_dir = "/Users/superdev/Desktop/Speech_Recognition/Speech_Recognition/hindi/test_known/audio/3"
# List of audio file paths
audio_files = os.listdir(audio_dir)

# Initialize the recognizer
recognizer = sr.Recognizer()

# Iterate over audio files
for audio_file in audio_files:
    # Load audio file
    audio_path = os.path.join(audio_dir, audio_file)
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    # Transcribe audio
    try:
        transcription = recognizer.recognize_google(audio_data)
        print(f"Transcription for {audio_file}: {transcription}")
        # You can save the transcription to a file or process it further
    except sr.UnknownValueError:
        print(f"Google Speech Recognition could not understand audio in {audio_file}")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
