import os
import torchaudio
import speech_recognition as sr

# Directory containing audio files
audio_dir = "/Users/superdev/Desktop/Speech_Recognition/Speech_Recognition/hindi/test/audio/153"

# Output file for saving transcriptions
output_file = "/Users/superdev/Desktop/Speech_Recognition/Speech_Recognition/hindi/test/test_transcription"

# Initialize the recognizer
recognizer = sr.Recognizer()

# Open the output file in write mode
with open(output_file, "w", encoding="utf-8") as f_out:
    # List of audio file paths
    audio_files = os.listdir(audio_dir)

    # Iterate over audio files
    for audio_file in audio_files:
        # Load audio file
        audio_path = os.path.join(audio_dir, audio_file)
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)

        # Transcribe audio
        try:
            transcription = recognizer.recognize_google(audio_data, language="hi-IN")  # Specify language for Hindi
            print(f"Transcription for {audio_file}: {transcription}")

            # Write transcription to the output file
            f_out.write(f"Audio file: {audio_file}\n")
            f_out.write(f"Transcription: {transcription}\n\n")
        except sr.UnknownValueError:
            print(f"Google Speech Recognition could not understand audio in {audio_file}")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

# Print message indicating the process is complete
print(f"Transcriptions saved to {output_file}")
