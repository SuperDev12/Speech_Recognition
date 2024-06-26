import whisper
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load the model
model = whisper.load_model("base")

# Load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("/Users/superdev/Desktop/Speech_Recognition/Speech_Recognition/test_dataset/01-00004-02.mp3")
audio = whisper.pad_or_trim(audio)

# Make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# Detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# Decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# Print the recognized text
print(result.text)