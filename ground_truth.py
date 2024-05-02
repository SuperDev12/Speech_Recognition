import librosa
import difflib

# Function to load audio file
def load_audio(audio_file):
    audio, sr = librosa.load(audio_file, sr=None)  # Load audio file
    return audio, sr

# Function to compare transcriptions with ground truth
def compare_transcriptions(transcription_file, ground_truth_file):
    with open(transcription_file, 'r') as file:
        transcription = file.read()  # Read transcription from file
    
    with open(ground_truth_file, 'r') as file:
        ground_truth = file.read()  # Read ground truth from file
    
    # Use difflib to find differences between transcriptions
    diff = difflib.ndiff(transcription.splitlines(), ground_truth.splitlines())
    
    # Print differences
    print("Differences between transcription and ground truth:")
    for line in diff:
        if line.startswith('-'):  # Lines only in transcription
            print("Transcription only:", line[2:])
        elif line.startswith('+'):  # Lines only in ground truth
            print("Ground truth only:", line[2:])

# Example usage
if __name__ == "__main__":
    # Replace with the paths to your audio file, transcription file, and ground truth file
    audio_file = '/Users/superdev/Desktop/Speech_Recognition/Speech_Recognition/hindi/test/audio/153/844424930627593-153-f.wav'
    transcription_file = '/Users/superdev/Desktop/Speech_Recognition/Speech_Recognition/hindi/test_known/transcription.txt'
    ground_truth_file = '/Users/superdev/Desktop/Speech_Recognition/Speech_Recognition/hindi/test_known/ground_truth.text'
    
    # Load audio file
    audio, sr = load_audio(audio_file)
    
    # Compare transcriptions with ground truth
    compare_transcriptions(transcription_file, ground_truth_file)
