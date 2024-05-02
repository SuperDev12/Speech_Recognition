import os
import whisper
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments
from transformers import Trainer
from transformers import default_data_collator

# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, audio_paths, transcripts):
        self.audio_paths = audio_paths
        self.transcripts = transcripts

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Load audio and transcript for the given index
        audio_path = self.audio_paths[idx]
        transcript = self.transcripts[idx]

        # Preprocess audio (e.g., load, pad/trim, convert to spectrogram)
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)

        # Convert transcript to token IDs
        tokens = whisper.encode(transcript)

        return {"input": mel, "labels": tokens}

# Example paths to your audio files and corresponding transcripts
audio_paths = ["/Users/superdev/Desktop/Speech_Recognition/Speech_Recognition/hindi/test/audio/153/844424930627593-153-f.wav", "/Users/superdev/Desktop/Speech_Recognition/Speech_Recognition/hindi/test/audio/153/844424930627596-153-f.wav"]
transcripts = ["समीक्षा इस दवा के बारे में स्त्री रोग विशेषज्ञ विश्वास प्रेरित करते हैं", "रिया चक्रवर्ती को आज भायखला जेल में चटाई बिछा कर रात गुजर नहीं पड़ेगी"]

# Create an instance of your custom dataset
dataset = CustomDataset(audio_paths, transcripts)

# Load the Whisper model
model = whisper.load_model("base")

# **Avoid SparseMPS backend (if applicable):**
# Set the device explicitly to 'cpu' or 'cuda' if available
device = torch.device("cpu")  # Or "cuda" for GPU training
model = model.to(device)

# Define the Trainer arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=whisper.tokenizer,
    data_collator=default_data_collator,
)

# Start training
trainer.train()