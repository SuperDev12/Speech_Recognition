import torchaudio
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Trainer, TrainingArguments
import jiwer

# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, audio_path, text):
        self.audio_path = audio_path
        self.text = text
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

    def __len__(self):
        return len(self.audio_path)

    def __getitem__(self, idx):
        # Load and preprocess your audio and text data
        audio, _ = torchaudio.load(self.audio_path[idx])
        text = self.text[idx]
        inputs = self.processor(audio.squeeze(0), return_tensors="pt", padding=True, truncation=True)
        labels = self.processor(text, return_tensors="pt").input_ids
        return {"input_values": inputs.input_values, "labels": labels}

# Replace with the actual path to your audio file
audio_path = "Speech_Recognition/hindi/test/audio/153/844424930627593-153-f.wav"
text = "Your ground truth text"

# Create your custom dataset instance
custom_dataset = CustomDataset(audio_path, text)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    save_total_limit=2,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_strategy="epoch",
)

# Define the compute_wer function
def compute_wer(pred_label_ids, predictions):
    predicted_texts = custom_dataset.processor.batch_decode(predictions, skip_special_tokens=True)
    true_texts = custom_dataset.processor.batch_decode(pred_label_ids, skip_special_tokens=True)
    wer_score = jiwer.wer(true_texts, predicted_texts)
    return {"wer": wer_score}

# Create your Wav2Vec2ForCTC model instance
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Create your Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=custom_dataset,
    eval_dataset=custom_dataset,
    compute_metrics=compute_wer,
)

# Train the model
trainer.train()