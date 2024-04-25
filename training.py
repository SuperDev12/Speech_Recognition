import whisper

# Load the dataset
dataset = whisper.load_dataset("/Users/superdev/Desktop/Speech_Recognition/Speech_Recognition/hindi/test")

# Define training parameters
batch_size = 32
num_epochs = 10

# Create a data loader
train_loader = whisper.create_data_loader(dataset.train, batch_size=batch_size, shuffle=True)

# Define the loss function and optimizer
loss_fn = whisper.CTCLoss()
optimizer = whisper.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the trained model
whisper.save_model(model, "/path/to/save/model")
