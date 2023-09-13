import torch
import torch.nn as nn


class ToyGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers):
        super(ToyGPT, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.transformer = nn.Transformer(
            embed_size,
            num_heads,
            num_layers,
            num_layers,
            dim_feedforward=embed_size * 4,
        )

        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = self.fc(x)

        return x


print("doing the byte pair encoding")
from bpe import encoded_corpus, vocab_to_id

print("done with the byte pair encoding")

vocab_size = len(vocab_to_id)  # Size of your vocabulary
embed_size = 64  # Embedding dimension
num_heads = 2  # Number of attention heads
num_layers = 2  # Number of layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ToyGPT(vocab_size, embed_size, num_heads, num_layers).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

import torch
from torch.utils.data import DataLoader, TensorDataset


# Function to split the corpus into sequences of a fixed length
def split_into_sequences(token_ids, seq_length):
    sequences = []
    for i in range(0, len(token_ids) - seq_length, seq_length):
        sequences.append(token_ids[i : i + seq_length])
    return sequences


# Function to create data and labels
def create_data_and_labels(sequences):
    data = []
    labels = []
    for seq in sequences:
        data.append(seq[:-1])
        labels.append(seq[1:])
    return data, labels


print("preparing the dataloader")
# Convert encoded_corpus into sequences of length 100 (for example)
seq_length = 100
sequences = split_into_sequences(encoded_corpus, seq_length)

# Create data and labels
data, labels = create_data_and_labels(sequences)

# Convert data and labels into PyTorch tensors
data = torch.tensor(data, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.long)

# Create a TensorDataset from data and labels
dataset = TensorDataset(data, labels)

# Create a DataLoader from the TensorDataset
batch_size = 512  # You can adjust the batch size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
print("starting the epochs")
best_loss = float("inf")

for epoch in range(10_000):  # Let's say 100 epochs for example
    total_loss = 0
    num_batches = 0
    for data_batch, label_batch in data_loader:
        data_batch, label_batch = data_batch.to(device), label_batch.to(device)

        optimizer.zero_grad()
        output = model(data_batch)
        loss = loss_fn(output.view(-1, vocab_size), label_batch.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

    # Save the model if it improves
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(model.state_dict(), "best_model_state_dict.pth")
