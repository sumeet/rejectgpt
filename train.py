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
            dim_feedforward=embed_size * 4
        )
        
        self.fc = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = self.fc(x)
        
        return x


vocab_size = 10000  # Size of your vocabulary
embed_size = 64  # Embedding dimension
num_heads = 2  # Number of attention heads
num_layers = 2  # Number of layers

model = ToyGPT(vocab_size, embed_size, num_heads, num_layers)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# Forward pass and backward pass
for epoch in range(100):  # For simplicity, 100 epochs
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output.view(-1, vocab_size), labels.view(-1))
    loss.backward()
    optimizer.step()
