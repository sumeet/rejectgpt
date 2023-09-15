import torch
import torch.nn as nn

from train import device, model
from bpe import encode_text, vocab_to_id, id_to_vocab


def prepare_input(text):
    tokens = encode_text(text)
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)


def generate_text(prompt, max_length=50):
    input = prepare_input(prompt)

    print(prompt.lower(), end=" ")
    for _ in range(max_length):
        with torch.no_grad():  # so we don't compute gradients
            output = model(input)
            output = output[:, -1:, :]  # Get the prediction for the last token only
            probabilities = nn.functional.softmax(output, dim=-1)
            next_token_id = torch.multinomial(probabilities[0, 0], 1)

        next_token_id = next_token_id.view(1, 1)
        input = torch.cat((input, next_token_id), dim=1)

        print(id_to_vocab[next_token_id.item()], end="")


print()

model.load_state_dict(torch.load("best_model_state_dict.pth"))
prompt = "buildings are"  # Replace with the start of your text

generate_text(prompt)
