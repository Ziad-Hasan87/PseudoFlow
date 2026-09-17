"""
build_notebook.py
Generates the complete PseudoFlow_Translator.ipynb notebook matching all user specifications.
"""

import json

cells = []

def add_cell(markdown_line, code_text):
    # Markdown cell with single line text
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [markdown_line + "\n"]
    })
    # Code cell with NO comments
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code_text.strip().splitlines()]
    })


# ------------------------------------------------------------------------------
# Cell 1: Libraries & Device
# ------------------------------------------------------------------------------
add_cell(
    "### 1. Import required libraries and configure compute device",
    """import os
import re
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Compute device: {device}")"""
)

# ------------------------------------------------------------------------------
# Cell 2: Load Raw Corpus
# ------------------------------------------------------------------------------
add_cell(
    "### 2. Load the raw pseudocode and mermaid corpus from disk",
    """with open('Raw_Corpus.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

corpus = raw_data['corpus']
print(f"Total raw samples loaded: {len(corpus)}")"""
)

# ------------------------------------------------------------------------------
# Cell 3: Clean & Normalize
# ------------------------------------------------------------------------------
add_cell(
    "### 3. Preprocess and normalize pseudocode and mermaid text representations and save to Clean_Corpus.json",
    """def normalize_text(text):
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\\n".join(lines)

clean_corpus = []
for item in corpus:
    clean_corpus.append({
        "pseudocode": normalize_text(item["pseudocode"]),
        "mermaid": normalize_text(item["mermaid"])
    })

with open('Clean_Corpus.json', 'w', encoding='utf-8') as f:
    json.dump({"corpus": clean_corpus}, f, indent=2)

print(f"Cleaned {len(clean_corpus)} samples and saved to Clean_Corpus.json")"""
)

# ------------------------------------------------------------------------------
# Cell 4: Tokenize Corpus
# ------------------------------------------------------------------------------
add_cell(
    "### 4. Tokenize pseudocode and mermaid programs into lexical units and save to Token_Corpus.json",
    """def tokenize_pseudocode(code_text):
    pattern = re.compile(r'(?:END\\s+IF|END\\s+FOR)|[A-Za-z_][A-Za-z0-9_]*|\\d+|==|!=|<=|>=|[+\\-*/%=<>()]')
    tokens = []
    for line in code_text.splitlines():
        line = line.strip()
        if line:
            tokens.extend(pattern.findall(line))
            tokens.append("<NL>")
    if tokens and tokens[-1] == "<NL>":
        tokens.pop()
    return tokens

def tokenize_mermaid(mermaid_text):
    pattern = re.compile(r'flowchart|TD|-->|--\\s+[A-Za-z0-9]+\\s+-->|\\[/|/\\]|\\(\\[|\\]\\)|\\{|\\}|\\[|\\]|\\"|==|!=|<=|>=|[+\\-*/%=<>]|[A-Za-z_][A-Za-z0-9_]*|\\d+')
    tokens = []
    for line in mermaid_text.splitlines():
        line = line.strip()
        if line:
            tokens.extend(pattern.findall(line))
            tokens.append("<NL>")
    if tokens and tokens[-1] == "<NL>":
        tokens.pop()
    return tokens

token_corpus = []
for item in clean_corpus:
    token_corpus.append({
        "src_tokens": tokenize_pseudocode(item["pseudocode"]),
        "trg_tokens": tokenize_mermaid(item["mermaid"])
    })

with open('Token_Corpus.json', 'w', encoding='utf-8') as f:
    json.dump({"corpus": token_corpus}, f, indent=2)

print(f"Tokenized {len(token_corpus)} samples and saved to Token_Corpus.json")
print("Sample source tokens:", token_corpus[0]["src_tokens"][:12])
print("Sample target tokens:", token_corpus[0]["trg_tokens"][:12])"""
)

# ------------------------------------------------------------------------------
# Cell 5: Vocabularies
# ------------------------------------------------------------------------------
add_cell(
    "### 5. Build source and target vocabulary index mappings and save to vocabularies.json",
    """PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

src_vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
trg_vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}

for item in token_corpus:
    for tok in item["src_tokens"]:
        if tok not in src_vocab:
            src_vocab[tok] = len(src_vocab)
    for tok in item["trg_tokens"]:
        if tok not in trg_vocab:
            trg_vocab[tok] = len(trg_vocab)

inv_src_vocab = {v: k for k, v in src_vocab.items()}
inv_trg_vocab = {v: k for k, v in trg_vocab.items()}

vocab_data = {
    "src_vocab": src_vocab,
    "trg_vocab": trg_vocab
}

with open('vocabularies.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_data, f, indent=2)

print(f"Source Vocab Size: {len(src_vocab)}")
print(f"Target Vocab Size: {len(trg_vocab)}")
print("Vocabularies saved to vocabularies.json")"""
)

# ------------------------------------------------------------------------------
# Cell 6: DataLoader
# ------------------------------------------------------------------------------
add_cell(
    "### 6. Encode, pad, and batch tokenized sequences into PyTorch DataLoaders",
    """def encode_sequence(tokens, vocab, add_sos_eos=True):
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]
    if add_sos_eos:
        return [vocab[SOS_TOKEN]] + ids + [vocab[EOS_TOKEN]]
    return ids

class PseudoFlowDataset(Dataset):
    def __init__(self, token_data, src_v, trg_v):
        self.samples = []
        for item in token_data:
            s_ids = encode_sequence(item["src_tokens"], src_v, add_sos_eos=True)
            t_ids = encode_sequence(item["trg_tokens"], trg_v, add_sos_eos=True)
            self.samples.append((s_ids, t_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    trg_lens = [len(t) for t in trg_batch]
    max_src_len = max(src_lens)
    max_trg_len = max(trg_lens)

    padded_src = [s + [src_vocab[PAD_TOKEN]] * (max_src_len - len(s)) for s in src_batch]
    padded_trg = [t + [trg_vocab[PAD_TOKEN]] * (max_trg_len - len(t)) for t in trg_batch]

    return torch.tensor(padded_src, dtype=torch.long), torch.tensor(padded_trg, dtype=torch.long)

full_dataset = PseudoFlowDataset(token_corpus, src_vocab, trg_vocab)
train_size = int(0.85 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Train batches: {len(train_loader)} ({train_size} samples)")
print(f"Validation batches: {len(val_loader)} ({val_size} samples)")"""
)

# ------------------------------------------------------------------------------
# Cell 7: Seq2Seq Architecture
# ------------------------------------------------------------------------------
add_cell(
    "### 7. Define the Encoder-Decoder Seq2Seq neural network architecture",
    """class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell

class Seq2SeqTranslation(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        decoder_input = trg[:, 0].unsqueeze(1)
        for t in range(1, trg_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = prediction.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(-1)
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1
        return outputs"""
)

# ------------------------------------------------------------------------------
# Cell 8: Model Setup
# ------------------------------------------------------------------------------
add_cell(
    "### 8. Instantiate model hyperparameters, loss criterion, and Adam optimizer",
    """EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.003

encoder = Seq2SeqEncoder(len(src_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Seq2SeqDecoder(len(trg_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = Seq2SeqTranslation(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab[PAD_TOKEN])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(model)"""
)

# ------------------------------------------------------------------------------
# Cell 9: Training & Checkpoint
# ------------------------------------------------------------------------------
add_cell(
    "### 9. Train the Seq2Seq translation model and save the trained weights locally",
    """MODEL_PATH = "pseudoflow_model.pt"
NUM_EPOCHS = 15

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded existing trained model from {MODEL_PATH}")
else:
    print(f"Training Seq2Seq model for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for src_batch, trg_batch in train_loader:
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)

            optimizer.zero_grad()
            output = model(src_batch, trg_batch, teacher_forcing_ratio=0.5)

            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), trg_batch[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Loss: {avg_loss:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab,
        "config": {
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT
        }
    }, MODEL_PATH)
    print(f"Model saved locally to {MODEL_PATH}")"""
)

# ------------------------------------------------------------------------------
# Cell 10: Validation
# ------------------------------------------------------------------------------
add_cell(
    "### 10. Evaluate the trained translation model on the validation dataset",
    """model.eval()
val_loss = 0.0
correct_tokens = 0
total_tokens = 0

with torch.no_grad():
    for src_batch, trg_batch in val_loader:
        src_batch = src_batch.to(device)
        trg_batch = trg_batch.to(device)

        output = model(src_batch, trg_batch, teacher_forcing_ratio=0.0)
        output_dim = output.shape[-1]
        loss = criterion(output[:, 1:].reshape(-1, output_dim), trg_batch[:, 1:].reshape(-1))
        val_loss += loss.item()

        preds = output[:, 1:].argmax(-1)
        targets = trg_batch[:, 1:]
        mask = targets != trg_vocab[PAD_TOKEN]
        correct_tokens += ((preds == targets) & mask).sum().item()
        total_tokens += mask.sum().item()

avg_val_loss = val_loss / len(val_loader)
accuracy = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0
print(f"Validation Loss: {avg_val_loss:.4f}")
print(f"Token-level Accuracy: {accuracy:.2f}%")"""
)

# ------------------------------------------------------------------------------
# Cell 11: Interactive Test Cell
# ------------------------------------------------------------------------------
add_cell(
    "### 11. Interactive inference cell to translate custom pseudocode into Mermaid flowcharts",
    """def format_mermaid_tokens(tokens):
    lines = []
    curr = []
    def clean_line(s):
        s = re.sub(r'(node\d+)\s+(\(\[|\[/|\[|\{)', r'\\1\\2', s)
        s = re.sub(r'(\(\[|\[/|\[|\{)\s*"', r'\\1"', s)
        s = re.sub(r'"\s*(\]\)|/\]|\]|\})', r'"\\1', s)
        s = re.sub(r'"\s+(.*?)\s+"', r'"\\1"', s)
        return s.strip()

    for tok in tokens:
        if tok == "<NL>":
            if curr:
                line_str = clean_line(" ".join(curr))
                if line_str == "flowchart TD" or "-->" in line_str or re.search(r'node\d+[\[\(\{]', line_str):
                    lines.append(f"    {line_str}" if line_str != "flowchart TD" else line_str)
                curr = []
        else:
            curr.append(tok)
    if curr:
        line_str = clean_line(" ".join(curr))
        if "-->" in line_str or re.search(r'node\d+[\[\(\{]', line_str):
            lines.append(f"    {line_str}")
    return "\\n".join(lines)

def translate_pseudocode(pseudocode_text, max_len=150):
    model.eval()
    tokens = tokenize_pseudocode(pseudocode_text)
    src_indices = [src_vocab.get(t, src_vocab[UNK_TOKEN]) for t in tokens]
    src_tensor = torch.tensor([[src_vocab[SOS_TOKEN]] + src_indices + [src_vocab[EOS_TOKEN]]], dtype=torch.long).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
        decoder_input = torch.tensor([[trg_vocab[SOS_TOKEN]]], dtype=torch.long).to(device)

        translated_tokens = []
        for _ in range(max_len):
            prediction, hidden, cell = model.decoder(decoder_input, hidden, cell)
            top1 = prediction.argmax(-1).item()

            if top1 == trg_vocab[EOS_TOKEN]:
                break

            tok_str = inv_trg_vocab.get(top1, UNK_TOKEN)
            translated_tokens.append(tok_str)
            decoder_input = torch.tensor([[top1]], dtype=torch.long).to(device)

    return format_mermaid_tokens(translated_tokens)

sample_test_pseudocode = \"\"\"FOR i = 1 TO n
    IF i % 2 == 0
        FOR j = 1 TO i
            OUTPUT j
        END FOR
    ELSE
        OUTPUT 0
    END IF
END FOR\"\"\"

generated_mermaid = translate_pseudocode(sample_test_pseudocode)
print("=== Input Pseudocode ===")
print(sample_test_pseudocode)
print("\\n=== Generated Mermaid Script ===")
print(generated_mermaid)"""
)

# ------------------------------------------------------------------------------
# Cell 12: Transformer Architecture & Training
# ------------------------------------------------------------------------------
add_cell(
    "### 12. Define and train the Transformer Seq2Seq translation model using CUDA if available",
    """class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.trg_tok_emb = nn.Embedding(trg_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, src_mask=None, trg_mask=None, src_padding_mask=None, trg_padding_mask=None):
        src_emb = self.pos_encoder(self.src_tok_emb(src) * (self.d_model ** 0.5))
        trg_emb = self.pos_encoder(self.trg_tok_emb(trg) * (self.d_model ** 0.5))
        outs = self.transformer(
            src_emb, trg_emb,
            src_mask=src_mask,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=trg_padding_mask
        )
        return self.fc_out(outs)

    def encode(self, src):
        src_emb = self.pos_encoder(self.src_tok_emb(src) * (self.d_model ** 0.5))
        return self.transformer.encoder(src_emb)

    def decode(self, trg, memory, trg_mask=None):
        trg_emb = self.pos_encoder(self.trg_tok_emb(trg) * (self.d_model ** 0.5))
        return self.transformer.decoder(trg_emb, memory, tgt_mask=trg_mask)

TRANSFORMER_PATH = "pseudoflow_transformer.pt"
transformer_model = TransformerSeq2Seq(len(src_vocab), len(trg_vocab)).to(device)
criterion_tf = nn.CrossEntropyLoss(ignore_index=trg_vocab[PAD_TOKEN])
optimizer_tf = optim.Adam(transformer_model.parameters(), lr=0.001)

if os.path.exists(TRANSFORMER_PATH):
    checkpoint_tf = torch.load(TRANSFORMER_PATH, map_location=device)
    transformer_model.load_state_dict(checkpoint_tf["model_state_dict"])
    print(f"Loaded existing Transformer model from {TRANSFORMER_PATH}")
else:
    print(f"Training Transformer on device: {device}")
    MAX_EPOCHS = 20
    for epoch in range(MAX_EPOCHS):
        transformer_model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for src_batch, trg_batch in train_loader:
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)

            trg_input = trg_batch[:, :-1]
            trg_expected = trg_batch[:, 1:]

            seq_len = trg_input.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            src_pad_mask = (src_batch == src_vocab[PAD_TOKEN]).to(device)
            trg_pad_mask = (trg_input == trg_vocab[PAD_TOKEN]).to(device)

            optimizer_tf.zero_grad()
            output = transformer_model(
                src_batch, trg_input,
                trg_mask=tgt_mask,
                src_padding_mask=src_pad_mask,
                trg_padding_mask=trg_pad_mask
            )

            loss = criterion_tf(output.reshape(-1, len(trg_vocab)), trg_expected.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)
            optimizer_tf.step()

            epoch_loss += loss.item()
            preds = output.argmax(dim=-1)
            mask = trg_expected != trg_vocab[PAD_TOKEN]
            correct += ((preds == trg_expected) & mask).sum().item()
            total += mask.sum().item()

        avg_loss = epoch_loss / len(train_loader)
        acc = (correct / total) if total > 0 else 0.0
        if (epoch + 1) % 5 == 0 or epoch == 0 or avg_loss <= 0.10:
            print(f"Epoch {epoch + 1}/{MAX_EPOCHS} - Loss: {avg_loss:.4f} - Token Accuracy: {acc * 100:.2f}%")

        if avg_loss <= 0.10:
            print(f"Target loss <= 0.10 reached at epoch {epoch + 1}!")
            break

    torch.save({
        "model_state_dict": transformer_model.state_dict(),
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab
    }, TRANSFORMER_PATH)
    print(f"Transformer model saved to {TRANSFORMER_PATH}")"""
)

# ------------------------------------------------------------------------------
# Cell 13: Transformer Inference Testing
# ------------------------------------------------------------------------------
add_cell(
    "### 13. Interactive inference cell to translate pseudocode using the trained Transformer model",
    """def translate_pseudocode_transformer(pseudocode_text, max_len=150):
    transformer_model.eval()
    tokens = tokenize_pseudocode(pseudocode_text)
    src_indices = [src_vocab.get(t, src_vocab[UNK_TOKEN]) for t in tokens]
    src_tensor = torch.tensor([[src_vocab[SOS_TOKEN]] + src_indices + [src_vocab[EOS_TOKEN]]], dtype=torch.long).to(device)

    with torch.no_grad():
        memory = transformer_model.encode(src_tensor)
        ys = torch.tensor([[trg_vocab[SOS_TOKEN]]], dtype=torch.long).to(device)

        for _ in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
            out = transformer_model.decode(ys, memory, trg_mask=tgt_mask)
            prob = transformer_model.fc_out(out[:, -1])
            next_word = prob.argmax(dim=-1).item()

            if next_word == trg_vocab[EOS_TOKEN]:
                break

            ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long).to(device)], dim=1)

    translated_tokens = [inv_trg_vocab.get(idx, UNK_TOKEN) for idx in ys.squeeze(0).tolist()[1:]]
    return format_mermaid_tokens(translated_tokens)

test_code_input = \"\"\"FOR i = 1 TO n
    IF i % 2 == 0
        FOR j = 1 TO i
            OUTPUT j
        END FOR
    ELSE
        OUTPUT 0
    END IF
END FOR\"\"\"

generated_mermaid_tf = translate_pseudocode_transformer(test_code_input)
print("=== Input Pseudocode ===")
print(test_code_input)
print("\\n=== Transformer Generated Mermaid Script ===")
print(generated_mermaid_tf)"""
)

notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (ziad)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.7"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("PseudoFlow_Translator.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print(f"Successfully created PseudoFlow_Translator.ipynb with {len(cells)} cells ({len(cells)//2} code stages).")
