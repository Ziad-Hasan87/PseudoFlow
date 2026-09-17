"""
build_notebook.py
Generates the Path A (Neuro-Symbolic Sequence Labeler + Graph Router) notebook.
"""

import json

cells = []

def add_cell(markdown_line, code_text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [markdown_line + "\n"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code_text.strip().splitlines()]
    })


# ------------------------------------------------------------------------------
# Stage 1: Libraries & Device
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
# Stage 2: Load Raw Corpus
# ------------------------------------------------------------------------------
add_cell(
    "### 2. Load the raw pseudocode corpus from disk",
    """with open('Raw_Corpus.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

raw_corpus = raw_data['corpus']
print(f"Total raw programs loaded: {len(raw_corpus)}")"""
)

# ------------------------------------------------------------------------------
# Stage 3: Normalize & Generate Tagged Corpus
# ------------------------------------------------------------------------------
add_cell(
    "### 3. Normalize pseudocode programs, assign statement roles, and save to Tagged_Corpus.json",
    """def classify_statement_rule(line):
    line_clean = line.strip()
    if line_clean == "START":
        return "START_NODE"
    elif line_clean == "END":
        return "END_NODE"
    elif line_clean == "END IF":
        return "END_IF"
    elif line_clean == "END FOR":
        return "END_FOR"
    elif line_clean == "ELSE":
        return "ELSE_BRANCH"
    elif line_clean.startswith("INPUT ") or line_clean.startswith("OUTPUT ") or line_clean == "INPUT" or line_clean == "OUTPUT":
        return "IO_NODE"
    elif line_clean.startswith("LET "):
        return "PROCESS_NODE"
    elif line_clean.startswith("IF "):
        return "IF_DECISION"
    elif line_clean.startswith("FOR "):
        return "LOOP_HEADER"
    else:
        return "PROCESS_NODE"

tagged_corpus = []
for item in raw_corpus:
    lines = [line.strip() for line in item["pseudocode"].splitlines() if line.strip()]
    tags = [classify_statement_rule(l) for l in lines]
    tagged_corpus.append({
        "lines": lines,
        "tags": tags
    })

with open('Tagged_Corpus.json', 'w', encoding='utf-8') as f:
    json.dump({"corpus": tagged_corpus}, f, indent=2)

print(f"Tagged {len(tagged_corpus)} programs and saved to Tagged_Corpus.json")
print("Sample lines:", tagged_corpus[0]["lines"][:4])
print("Sample tags:", tagged_corpus[0]["tags"][:4])"""
)

# ------------------------------------------------------------------------------
# Stage 4: Vocabularies
# ------------------------------------------------------------------------------
add_cell(
    "### 4. Build word-level and tag index mappings and save to vocabularies.json",
    """PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

word_vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
tag_to_ix = {
    PAD_TOKEN: 0,
    "START_NODE": 1,
    "END_NODE": 2,
    "IO_NODE": 3,
    "PROCESS_NODE": 4,
    "IF_DECISION": 5,
    "ELSE_BRANCH": 6,
    "LOOP_HEADER": 7,
    "END_IF": 8,
    "END_FOR": 9
}
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

pattern = re.compile(r'(?:END\\s+IF|END\\s+FOR)|[A-Za-z_][A-Za-z0-9_]*|\\d+|==|!=|<=|>=|[+\\-*/%=<>()]')
for item in tagged_corpus:
    for line in item["lines"]:
        words = pattern.findall(line)
        for w in words:
            if w not in word_vocab:
                word_vocab[w] = len(word_vocab)

inv_word_vocab = {v: k for k, v in word_vocab.items()}

vocab_data = {
    "word_vocab": word_vocab,
    "tag_to_ix": tag_to_ix
}

with open('vocabularies.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_data, f, indent=2)

print(f"Word Vocabulary Size: {len(word_vocab)}")
print(f"Total Statement Tags: {len(tag_to_ix)}")
print("Vocabularies saved to vocabularies.json")"""
)

# ------------------------------------------------------------------------------
# Stage 5: Dataset & DataLoader
# ------------------------------------------------------------------------------
add_cell(
    "### 5. Encode, pad, and batch sequence pairs into PyTorch DataLoaders",
    """def encode_line_to_words(line, max_words=10):
    words = pattern.findall(line)
    ids = [word_vocab.get(w, word_vocab[UNK_TOKEN]) for w in words][:max_words]
    ids = ids + [word_vocab[PAD_TOKEN]] * (max_words - len(ids))
    return ids

class StatementSequenceDataset(Dataset):
    def __init__(self, data):
        self.samples = []
        for item in data:
            encoded_lines = [encode_line_to_words(l) for l in item["lines"]]
            tag_ids = [tag_to_ix[t] for t in item["tags"]]
            self.samples.append((encoded_lines, tag_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_seq_fn(batch):
    lines_batch, tags_batch = zip(*batch)
    max_lines = max(len(l) for l in lines_batch)
    max_words = len(lines_batch[0][0])

    padded_lines = []
    padded_tags = []
    for lines, tags in zip(lines_batch, tags_batch):
        pad_count = max_lines - len(lines)
        padded_l = lines + [[word_vocab[PAD_TOKEN]] * max_words] * pad_count
        padded_t = tags + [tag_to_ix[PAD_TOKEN]] * pad_count
        padded_lines.append(padded_l)
        padded_tags.append(padded_t)

    return torch.tensor(padded_lines, dtype=torch.long), torch.tensor(padded_tags, dtype=torch.long)

full_dataset = StatementSequenceDataset(tagged_corpus)
train_size = int(0.85 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_seq_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_seq_fn)

print(f"Train batches: {len(train_loader)} ({train_size} samples)")
print(f"Validation batches: {len(val_loader)} ({val_size} samples)")"""
)

# ------------------------------------------------------------------------------
# Stage 6: BiRNN Architecture
# ------------------------------------------------------------------------------
add_cell(
    "### 6. Define the BiRNN sequence labeling neural network architecture",
    """class BiRNNStatementClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, line_batches):
        b, l, w = line_batches.shape
        flat_lines = line_batches.view(b * l, w)
        embeds = self.embedding(flat_lines)
        mask = (flat_lines != 0).unsqueeze(-1).float()
        line_vecs = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        line_vecs = line_vecs.view(b, l, -1)
        rnn_out, _ = self.rnn(line_vecs)
        logits = self.fc(rnn_out)
        return logits

EMBED_DIM = 64
HIDDEN_DIM = 64
tagger_model = BiRNNStatementClassifier(len(word_vocab), EMBED_DIM, HIDDEN_DIM, len(tag_to_ix)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tag_to_ix[PAD_TOKEN])
optimizer = optim.Adam(tagger_model.parameters(), lr=0.005)

print(tagger_model)"""
)

# ------------------------------------------------------------------------------
# Stage 7: Training
# ------------------------------------------------------------------------------
add_cell(
    "### 7. Train the BiRNN sequence labeler and save the trained weights locally",
    """TAGGER_PATH = "pseudoflow_tagger.pt"
NUM_EPOCHS = 10

if os.path.exists(TAGGER_PATH):
    checkpoint = torch.load(TAGGER_PATH, map_location=device)
    tagger_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded existing trained tagger from {TAGGER_PATH}")
else:
    print(f"Training BiRNN Sequence Labeler for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        tagger_model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = tagger_model(x_batch)

            loss = criterion(logits.view(-1, len(tag_to_ix)), y_batch.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Loss: {avg_loss:.4f}")

    torch.save({
        "model_state_dict": tagger_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "word_vocab": word_vocab,
        "tag_to_ix": tag_to_ix
    }, TAGGER_PATH)
    print(f"Model saved locally to {TAGGER_PATH}")"""
)

# ------------------------------------------------------------------------------
# Stage 8: Evaluation
# ------------------------------------------------------------------------------
add_cell(
    "### 8. Evaluate the trained sequence labeler on the validation dataset",
    """tagger_model.eval()
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = tagger_model(x_batch)
        loss = criterion(logits.view(-1, len(tag_to_ix)), y_batch.view(-1))
        val_loss += loss.item()

        preds = logits.argmax(dim=-1)
        mask = y_batch != tag_to_ix[PAD_TOKEN]
        correct += ((preds == y_batch) & mask).sum().item()
        total += mask.sum().item()

avg_val_loss = val_loss / len(val_loader)
acc = (correct / total) * 100 if total > 0 else 0.0
print(f"Validation Loss: {avg_val_loss:.4f}")
print(f"Tag Classification Accuracy: {acc:.2f}%")"""
)

# ------------------------------------------------------------------------------
# Stage 9: Graph Router
# ------------------------------------------------------------------------------
add_cell(
    "### 9. Define the deterministic Stack-based Graph Router to assemble Mermaid flowcharts",
    """class FlowchartAssembler:
    def __init__(self):
        self.node_count = 0
        self.nodes = []
        self.edges = []

    def _new_id(self):
        self.node_count += 1
        return f"node{self.node_count}"

    def assemble(self, lines, tags):
        self.node_count = 0
        self.nodes = []
        self.edges = []

        if_stack = []
        loop_stack = []
        pending_exits = []

        for line, tag in zip(lines, tags):
            clean_text = line.replace('"', "'")

            if tag == "START_NODE":
                nid = self._new_id()
                self.nodes.append(f'    {nid}(["{clean_text}"])')
                for src, lbl in pending_exits:
                    edge = f'    {src} -- {lbl} --> {nid}' if lbl else f'    {src} --> {nid}'
                    self.edges.append(edge)
                pending_exits = [(nid, "")]

            elif tag == "END_NODE":
                nid = self._new_id()
                self.nodes.append(f'    {nid}(["{clean_text}"])')
                for src, lbl in pending_exits:
                    edge = f'    {src} -- {lbl} --> {nid}' if lbl else f'    {src} --> {nid}'
                    self.edges.append(edge)
                pending_exits = []

            elif tag == "IO_NODE":
                nid = self._new_id()
                self.nodes.append(f'    {nid}[/"{clean_text}"/]')
                for src, lbl in pending_exits:
                    edge = f'    {src} -- {lbl} --> {nid}' if lbl else f'    {src} --> {nid}'
                    self.edges.append(edge)
                pending_exits = [(nid, "")]

            elif tag == "PROCESS_NODE":
                nid = self._new_id()
                self.nodes.append(f'    {nid}["{clean_text}"]')
                for src, lbl in pending_exits:
                    edge = f'    {src} -- {lbl} --> {nid}' if lbl else f'    {src} --> {nid}'
                    self.edges.append(edge)
                pending_exits = [(nid, "")]

            elif tag == "IF_DECISION":
                cond_text = clean_text[3:].strip() if clean_text.startswith("IF ") else clean_text
                nid = self._new_id()
                self.nodes.append(f'    {nid}{{"{cond_text}"}}')
                for src, lbl in pending_exits:
                    edge = f'    {src} -- {lbl} --> {nid}' if lbl else f'    {src} --> {nid}'
                    self.edges.append(edge)
                if_stack.append({
                    'cond_id': nid,
                    'has_else': False,
                    'then_exits': [],
                    'else_exits': []
                })
                pending_exits = [(nid, "Yes")]

            elif tag == "ELSE_BRANCH":
                if if_stack:
                    ctx = if_stack[-1]
                    ctx['has_else'] = True
                    ctx['then_exits'] = list(pending_exits)
                    pending_exits = [(ctx['cond_id'], "No")]

            elif tag == "END_IF":
                if if_stack:
                    ctx = if_stack.pop()
                    if ctx['has_else']:
                        pending_exits = ctx['then_exits'] + pending_exits
                    else:
                        pending_exits = pending_exits + [(ctx['cond_id'], "No")]

            elif tag == "LOOP_HEADER":
                nid = self._new_id()
                self.nodes.append(f'    {nid}{{"{clean_text}"}}')
                for src, lbl in pending_exits:
                    edge = f'    {src} -- {lbl} --> {nid}' if lbl else f'    {src} --> {nid}'
                    self.edges.append(edge)
                loop_stack.append({'for_id': nid})
                pending_exits = [(nid, "Yes")]

            elif tag == "END_FOR":
                if loop_stack:
                    ctx = loop_stack.pop()
                    for src, lbl in pending_exits:
                        edge = f'    {src} -- {lbl} --> {ctx["for_id"]}' if lbl else f'    {src} --> {ctx["for_id"]}'
                        self.edges.append(edge)
                    pending_exits = [(ctx['for_id'], "No")]

        if pending_exits:
            nid = self._new_id()
            self.nodes.append(f'    {nid}(["END"])')
            for src, lbl in pending_exits:
                edge = f'    {src} -- {lbl} --> {nid}' if lbl else f'    {src} --> {nid}'
                self.edges.append(edge)

        return "flowchart TD\\n" + "\\n".join(self.nodes) + "\\n\\n" + "\\n".join(self.edges)

assembler = FlowchartAssembler()"""
)

# ------------------------------------------------------------------------------
# Stage 10: Interactive Test Cell
# ------------------------------------------------------------------------------
add_cell(
    "### 10. Interactive inference cell to translate pseudocode into 100% valid Mermaid flowcharts",
    """if 'tagger_model' not in globals():
    if 'device' not in globals():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'word_vocab' not in globals():
        with open('vocabularies.json', 'r', encoding='utf-8') as f:
            vdata = json.load(f)
        word_vocab = vdata['word_vocab']
        tag_to_ix = vdata['tag_to_ix']
        ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    TAGGER_PATH = 'pseudoflow_tagger.pt'
    checkpoint = torch.load(TAGGER_PATH, map_location=device)
    tagger_model = BiRNNStatementClassifier(len(word_vocab), 64, 64, len(tag_to_ix)).to(device)
    tagger_model.load_state_dict(checkpoint['model_state_dict'])

def translate_pseudocode(pseudocode_text):
    tagger_model.eval()
    lines = [line.strip() for line in pseudocode_text.splitlines() if line.strip()]
    encoded_lines = [encode_line_to_words(l) for l in lines]
    input_tensor = torch.tensor([encoded_lines], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = tagger_model(input_tensor)
        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
        predicted_tags = [ix_to_tag.get(pid, "PROCESS_NODE") for pid in pred_ids]

    mermaid_script = assembler.assemble(lines, predicted_tags)
    return mermaid_script, predicted_tags

sample_input_1 = \"\"\"START
INPUT n
IF n % 2 == 0
    OUTPUT "Even"
ELSE
    OUTPUT "Odd"
END IF
END\"\"\"

mermaid_out_1, tags_out_1 = translate_pseudocode(sample_input_1)
print("=== Sample 1 (IF-ELSE with Strings) ===")
print(sample_input_1)
print("\\n=== Predicted Statement Roles ===")
for l, t in zip(sample_input_1.splitlines(), tags_out_1):
    print(f"  {l:<25} -> {t}")
print("\\n=== Generated Mermaid Script ===")
print(mermaid_out_1)

sample_input_2 = \"\"\"FOR i = 1 TO n
    IF i % 2 == 0
        FOR j = 1 TO i
            OUTPUT j
        END FOR
    ELSE
        OUTPUT 0
    END IF
END FOR\"\"\"

mermaid_out_2, tags_out_2 = translate_pseudocode(sample_input_2)
print("\\n" + "=" * 55 + "\\n")
print("=== Sample 2 (Nested FOR and IF-ELSE) ===")
print(sample_input_2)
print("\\n=== Generated Mermaid Script ===")
print(mermaid_out_2)"""
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

print(f"Successfully generated PseudoFlow_Translator.ipynb with {len(cells)} cells ({len(cells)//2} stages).")
