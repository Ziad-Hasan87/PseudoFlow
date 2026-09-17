#!/usr/bin/env python3
"""
PseudoFlow Local Web Server
Serves the web UI and provides an optional Neural Model inference API endpoint (/api/translate).
"""

import http.server
import socketserver
import os
import sys
import json
import re

# Add parent directory to sys.path so we can import models and utils if needed
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Optional PyTorch Neural Model Loading
device = None
model = None
src_vocab = None
tag_to_id = None
id_to_tag = None

try:
    import torch
    import torch.nn as nn

    class BiRNNStatementClassifier(nn.Module):
        def __init__(self, vocab_size, num_classes, emb_dim=64, hidden_dim=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x):
            embedded = self.embedding(x)
            outputs, hidden = self.gru(embedded)
            # Pool across sequence length
            pooled = torch.max(outputs, dim=1)[0]
            logits = self.fc(pooled)
            return logits

    vocab_file = os.path.join(PARENT_DIR, 'vocabularies.json')
    model_file = os.path.join(PARENT_DIR, 'pseudoflow_tagger.pt')

    if os.path.exists(vocab_file) and os.path.exists(model_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vdata = json.load(f)
        src_vocab = vdata['src_vocab']
        tag_to_id = vdata.get('tag_to_id', {})
        id_to_tag = {v: k for k, v in tag_to_id.items()}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BiRNNStatementClassifier(len(src_vocab), len(tag_to_id)).to(device)
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"[PseudoFlow Server] Successfully loaded trained BiRNN model from {model_file} on {device}")
    else:
        print("[PseudoFlow Server] Trained weights not found. Falling back to rule-based neural assembler.")
except Exception as e:
    print(f"[PseudoFlow Server] PyTorch loading skipped ({e}). Falling back to heuristic classifier.")


def neural_classify_line(line):
    """Classifies a line using the PyTorch model if available, else rule-based fallback."""
    if model is not None and src_vocab is not None and id_to_tag is not None:
        try:
            tokens = re.findall(r'[a-zA-Z_]\w*|[0-9]+|[^\s\w]', line.strip())
            if not tokens:
                return "NONE"
            indices = [src_vocab.get(t, src_vocab.get('<unk>', 1)) for t in tokens]
            tensor = torch.tensor([indices], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(tensor)
                pred_idx = torch.argmax(logits, dim=1).item()
                return id_to_tag.get(pred_idx, "PROCESS_NODE")
        except Exception:
            pass

    # Heuristic fallback
    trimmed = line.strip()
    if not trimmed: return "NONE"
    if re.match(r'^START\b', trimmed, re.I): return "START_NODE"
    if re.match(r'^END\b', trimmed, re.I) and not re.match(r'^(END\s+IF|END\s+FOR|END\s+WHILE)\b', trimmed, re.I): return "END_NODE"
    if re.match(r'^IF\b', trimmed, re.I): return "IF_DECISION"
    if re.match(r'^ELSE\b', trimmed, re.I): return "ELSE_BRANCH"
    if re.match(r'^END\s+IF\b', trimmed, re.I): return "END_IF"
    if re.match(r'^(FOR|WHILE)\b', trimmed, re.I): return "LOOP_HEADER"
    if re.match(r'^END\s+FOR\b', trimmed, re.I): return "END_FOR"
    if re.match(r'^END\s+WHILE\b', trimmed, re.I): return "END_LOOP"
    if re.match(r'^(INPUT|OUTPUT|PRINT|READ|WRITE)\b', trimmed, re.I): return "IO_NODE"
    return "PROCESS_NODE"


class PseudoFlowRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == '/api/classify':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(body)
                code = data.get('code', '')
                lines = code.split('\n')
                classified = []
                for line in lines:
                    tag = neural_classify_line(line)
                    classified.append({"line": line, "tag": tag})
                
                response_data = {"status": "success", "classified": classified}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            except Exception as ex:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(ex)}).encode('utf-8'))
            return

        self.send_response(404)
        self.end_headers()


def run_server(port=PORT):
    # Allow address reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), PseudoFlowRequestHandler) as httpd:
        print(f"=================================================================")
        print(f" PseudoFlow Web UI Server running at: http://localhost:{port}")
        print(f" Serving directory: {DIRECTORY}")
        print(f" Press Ctrl+C to stop server")
        print(f"=================================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopping...")
            httpd.server_close()


if __name__ == '__main__':
    port = PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
