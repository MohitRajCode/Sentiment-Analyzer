from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn

app = FastAPI()

# =========================
# LOAD YOUR TRAINED MODEL
# =========================
word_to_idx = torch.load("vocab.pt")

class SentimentModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.rnn = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.rnn(x)
        return self.fc(hidden[-1])

model = SentimentModel(len(word_to_idx))
model.load_state_dict(torch.load("model.pt"))
model.eval()

# =========================
# ENCODE FUNCTION
# =========================
def encode(sentence):
    return torch.tensor([
        word_to_idx.get(word.lower(), 0)
        for word in sentence.split()
    ]).unsqueeze(0)

# =========================
# UI PAGE
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Sentiment Analyzer</title>
        <style>
            body { font-family: Arial; text-align: center; margin-top: 50px; }
            textarea { width: 400px; height: 100px; }
            button { padding: 10px 20px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>💬 Sentiment Analyzer</h1>
        <form action="/predict" method="post">
            <textarea name="text" placeholder="Enter your review..."></textarea><br>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """

# =========================
# PREDICT ROUTE
# =========================
@app.post("/predict", response_class=HTMLResponse)
def predict(text: str = Form(...)):
    x = encode(text)

    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()

    result = "Positive 😄" if pred == 1 else "Negative 😑"

    return f"""
    <html>
    <body style="text-align:center; font-family:Arial;">
        <h2>Result: {result}</h2>
        <a href="/">Try another</a>
    </body>
    </html>
    """
