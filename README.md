# Decoder-Only Transformer

A modular GPT-style Transformer architecture implemented in PyTorch.

---

## Features

- Token Embedding Layer
- Sinusoidal Positional Encoding
- Decoder Stack with Causal Masking
- Output Projection Head
- Modular Architecture Design
- PyTorch Implementation

---

## Project Structure

project/
│
├── model/
│   ├── __init__.py
│   ├── EmbeddingBlock.py
│   ├── PositionalBlock.py
│   ├── DecoderStack.py
│   ├── Head.py
│   ├── Model.py
│   └── main.py
│
├── requirements.txt
└── README.md

Model Architecture:

Input Tokens
      ↓
Embedding Layer
      ↓
Positional Encoding
      ↓
Decoder Stack
      ↓
LayerNorm
      ↓
Output Head
      ↓
Vocabulary Logits

