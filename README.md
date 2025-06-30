# Simple GPT from Scratch

This repository contains a minimalist, character-level Generative Pre-trained Transformer (GPT) model implemented in TensorFlow/Keras. It's built from the ground up to demonstrate the fundamental components of a transformer-based language model, including self-attention and positional embeddings, in a straightforward and understandable manner.

The model trains on a small, embedded text corpus (excerpts from Shakespeare) and can generate new character sequences based on a user-provided prompt.

## Features

* **From Scratch Implementation:** Core transformer concepts (MultiHeadAttention, positional embeddings) are manually constructed.
* **Character-Level:** Processes text character by character for a simplified vocabulary.
* **Interactive Generation:** Allows direct text input for real-time generation.
* **Model Persistence:** Automatically saves and loads trained weights.
* **Sampling Controls:** Includes `temperature` and `top-k` parameters to influence generation style.

