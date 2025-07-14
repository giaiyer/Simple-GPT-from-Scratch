#testing

import tensorflow as tf
import numpy as np
import os

text = """
to be or not to be that is the question whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune or to take arms against a sea of troubles
and by opposing end them to die to sleep no more and by a sleep to say we end
the heart-ache and the thousand natural shocks that flesh is heir to
tis a consummation devoutly to be wish’d to die to sleep to sleep perchance to dream
ay there's the rub for in that sleep of death what dreams may come
when we have shuffled off this mortal coil must give us pause
there's the respect that makes calamity of so long life

but soft what light through yonder window breaks
it is the east and juliet is the sun arise fair sun and kill the envious moon
who is already sick and pale with grief that thou her maid art far more fair than she
be not her maid since she is envious her vestal livery is but sick and green
and none but fools do wear it cast it off

friends romans countrymen lend me your ears
i come to bury caesar not to praise him the evil that men do lives after them
the good is oft interred with their bones so let it be with caesar
the noble brutus hath told you caesar was ambitious if it were so it was a grievous fault
and grievously hath caesar answered it here under leave of brutus and the rest
for brutus is an honourable man so are they all all honourable men
"""

vocab = sorted(set(text))
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(vocab)
if ' ' not in char2idx:
    char2idx['<PAD>'] = 0
    idx2char[0] = '<PAD>'
    if ' ' in vocab:
        PAD_TOKEN_ID = char2idx[' ']
    else:
        if 0 in idx2char and idx2char[0] != ' ':
             old_char_at_0 = idx2char[0]
             new_index = max(char2idx.values()) + 1
             char2idx[old_char_at_0] = new_index
             idx2char[new_index] = old_char_at_0
        PAD_TOKEN_ID = 0
        char2idx[' '] = PAD_TOKEN_ID
        idx2char[PAD_TOKEN_ID] = ' '
        vocab_size += 1
else:
    PAD_TOKEN_ID = char2idx[' ']

encoded = np.array([char2idx[c] for c in text], dtype=np.int32)

block_size = 32
def get_data(encoded, block_size):
    X, Y = [], []
    for i in range(len(encoded) - block_size):
        X.append(encoded[i:i+block_size])
        Y.append(encoded[i+1:i+block_size+1])
    return np.array(X), np.array(Y)

X, Y = get_data(encoded, block_size)

class MiniGPT(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim=64, max_len=64):
        super().__init__()
        self.token_embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_embed = tf.keras.layers.Embedding(max_len, embed_dim)
        
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=embed_dim)
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.final_ln = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        
        self.out_head = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        B, T = tf.shape(x)[0], tf.shape(x)[1]

        tok = self.token_embed(x)
        pos = self.pos_embed(tf.range(0, T))
        
        h = tok + pos
        
        norm_h1 = self.ln1(h)
        
        attn_out = self.attn(query=norm_h1, value=norm_h1, key=norm_h1, use_causal_mask=True)
        
        h = h + attn_out
        
        norm_h2 = self.ln2(h)
        
        ffn_out = self.ffn(norm_h2)
        
        h = h + ffn_out
        
        final_norm_h = self.final_ln(h)
        
        return self.out_head(final_norm_h)

model_path = "mini_gpt.weights.h5"
model = MiniGPT(vocab_size, max_len=block_size)

if os.path.exists(model_path):
    print("Loading model...")
    model(tf.zeros((1, block_size), dtype=tf.int32))
    model.load_weights(model_path)
else:
    print("Training model...")
    model(tf.zeros((1, block_size), dtype=tf.int32))
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor='loss', save_best_only=True, save_weights_only=True
        )
    ]
    model.fit(X, Y, epochs=200, batch_size=16, callbacks=callbacks)

def generate_text(start_string, length=200, temperature=0.7, top_k=None):
    start_string = start_string.lower()
    input_ids = []
    for char in start_string:
        if char in char2idx:
            input_ids.append(char2idx[char])
        else:
            input_ids.append(PAD_TOKEN_ID) 
            print(f"Warning: Character '{char}' in prompt not in vocabulary. Replacing with space.")

    generated_text = list(start_string)

    for _ in range(length):
        context_ids = input_ids[-block_size:]
        
        if len(context_ids) < block_size:
            context_ids = [PAD_TOKEN_ID] * (block_size - len(context_ids)) + context_ids

        x = tf.constant([context_ids], dtype=tf.int32)

        logits = model(x)[0, -1]

        logits = logits / temperature

        if top_k is not None:
            top_k_values, top_k_indices = tf.math.top_k(logits, k=top_k, sorted=False)
            
            mask = tf.scatter_nd(tf.expand_dims(top_k_indices, axis=1), tf.ones_like(top_k_indices, dtype=tf.float32), logits.shape)
            logits = logits * mask + (1 - mask) * tf.constant(-1e9, dtype=tf.float32)
        
        next_id = tf.random.categorical(tf.expand_dims(logits, 0), num_samples=1)[0,0].numpy()
        
        input_ids.append(next_id)
        
        if next_id in idx2char:
            generated_text.append(idx2char[next_id])
        else:
            generated_text.append('?')

    return "".join(generated_text)

print("\n--- MiniGPT Text Generator ---")
print(f"Vocabulary size: {vocab_size}")
print(f"Context window (block_size): {block_size}")
print("Enter a prompt to generate text. Type 'exit' to quit.")

while True:
    prompt = input("\nEnter prompt (or 'exit'): ").strip()
    if prompt.lower() == "exit":
        break
    
    print("\n--- Generated Text ---")
    generated_output = generate_text(prompt, length=200, temperature=1.0, top_k=5)
    print(generated_output)
    print("----------------------")