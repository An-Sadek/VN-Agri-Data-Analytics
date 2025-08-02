import torch
import torch.nn as nn
torch.torch.manual_seed(42)
vocab = ["i", "love", "pizza"]

word_to_index = {"i": 0, "love": 1, "pizza": 2}
sentence = ["i", "love", "pizza"]
indices = [word_to_index[word] for word in sentence]  # [0, 1, 2]

# Create embedding layer
vocab_size = len(vocab)        # 3
embedding_dim = 5              # You choose this (e.g., 50, 100, etc.)
embedding = nn.Embedding(vocab_size, embedding_dim)

# Convert indices to tensor
input_indices = torch.tensor(indices)  # shape: [3]

# Get embeddings
embedded = embedding(input_indices)    # shape: [3, 5]
print(embedded)

d = embedded.shape[1]
dk = 5
dv = 4

W_q = torch.nn.Parameter(torch.rand(d, dk))
W_k = torch.nn.Parameter(torch.rand(d, dk))
W_v = torch.nn.Parameter(torch.rand(d, dv))

Q = embedded @ W_q
K = embedded @ W_k
V = embedded @ W_v

print("\nQ")
print(Q)

print("\nK")
print(K)

print("\nV")
print(V)

scores = Q @ K.T
print(scores)

scaled_scores = scores/(d**1/2)
print(scaled_scores)

attention_weights = torch.softmax(scaled_scores, dim=1)
print(attention_weights)

output = attention_weights @ V
print(output)