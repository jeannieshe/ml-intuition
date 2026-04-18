import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
lr = 1e-3
max_iters = 5000
eval_interval = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.3

# reproducibility
torch.manual_seed(1337)

# loading in the data
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding="utf-8") as f:
    text = f.read()

# grab unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from chars to ints and back
char_to_int = { ch:i for i, ch in enumerate(chars) }
int_to_char = { i:ch for i, ch in enumerate(chars) }
encode = lambda input: [char_to_int[ch] for ch in input]
decode = lambda input: "".join([int_to_char[ch] for ch in input])

# data loader into torch tensors
data = torch.tensor(encode(text), dtype=torch.long)

# train and test splits
n = int(0.9 * len(data)) # first 90% as train, rest as validation
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a batch size of inputs x and targets y
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,)) # generates a tensor of shape (4,) that randomly samples various starting indices for the training data
    x = torch.stack([data[i:i + block_size] for i in idx]) # row in tensor of size (4, 8)
    y = torch.stack([data[i+1:i + block_size+1] for i in idx]) # grab the predicted tokens which would be moving forward by one by the input
    return x, y

# loss estimator
@torch.no_grad() # we are never calling backward here, so tell pytorch that it doesn't need to worry about creating or storing the computation graph.
def estimate_loss(): # calculate a significantly less noisy loss
    out = {}
    model.eval() # put model in evaluation mode. some layers have different behaviors in diff mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # create a tensor of size (eval_iters)
        for k in range(eval_iters): # evaluate it multiple times, then store the mean() of those evaluations to try and calculate the loss averaged over multiple batches
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # stores the losses within keys 'train', 'val'
    model.train() # put model back to train mode
    return out

class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        # define the query, key, value forward passes
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Directly calculate one forward pass of an attention head"""
        # construct the key, query vectors
        B, T, C = x.shape
        q = self.query(x) # (B, T, C)
        k = self.key(x) # (B, T, C)

        # calculate the attention weights and scale them by dividing by sqrt(dim) where dim == C in this case
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) x (B, C, T) = (B, T, T)
        # mask the future tokens so this is a decoder attention mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -float("inf"))
        # softmax to exponentiate and normalize across all of the columns, so each row sums to 1
        # have to softmax after the masking so that the probability distribution holds as expected
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # apply the weight to the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) x (B, T, C) = (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.multihead = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # blend info across all heads before writing back to the residual streamhow. needs to be the same dim as the residual connections.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.multihead], dim=-1) # self attention output
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # final projection layer going back into the residual pathway. needs to be the same dim as the residual connections.
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """One transformer block: communication followed by computation."""
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads to parallelize
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # adding residual connections
        x = x + self.ffwd(self.ln2(x))
        return x
    
class LayerNorm:
    """Normalizes the rows instead of the columns over our inputs."""
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.ones(dim)

    def __call__(self, x):
        # x.shape = (B, T, C). take average over the time dimension
        xmean = x.mean(1, keepdim = True)
        xvar = x.var(1, keepdim = True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

# construct a simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # every integer in the input refers to the lookup table, plucks out a row corresponding to that index (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # produces (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # NEW! apply multiple BLOCKS of transformer
        logits = self.lm_head(x) # (B, T, vocab_size)
        # organized into a (B, T, C) -> batch, time, channel -> 4, 8, 65 (vocab_size) shape tensor

        if targets is None:
            loss = None
        else:
            # need to reshape the logits into (B, C, T) since cross_entropy requires the channel dim as the second dim
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets) # measures the quality of the logits to the targets. how well are we predicting the next character based on the logits?

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) tensor
        for _ in range(max_new_tokens):
            # ensure that idx is no more than block_size going in. grab the last block_size elements in the time T dimension
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C) since we grabbed the last token in the time dimension
            # apply softmax to get probabilities across the channels
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # given a probability distribution, sample one token for each of the channels (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# simple training script
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch to train
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)

    # update the parameters
    optimizer.zero_grad() # clear the gradients
    loss.backward() # calculate new gradients
    optimizer.step() # update the parameters

# generate from the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = m.generate(context, max_new_tokens=500) # 500 new tokens

generated = generated[0].tolist() # decode needs a list passed into it. grab the first row of the generated tensor which has shape (1, 501)
print(decode(generated))