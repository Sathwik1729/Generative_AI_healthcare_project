import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from collections import Counter
import re

# 1. TOKENIZATION AND VOCABULARY BUILDING
class SimpleTokenizer:
    """
    Simple tokenizer for educational purposes
    In practice, you'd use BPE (Byte-Pair Encoding) like GPT or SentencePiece
    """
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        # Special tokens
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
    def build_vocab(self, texts):
        """Build vocabulary from text corpus"""
        word_freq = Counter()
        
        for text in texts:
            words = self.tokenize_text(text)
            word_freq.update(words)
        
        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        
        # Add most frequent words
        most_common = word_freq.most_common(self.vocab_size - len(self.special_tokens))
        
        for word, _ in most_common:
            if word not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
    
    def tokenize_text(self, text):
        """Simple word tokenization"""
        text = text.lower()
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def encode(self, text, add_special_tokens=True):
        """Convert text to token IDs"""
        words = self.tokenize_text(text)
        
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.word_to_id[self.bos_token])
        
        for word in words:
            token_ids.append(self.word_to_id.get(word, self.word_to_id[self.unk_token]))
        
        if add_special_tokens:
            token_ids.append(self.word_to_id[self.eos_token])
        
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Convert token IDs back to text"""
        words = []
        for token_id in token_ids:
            word = self.id_to_word.get(token_id, self.unk_token)
            if skip_special_tokens and word in self.special_tokens:
                continue
            words.append(word)
        return ' '.join(words)
    
    @property
    def pad_token_id(self):
        return self.word_to_id[self.pad_token]
    
    @property
    def eos_token_id(self):
        return self.word_to_id[self.eos_token]

# 2. DATASET FOR LANGUAGE MODELING
class LanguageModelingDataset(Dataset):
    """
    Dataset for training language models
    """
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Concatenate all texts with separator tokens
        all_token_ids = []
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            all_token_ids.extend(token_ids)
        
        # Create sequences of max_length
        for i in range(0, len(all_token_ids) - max_length + 1, max_length):
            sequence = all_token_ids[i:i + max_length]
            if len(sequence) == max_length:
                self.examples.append(sequence)
        
        # If no full sequences, create shorter ones and pad
        if len(self.examples) == 0:
            # Create sequences from individual texts, padding if necessary
            for text in texts:
                token_ids = tokenizer.encode(text, add_special_tokens=True)
                if len(token_ids) >= 2:  # At least input and target
                    # Pad or truncate to max_length
                    if len(token_ids) < max_length:
                        # Pad with pad tokens
                        token_ids.extend([tokenizer.pad_token_id] * (max_length - len(token_ids)))
                    else:
                        # Truncate
                        token_ids = token_ids[:max_length]
                    
                    self.examples.append(token_ids)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        sequence = self.examples[idx]
        
        # Input is all tokens except the last
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        # Target is all tokens except the first (shifted by 1)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'targets': target_ids
        }

# 3. ROTARY POSITIONAL EMBEDDING (RoPE) - Used in modern LLMs
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) - more effective than standard positional encoding
    Used in models like LLaMA, GPT-NeoX
    """
    def __init__(self, dim, max_seq_length=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # Create position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # Compute frequencies
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]
        
        return cos_emb, sin_emb

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to query and key tensors"""
    # Split the last dimension in half
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    
    # Apply rotation
    q_rot = torch.cat([-q2, q1], dim=-1)
    k_rot = torch.cat([-k2, k1], dim=-1)
    
    # Apply cos and sin
    q = q * cos + q_rot * sin
    k = k * cos + k_rot * sin
    
    return q, k

# 4. IMPROVED ATTENTION WITH RoPE
class RoPEMultiHeadAttention(nn.Module):
    """Multi-head attention with Rotary Positional Embedding"""
    def __init__(self, d_model, num_heads, max_seq_length=2048):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_length)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(x, seq_len)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(out)

# 5. LAYER NORMALIZATION VARIANTS
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization - used in LLaMA
    More efficient than standard LayerNorm
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# 6. SWISH/SiLU ACTIVATION FUNCTION
class SwiGLU(nn.Module):
    """
    SwiGLU activation function used in modern transformers like LLaMA
    Combines Swish activation with gating mechanism
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

# 7. MODERN TRANSFORMER BLOCK
class ModernTransformerBlock(nn.Module):
    """
    Modern transformer block with RoPE, RMSNorm, and SwiGLU
    Similar to what's used in LLaMA and other recent models
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = RoPEMultiHeadAttention(d_model, num_heads)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # SwiGLU requires 2/3 * d_ff hidden dimension
        hidden_dim = int(2 * d_ff / 3)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2 * hidden_dim, bias=False),
            SwiGLU(2 * hidden_dim),
            nn.Linear(hidden_dim, d_model, bias=False)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm architecture (norm before attention/ffn)
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

# 8. COMPLETE GPT-STYLE TRANSFORMER MODEL
class GPTTransformer(nn.Module):
    """
    Complete GPT-style transformer for language modeling
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token embeddings (no positional encoding - will be handled by RoPE in attention)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            ModernTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection to vocabulary
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights properly"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_length, device):
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        return mask == 0
    
    def forward(self, x):
        seq_length = x.size(1)
        device = x.device
        
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(x)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_length, device)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

# 9. TRAINING LOOP WITH GRADIENT ACCUMULATION
class Trainer:
    """
    Training class with modern techniques
    """
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader=None, 
                 lr=1e-4, weight_decay=0.01, warmup_steps=1000, max_steps=10000):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0) if step < warmup_steps 
                                   else 0.5 ** ((step - warmup_steps) / (max_steps - warmup_steps))
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
    def train_epoch(self, gradient_accumulation_steps=1):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for i, batch in enumerate(self.train_dataloader):
            input_ids = batch['input_ids']
            targets = batch['targets']
            
            # Forward pass
            logits = self.model(input_ids)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
        
        return total_loss / len(self.train_dataloader)
    
    def evaluate(self):
        """Evaluate the model"""
        if self.val_dataloader is None:
            return None
            
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids']
                targets = batch['targets']
                
                logits = self.model(input_ids)
                loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss.item()
        
        return total_loss / len(self.val_dataloader)

# 10. ADVANCED TEXT GENERATION STRATEGIES (FIXED)
def nucleus_sampling(logits, p=0.9, temperature=1.0):
    """
    Nucleus (top-p) sampling - more dynamic than top-k
    """
    if temperature > 0:
        logits = logits / temperature
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = 0
    
    # Scatter back to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('inf')
    
    return logits

def generate_with_strategies(model, tokenizer, prompt, max_length=50, 
                           strategy='nucleus', temperature=1.0, top_k=50, top_p=0.9):
    """
    Advanced text generation with multiple sampling strategies (FIXED)
    """
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            
            if strategy == 'greedy':
                next_token = torch.argmax(next_token_logits, dim=-1)
            elif strategy == 'top_k':
                # Top-k sampling
                if top_k > 0:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < values[-1]] = -float('inf')
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            elif strategy == 'nucleus':
                # Nucleus (top-p) sampling
                next_token_logits = nucleus_sampling(next_token_logits, top_p, temperature)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Standard sampling
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # FIX: Properly handle tensor dimensions
            next_token = next_token.view(1, 1)  # Ensure it's (1, 1) shape
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0].tolist())

# 11. ATTENTION PATTERN ANALYSIS
def analyze_attention_patterns(model, text, tokenizer, layer_idx=0):
    """
    Analyze attention patterns to understand what the model is focusing on
    """
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    # Hook to capture attention weights
    attention_weights = []
    
    def attention_hook(module, input, output):
        # Assuming output is (attention_output, attention_weights)
        if isinstance(output, tuple) and len(output) == 2:
            attention_weights.append(output[1].detach())
    
    # Register hook on the specified layer
    hook = model.layers[layer_idx].attention.register_forward_hook(attention_hook)
    
    with torch.no_grad():
        _ = model(input_ids)
    
    hook.remove()
    
    if attention_weights:
        # Average over heads
        attn = attention_weights[0].mean(dim=1).squeeze(0)  # Shape: (seq_len, seq_len)
        
        # Convert to numpy for easier analysis
        attn_np = attn.cpu().numpy()
        
        # Find tokens with highest attention
        token_words = [tokenizer.decode([token]) for token in tokens]
        
        print(f"Attention analysis for layer {layer_idx}:")
        print(f"Tokens: {token_words}")
        print(f"Attention matrix shape: {attn_np.shape}")
        
        # Show which tokens each position attends to most
        for i, word in enumerate(token_words):
            top_attended = np.argsort(attn_np[i])[-3:][::-1]  # Top 3
            attended_words = [token_words[j] for j in top_attended]
            print(f"'{word}' attends most to: {attended_words}")
        
        return attn_np, token_words
    
    return None, None

# 12. KNOWLEDGE DISTILLATION
class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss for training smaller student models
    """
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, targets):
        # Distillation loss (KL divergence between teacher and student)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(student_logits, targets)
        
        # Combined loss
        return self.alpha * distill_loss + (1 - self.alpha) * ce_loss

# 13. MODEL PROFILING AND OPTIMIZATION
def profile_model(model, input_shape=(1, 512), vocab_size=50000):
    """
    Profile model for memory usage and inference speed
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, vocab_size, input_shape)
    
    # Measure memory usage
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Warm up
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    # Measure inference time
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    
    avg_time = (time.time() - start_time) / 100
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Profile:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"Average inference time: {avg_time * 1000:.2f} ms")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'inference_time_ms': avg_time * 1000
    }

# 14. PERPLEXITY CALCULATION
def calculate_perplexity(model, tokenizer, texts, max_length=512):
    """
    Calculate perplexity - a common metric for language models
    Lower perplexity indicates better model performance
    """
    model.eval()
    total_log_likelihood = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) < 2:
                continue
                
            # Truncate if too long
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            input_ids = torch.tensor([tokens], dtype=torch.long)
            
            # Get model predictions
            outputs = model(input_ids)
            logits = outputs[0, :-1, :]  # All but last position
            targets = input_ids[0, 1:]   # All but first position
            
            # Calculate log likelihood
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            # Filter out padding tokens if any
            mask = targets != tokenizer.pad_token_id
            token_log_probs = token_log_probs[mask]
            
            total_log_likelihood += token_log_probs.sum().item()
            total_tokens += len(token_log_probs)
    
    if total_tokens == 0:
        return float('inf')
    
    avg_log_likelihood = total_log_likelihood / total_tokens
    perplexity = math.exp(-avg_log_likelihood)
    
    return perplexity

# 15. MODEL CHECKPOINTING
def save_checkpoint(model, tokenizer, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'vocab_size': len(tokenizer.word_to_id),
        'word_to_id': tokenizer.word_to_id,
        'id_to_word': tokenizer.id_to_word,
        'model_config': {
            'vocab_size': len(tokenizer.word_to_id),
            'd_model': model.d_model,
            'num_heads': model.layers[0].attention.num_heads,
            'num_layers': len(model.layers),
            'd_ff': model.layers[0].feed_forward[0].out_features // 2,  # Approximate
            'max_seq_length': model.max_seq_length
        }
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    # Reconstruct tokenizer
    tokenizer = SimpleTokenizer(vocab_size=checkpoint['vocab_size'])
    tokenizer.word_to_id = checkpoint['word_to_id']
    tokenizer.id_to_word = checkpoint['id_to_word']
    
    # Reconstruct model
    config = checkpoint['model_config']
    model = GPTTransformer(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return model, tokenizer, checkpoint
def batch_generate(model, tokenizer, prompts, max_length=50, strategy='nucleus', 
                  temperature=0.8, top_p=0.9, batch_size=4):
    """
    Generate text for multiple prompts in batches for efficiency
    """
    model.eval()
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_results = []
        
        # Encode all prompts in batch
        batch_token_ids = []
        max_prompt_len = 0
        
        for prompt in batch_prompts:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            batch_token_ids.append(tokens)
            max_prompt_len = max(max_prompt_len, len(tokens))
        
        # Pad prompts to same length
        padded_batch = []
        for tokens in batch_token_ids:
            if len(tokens) < max_prompt_len:
                # Pad at the beginning with pad tokens
                padded_tokens = [tokenizer.pad_token_id] * (max_prompt_len - len(tokens)) + tokens
            else:
                padded_tokens = tokens
            padded_batch.append(padded_tokens)
        
        # Convert to tensor
        input_ids = torch.tensor(padded_batch, dtype=torch.long)
        original_seq_len = input_ids.size(1)
        
        # Track which sequences are still active (not finished)
        active_sequences = torch.ones(input_ids.size(0), dtype=torch.bool)
        
        with torch.no_grad():
            for step in range(max_length):
                if not active_sequences.any():
                    break
                
                # Get predictions for all sequences
                outputs = model(input_ids)
                next_token_logits = outputs[:, -1, :]  # Shape: (batch_size, vocab_size)
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Generate next tokens based on strategy
                if strategy == 'greedy':
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                elif strategy == 'top_k':
                    # Top-k sampling for each sequence in batch
                    next_tokens = torch.zeros(input_ids.size(0), dtype=torch.long)
                    for seq_idx in range(input_ids.size(0)):
                        if active_sequences[seq_idx]:
                            logits = next_token_logits[seq_idx].clone()
                            # Apply top-k filtering
                            if hasattr(model, 'top_k') and model.top_k > 0:
                                top_k = getattr(model, 'top_k', 50)
                                values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                                logits[logits < values[-1]] = -float('inf')
                            
                            probs = F.softmax(logits, dim=-1)
                            next_tokens[seq_idx] = torch.multinomial(probs, num_samples=1)
                
                elif strategy == 'nucleus':
                    # Nucleus (top-p) sampling for each sequence in batch
                    next_tokens = torch.zeros(input_ids.size(0), dtype=torch.long)
                    for seq_idx in range(input_ids.size(0)):
                        if active_sequences[seq_idx]:
                            logits = next_token_logits[seq_idx].clone()
                            # Apply nucleus sampling
                            logits = nucleus_sampling(logits, top_p, temperature=1.0)  # Temperature already applied
                            probs = F.softmax(logits, dim=-1)
                            next_tokens[seq_idx] = torch.multinomial(probs, num_samples=1)
                
                else:  # Standard sampling
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # Append next tokens to sequences
                next_tokens = next_tokens.unsqueeze(1)
                input_ids = torch.cat([input_ids, next_tokens], dim=1)
                
                # Check for EOS tokens and update active sequences
                for seq_idx in range(input_ids.size(0)):
                    if next_tokens[seq_idx, 0].item() == tokenizer.eos_token_id:
                        active_sequences[seq_idx] = False
        
        # Decode generated sequences
        for seq_idx, prompt in enumerate(batch_prompts):
            # Get the generated part (excluding the original prompt)
            generated_tokens = input_ids[seq_idx, original_seq_len:].tolist()
            
            # Remove EOS token if present
            if tokenizer.eos_token_id in generated_tokens:
                eos_idx = generated_tokens.index(tokenizer.eos_token_id)
                generated_tokens = generated_tokens[:eos_idx]
            
            # Decode and combine with original prompt
            if generated_tokens:
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_text = prompt + " " + generated_text
            else:
                full_text = prompt
            
            batch_results.append(full_text.strip())
        
        results.extend(batch_results)
    
    return results

# 17. EXAMPLE USAGE AND DEMONSTRATION
def demo_modern_transformer():
    """
    Complete demonstration of the modern transformer implementation
    """
    print("=== Modern Transformer Language Model Demo ===\n")
    
    # Sample training data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world of technology.",
        "Machine learning models require large amounts of data for training.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning has revolutionized computer vision and speech recognition.",
        "Transformers have become the dominant architecture for language models.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Large language models can generate coherent and contextually relevant text.",
        "Training neural networks requires careful tuning of hyperparameters.",
        "The future of AI depends on developing more efficient and capable models."
    ]
    
    print("1. Building vocabulary...")
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab(sample_texts)
    print(f"Vocabulary size: {len(tokenizer.word_to_id)}")
    print(f"Sample tokens: {list(tokenizer.word_to_id.keys())[:10]}")
    
    print("\n2. Creating dataset...")
    dataset = LanguageModelingDataset(sample_texts, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"Dataset size: {len(dataset)} sequences")
    
    print("\n3. Initializing model...")
    model = GPTTransformer(
        vocab_size=len(tokenizer.word_to_id),
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_length=512,
        dropout=0.1
    )
    
    # Profile the model
    print("\n4. Model profiling...")
    profile_stats = profile_model(model, input_shape=(1, 64), vocab_size=len(tokenizer.word_to_id))
    
    print("\n5. Training setup...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=dataloader,
        lr=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        max_steps=1000
    )
    
    print("\n6. Training for a few epochs...")
    for epoch in range(3):
        train_loss = trainer.train_epoch(gradient_accumulation_steps=2)
        print(f"Epoch {epoch + 1}: Training Loss = {train_loss:.4f}")
    
    print("\n7. Text generation examples...")
    test_prompts = [
        "Artificial intelligence",
        "The future of",
        "Machine learning"
    ]
    
    strategies = ['greedy', 'nucleus', 'top_k']
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} Generation ---")
        generated_texts = batch_generate(
            model, tokenizer, test_prompts, 
            max_length=20, strategy=strategy, 
            temperature=0.8, top_p=0.9
        )
        for prompt, generated in zip(test_prompts, generated_texts):
            print(f"Prompt: '{prompt}'")
            print(f"Generated: '{generated}'")
            print()
    
    print("\n8. Calculating perplexity...")
    perplexity = calculate_perplexity(model, tokenizer, sample_texts[:3])
    print(f"Model perplexity: {perplexity:.2f}")
    
    print("\n9. Attention analysis...")
    attn_matrix, tokens = analyze_attention_patterns(
        model, "Artificial intelligence is amazing", tokenizer, layer_idx=0
    )
    
    print("\n10. Saving checkpoint...")
    save_checkpoint(model, tokenizer, trainer.optimizer, 3, train_loss, "model_checkpoint.pt")
    
    print("\n=== Demo Complete ===")
    return model, tokenizer

# 18. ADVANCED FEATURES AND UTILITIES

class ModelEvaluator:
    """
    Comprehensive model evaluation utilities
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate_generation_quality(self, prompts, num_samples=5):
        """
        Evaluate generation quality using multiple metrics
        """
        results = {
            'diversity': [],
            'coherence': [],
            'fluency': []
        }
        
        for prompt in prompts:
            generations = []
            for _ in range(num_samples):
                generated = generate_with_strategies(
                    self.model, self.tokenizer, prompt,
                    max_length=30, strategy='nucleus', temperature=0.8
                )
                generations.append(generated)
            
            # Calculate diversity (unique n-grams)
            all_words = ' '.join(generations).split()
            unique_words = set(all_words)
            diversity = len(unique_words) / len(all_words) if all_words else 0
            results['diversity'].append(diversity)
            
            # Simple coherence measure (sentence completion)
            avg_length = sum(len(g.split()) for g in generations) / len(generations)
            coherence = min(avg_length / 20, 1.0)  # Normalize
            results['coherence'].append(coherence)
            
            # Simple fluency measure (no repeated words in sequence)
            fluency_scores = []
            for gen in generations:
                words = gen.split()
                repeated = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
                fluency = max(0, 1 - repeated / max(len(words), 1))
                fluency_scores.append(fluency)
            results['fluency'].append(sum(fluency_scores) / len(fluency_scores))
        
        return {
            metric: sum(values) / len(values) 
            for metric, values in results.items()
        }
    
    def benchmark_inference_speed(self, sequence_lengths=[64, 128, 256, 512]):
        """
        Benchmark inference speed for different sequence lengths
        """
        import time
        
        results = {}
        self.model.eval()
        
        for seq_len in sequence_lengths:
            # Create dummy input
            dummy_input = torch.randint(
                0, len(self.tokenizer.word_to_id), 
                (1, seq_len)
            )
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_input)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(50):
                    start = time.time()
                    _ = self.model(dummy_input)
                    times.append(time.time() - start)
            
            avg_time = sum(times) / len(times)
            results[seq_len] = {
                'avg_time_ms': avg_time * 1000,
                'tokens_per_second': seq_len / avg_time
            }
        
        return results

# 19. EXPORT AND DEPLOYMENT UTILITIES

def export_for_inference(model, tokenizer, export_path="model_for_inference.pt"):
    """
    Export model in a format optimized for inference
    """
    model.eval()
    
    # Create a simplified version for inference
    inference_model = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': len(tokenizer.word_to_id),
            'd_model': model.d_model,
            'num_heads': model.layers[0].attention.num_heads,
            'num_layers': len(model.layers),
            'd_ff': model.layers[0].feed_forward[0].out_features // 2,
            'max_seq_length': model.max_seq_length
        },
        'tokenizer': {
            'word_to_id': tokenizer.word_to_id,
            'id_to_word': tokenizer.id_to_word,
            'vocab_size': len(tokenizer.word_to_id)
        }
    }
    
    torch.save(inference_model, export_path)
    print(f"Model exported for inference to {export_path}")

def load_for_inference(model_path, device='cpu'):
    """
    Load model optimized for inference
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct tokenizer
    tokenizer = SimpleTokenizer(vocab_size=checkpoint['tokenizer']['vocab_size'])
    tokenizer.word_to_id = checkpoint['tokenizer']['word_to_id']
    tokenizer.id_to_word = checkpoint['tokenizer']['id_to_word']
    
    # Reconstruct model
    model = GPTTransformer(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

# 20. FINAL EXAMPLE: COMPLETE TRAINING PIPELINE

if __name__ == "__main__":
    # Run the complete demonstration
    print("Starting Modern Transformer Training Pipeline...\n")
    
    try:
        model, tokenizer = demo_modern_transformer()
        print("\n✅ Training pipeline completed successfully!")
        
        # Additional evaluations
        print("\nRunning additional evaluations...")
        evaluator = ModelEvaluator(model, tokenizer)
        
        # Generation quality
        test_prompts = ["Technology is", "The future", "Learning"]
        quality_scores = evaluator.evaluate_generation_quality(test_prompts)
        print(f"Generation Quality Scores: {quality_scores}")
        
        # Speed benchmark
        speed_results = evaluator.benchmark_inference_speed([64, 128, 256])
        print(f"Speed Benchmark: {speed_results}")
        
        # Export for deployment
        export_for_inference(model, tokenizer, "final_model.pt")
        
        print("\n🎉 All evaluations and exports completed!")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()