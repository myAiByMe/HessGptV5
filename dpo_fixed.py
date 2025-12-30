#!/usr/bin/env python3
"""
🚀 HessGPT DPO - FIXED VERSION
✅ Format ChatML unifié avec SFT
✅ Masking correct (loss sur réponses uniquement)
✅ Datasets de qualité
✅ Early stopping KL
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys, os, time, math, random, json
from tqdm import tqdm
from transformers import GPT2Tokenizer
from datasets import load_dataset
from datetime import datetime

sys.path.append('./Core/Model')

print("="*80)
print("🎯 HessGPT DPO - FIXED")
print("="*80)

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    # Model
    'vocab_size': 50257,
    'embed_dim': 1280,
    'num_heads': 20,
    'num_layers': 20,
    'max_seq_len': 1024,
    'dropout': 0.1,
    
    # DPO Training
    'batch_size': 8,
    'gradient_accumulation': 8,
    'num_epochs': 1,
    'learning_rate': 5e-7,     # Très conservateur
    'beta': 0.1,               # DPO temperature
    'warmup_ratio': 0.1,
    'max_grad_norm': 0.3,
    'weight_decay': 0.0,
    
    # Datasets
    'orca_dpo_samples': 30000,
    'hh_rlhf_samples': 10000,
    'ultrafeedback_samples': 10000,
    
    # Filters
    'min_length': 20,
    'max_length': 512,
    'min_score_diff': 0.1,
    
    # Validation
    'val_split': 0.05,
    'validate_every': 500,        # ✅ Réduit
    'max_kl_divergence': 0.5,
    
    # Checkpoints
    'checkpoint_dir': './checkpoints/dpo_fixed',
    'sft_checkpoint': './checkpoints/sft_optimal/best.pt',
}

os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("❌ GPU requise!")
    sys.exit(1)

print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ============================================
# FORMAT ChatML (ALIGNÉ AVEC SFT)
# ============================================
def format_chatml(system_prompt, user_msg, assistant_msg):
    """Format ChatML unifié"""
    if system_prompt:
        text = f"<|system|>\n{system_prompt}\n<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}"
    else:
        text = f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}"
    return text

def extract_prompt_and_response(text):
    """Extrait prompt et réponse depuis ChatML"""
    assistant_marker = "<|assistant|>\n"
    
    if assistant_marker in text:
        parts = text.split(assistant_marker, 1)
        prompt = parts[0] + assistant_marker
        response = parts[1] if len(parts) > 1 else ""
        return prompt, response
    
    return text, ""

# ============================================
# DPO DATASET (AVEC MASKING)
# ============================================
class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize prompt (shared)
        prompt = item['prompt']
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # Tokenize responses
        chosen_ids = self.tokenizer.encode(item['chosen'], add_special_tokens=False)
        rejected_ids = self.tokenizer.encode(item['rejected'], add_special_tokens=False)
        
        # Combine: prompt + response + eos
        chosen_full = prompt_ids + chosen_ids + [self.tokenizer.eos_token_id]
        rejected_full = prompt_ids + rejected_ids + [self.tokenizer.eos_token_id]
        
        # Truncate
        chosen_full = chosen_full[:self.max_length]
        rejected_full = rejected_full[:self.max_length]
        
        # Pad
        chosen_padded = chosen_full + [self.tokenizer.pad_token_id] * (self.max_length - len(chosen_full))
        rejected_padded = rejected_full + [self.tokenizer.pad_token_id] * (self.max_length - len(rejected_full))
        
        return {
            'chosen_ids': torch.tensor(chosen_padded, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_padded, dtype=torch.long),
            'chosen_length': len(chosen_full),
            'rejected_length': len(rejected_full),
            'prompt_length': len(prompt_ids),  # ✅ AJOUTÉ
        }

# ============================================
# LOAD DATASETS (FORMAT UNIFIÉ)
# ============================================
print("\n" + "="*80)
print("📥 CHARGEMENT DATASETS")
print("="*80)

def load_with_cache(name, loader_func, max_samples):
    cache_file = f"data/cache/{name}_dpo_fixed.pt"
    os.makedirs("data/cache", exist_ok=True)
    
    if os.path.exists(cache_file):
        print(f"✅ Cache: {cache_file}")
        return torch.load(cache_file)['data']
    
    print(f"⏳ Loading {name}...")
    data = loader_func(max_samples)
    
    torch.save({'data': data}, cache_file)
    print(f"💾 Saved: {len(data):,} pairs")
    
    return data

def load_orca_dpo(max_samples):
    """Intel Orca DPO - Format ChatML"""
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
    data = []
    
    for item in tqdm(dataset, desc="Orca DPO"):
        if len(data) >= max_samples:
            break
        
        system = item.get('system', '').strip()
        question = item.get('question', '').strip()
        chosen = item.get('chosen', '').strip()
        rejected = item.get('rejected', '').strip()
        
        if not (question and chosen and rejected):
            continue
        
        # Validate lengths
        if len(tokenizer.encode(chosen)) < CONFIG['min_length']:
            continue
        if len(tokenizer.encode(rejected)) < CONFIG['min_length']:
            continue
        
        # ✅ Format ChatML unifié
        prompt_text = f"<|system|>\n{system}\n<|user|>\n{question}\n<|assistant|>\n" if system else f"<|user|>\n{question}\n<|assistant|>\n"
        
        data.append({
            'prompt': prompt_text,
            'chosen': chosen,
            'rejected': rejected,
            'source': 'orca_dpo'
        })
    
    random.shuffle(data)
    return data

def load_hh_rlhf_dpo(max_samples):
    """Anthropic HH-RLHF - Format ChatML"""
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    data = []
    
    for item in tqdm(dataset, desc="HH-RLHF"):
        if len(data) >= max_samples:
            break
        
        chosen_text = item.get('chosen', '').strip()
        rejected_text = item.get('rejected', '').strip()
        
        if not (chosen_text and rejected_text):
            continue
        
        # Parse conversations
        def parse_hh_conversation(text):
            """Parse HH-RLHF vers ChatML"""
            lines = text.split('\n\n')
            messages = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('Human:'):
                    msg = line.replace('Human:', '').strip()
                    messages.append(('user', msg))
                elif line.startswith('Assistant:'):
                    msg = line.replace('Assistant:', '').strip()
                    messages.append(('assistant', msg))
            
            # Build ChatML prompt (all user messages)
            prompt = ""
            last_response = ""
            
            for role, msg in messages:
                if role == 'user':
                    prompt += f"<|user|>\n{msg}\n"
                elif role == 'assistant':
                    last_response = msg
            
            prompt += "<|assistant|>\n"
            
            return prompt, last_response
        
        prompt_chosen, response_chosen = parse_hh_conversation(chosen_text)
        prompt_rejected, response_rejected = parse_hh_conversation(rejected_text)
        
        # Prompts should be identical
        if prompt_chosen != prompt_rejected:
            continue
        
        if response_chosen and response_rejected:
            # Validate
            if len(tokenizer.encode(response_chosen)) < CONFIG['min_length']:
                continue
            if len(tokenizer.encode(response_rejected)) < CONFIG['min_length']:
                continue
            
            data.append({
                'prompt': prompt_chosen,
                'chosen': response_chosen,
                'rejected': response_rejected,
                'source': 'hh_rlhf'
            })
    
    random.shuffle(data)
    return data

def load_ultrafeedback(max_samples):
    """Argilla UltraFeedback - Format ChatML"""
    try:
        dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
        data = []
        
        for item in tqdm(dataset, desc="UltraFeedback"):
            if len(data) >= max_samples:
                break
            
            prompt = item.get('prompt', '').strip()
            chosen_list = item.get('chosen', [])
            rejected_list = item.get('rejected', [])
            
            if not (prompt and chosen_list and rejected_list):
                continue
            
            # ✅ Handle different formats
            if isinstance(chosen_list, list) and len(chosen_list) > 0:
                if isinstance(chosen_list[0], dict):
                    chosen_text = chosen_list[0].get('content', '').strip()
                else:
                    chosen_text = str(chosen_list[0]).strip()
            else:
                continue
            
            if isinstance(rejected_list, list) and len(rejected_list) > 0:
                if isinstance(rejected_list[0], dict):
                    rejected_text = rejected_list[0].get('content', '').strip()
                else:
                    rejected_text = str(rejected_list[0]).strip()
            else:
                continue
            
            if chosen_text and rejected_text:
                # Validate
                if len(tokenizer.encode(chosen_text)) < CONFIG['min_length']:
                    continue
                if len(tokenizer.encode(rejected_text)) < CONFIG['min_length']:
                    continue
                
                # ✅ Format ChatML
                formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
                
                data.append({
                    'prompt': formatted_prompt,
                    'chosen': chosen_text,
                    'rejected': rejected_text,
                    'source': 'ultrafeedback'
                })
        
        random.shuffle(data)
        return data
    
    except Exception as e:
        print(f"⚠️ UltraFeedback error: {e}")
        return []

# Load datasets
all_data = []
dataset_stats = {}

loaders = [
    ('orca_dpo', load_orca_dpo, CONFIG['orca_dpo_samples']),
    ('hh_rlhf', load_hh_rlhf_dpo, CONFIG['hh_rlhf_samples']),
    ('ultrafeedback', load_ultrafeedback, CONFIG['ultrafeedback_samples']),
]

for name, loader, max_samples in loaders:
    print(f"\n📚 {name.upper()}")
    data = load_with_cache(name, loader, max_samples)
    all_data.extend(data)
    dataset_stats[name] = len(data)

# Stats
print("\n" + "="*80)
print("📊 COMPOSITION")
print("="*80)

total = len(all_data)
for name, count in dataset_stats.items():
    pct = (count / total * 100) if total > 0 else 0
    print(f"  {name:15s} {count:>7,} ({pct:>5.1f}%)")
print(f"  {'─'*50}")
print(f"  {'TOTAL':15s} {total:>7,}")

# Split
random.shuffle(all_data)
split_idx = int(len(all_data) * (1 - CONFIG['val_split']))
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]

print(f"\n✅ Train: {len(train_data):,}")
print(f"✅ Val:   {len(val_data):,}")

# Dataloaders
train_dataset = DPODataset(train_data, tokenizer, CONFIG['max_seq_len'])
val_dataset = DPODataset(val_data, tokenizer, CONFIG['max_seq_len'])

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ============================================
# MODEL
# ============================================
print("\n" + "="*80)
print("🤖 MODÈLE")
print("="*80)

from HessGpt import HessGPT

policy_model = HessGPT(
    vocab_size=CONFIG['vocab_size'],
    embed_dim=CONFIG['embed_dim'],
    num_heads=CONFIG['num_heads'],
    num_layers=CONFIG['num_layers'],
    max_seq_len=CONFIG['max_seq_len'],
    dropout=CONFIG['dropout']
).to(device)

reference_model = HessGPT(
    vocab_size=CONFIG['vocab_size'],
    embed_dim=CONFIG['embed_dim'],
    num_heads=CONFIG['num_heads'],
    num_layers=CONFIG['num_layers'],
    max_seq_len=CONFIG['max_seq_len'],
    dropout=CONFIG['dropout']
).to(device)

# Load SFT
if not os.path.exists(CONFIG['sft_checkpoint']):
    print(f"❌ SFT checkpoint not found: {CONFIG['sft_checkpoint']}")
    sys.exit(1)

print(f"\n📂 Loading SFT: {CONFIG['sft_checkpoint']}")
sft_checkpoint = torch.load(CONFIG['sft_checkpoint'], map_location=device)

policy_model.load_state_dict(sft_checkpoint['model_state_dict'])
reference_model.load_state_dict(sft_checkpoint['model_state_dict'])

# Freeze reference
for param in reference_model.parameters():
    param.requires_grad = False
reference_model.eval()

print(f"✅ Policy model loaded (trainable)")
print(f"✅ Reference model loaded (frozen)")

# ============================================
# DPO LOSS (AVEC MASKING)
# ============================================
def compute_dpo_loss(policy_model, reference_model, batch, beta):
    """
    DPO Loss avec masking correct:
    - Loss uniquement sur les tokens de réponse (après <|assistant|>)
    - Ignore les tokens de prompt
    """
    chosen_ids = batch['chosen_ids'].to(device)
    rejected_ids = batch['rejected_ids'].to(device)
    prompt_length = batch['prompt_length'].to(device)
    
    # Policy logits
    with torch.amp.autocast('cuda'):
        policy_chosen_logits, _ = policy_model(chosen_ids[:, :-1])
        policy_rejected_logits, _ = policy_model(rejected_ids[:, :-1])
        
        # Reference logits (no grad)
        with torch.no_grad():
            ref_chosen_logits, _ = reference_model(chosen_ids[:, :-1])
            ref_rejected_logits, _ = reference_model(rejected_ids[:, :-1])
    
    # Compute log probs (AVEC MASKING)
    def get_log_probs(logits, labels, prompt_len, total_len):
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs
        token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # ✅ MASK: ignore prompt + padding
        batch_size, seq_len = labels.shape
        mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
        
        for i in range(batch_size):
            # Start après prompt, end avant padding
            start = prompt_len[i].item()
            end = total_len[i].item() - 1  # -1 car on a shifted
            if start < end:
                mask[i, start:end] = True
        
        token_log_probs = token_log_probs * mask.float()
        
        # Sum over response tokens only
        return token_log_probs.sum(dim=1)
    
    policy_chosen_lp = get_log_probs(
        policy_chosen_logits,
        chosen_ids[:, 1:],
        prompt_length,
        batch['chosen_length']
    )
    
    policy_rejected_lp = get_log_probs(
        policy_rejected_logits,
        rejected_ids[:, 1:],
        prompt_length,
        batch['rejected_length']
    )
    
    ref_chosen_lp = get_log_probs(
        ref_chosen_logits,
        chosen_ids[:, 1:],
        prompt_length,
        batch['chosen_length']
    )
    
    ref_rejected_lp = get_log_probs(
        ref_rejected_logits,
        rejected_ids[:, 1:],
        prompt_length,
        batch['rejected_length']
    )
    
    # Log ratios
    chosen_log_ratio = policy_chosen_lp - ref_chosen_lp
    rejected_log_ratio = policy_rejected_lp - ref_rejected_lp
    
    # DPO loss
    logits_diff = beta * (chosen_log_ratio - rejected_log_ratio)
    loss = -F.logsigmoid(logits_diff).mean()
    
    # Stats
    chosen_rewards = beta * chosen_log_ratio.detach()
    rejected_rewards = beta * rejected_log_ratio.detach()
    reward_margin = (chosen_rewards - rejected_rewards).mean()
    
    # KL divergence
    kl_div = (chosen_log_ratio.detach().abs().mean() + rejected_log_ratio.detach().abs().mean()) / 2
    
    return loss, {
        'reward_margin': reward_margin.item(),
        'chosen_reward': chosen_rewards.mean().item(),
        'rejected_reward': rejected_rewards.mean().item(),
        'kl_divergence': kl_div.item(),
    }

# ============================================
# TRAINING
# ============================================
print("\n" + "="*80)
print("🚀 DPO TRAINING")
print("="*80)

total_steps = (len(train_loader) * CONFIG['num_epochs']) // CONFIG['gradient_accumulation']
warmup_steps = int(CONFIG['warmup_ratio'] * total_steps)

print(f"\n📊 Config:")
print(f"   Steps:   {total_steps:,}")
print(f"   Warmup:  {warmup_steps:,}")
print(f"   LR:      {CONFIG['learning_rate']:.2e}")
print(f"   Beta:    {CONFIG['beta']}")

optimizer = torch.optim.AdamW(
    policy_model.parameters(),
    lr=CONFIG['learning_rate'],
    betas=(0.9, 0.999),
    weight_decay=CONFIG['weight_decay'],
    fused=True
)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = torch.amp.GradScaler('cuda')
policy_model.train()

global_step = 0
training_history = []

@torch.no_grad()
def validate():
    policy_model.eval()
    
    total_loss = 0
    total_margin = 0
    total_kl = 0
    count = 0
    
    for batch in tqdm(val_loader, desc="Validation", leave=False):
        loss, stats = compute_dpo_loss(policy_model, reference_model, batch, CONFIG['beta'])
        
        total_loss += loss.item()
        total_margin += stats['reward_margin']
        total_kl += stats['kl_divergence']
        count += 1
    
    policy_model.train()
    
    return {
        'loss': total_loss / count,
        'margin': total_margin / count,
        'kl': total_kl / count,
    }

start_time = time.time()

for epoch in range(CONFIG['num_epochs']):
    epoch_loss = 0
    epoch_margin = 0
    epoch_kl = 0
    valid_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    
    for batch_idx, batch in enumerate(pbar):
        loss, stats = compute_dpo_loss(policy_model, reference_model, batch, CONFIG['beta'])
        loss = loss / CONFIG['gradient_accumulation']
        
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                CONFIG['max_grad_norm']
            )
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            global_step += 1
            
            # Validate
            if global_step % CONFIG['validate_every'] == 0:
                val_stats = validate()
                
                print(f"\n{'─'*80}")
                print(f"📊 Step {global_step:,}")
                print(f"   Val Loss:   {val_stats['loss']:.4f}")
                print(f"   Margin:     {val_stats['margin']:.4f}")
                print(f"   KL Div:     {val_stats['kl']:.4f}")
                print(f"{'─'*80}\n")
                
                training_history.append({
                    'step': global_step,
                    'val_loss': val_stats['loss'],
                    'val_margin': val_stats['margin'],
                    'val_kl': val_stats['kl'],
                    'train_loss': loss.item() * CONFIG['gradient_accumulation'],
                    'lr': scheduler.get_last_lr()[0],
                })
                
                # Check KL divergence
                if val_stats['kl'] > CONFIG['max_kl_divergence']:
                    print(f"\n⚠️ KL divergence trop élevé: {val_stats['kl']:.4f}")
                    print(f"   Stopping pour éviter drift")
                    break
        
        epoch_loss += loss.item() * CONFIG['gradient_accumulation']
        epoch_margin += stats['reward_margin']
        epoch_kl += stats['kl_divergence']
        valid_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item() * CONFIG["gradient_accumulation"]:.4f}',
            'margin': f'{stats["reward_margin"]:.3f}',
            'kl': f'{stats["kl_divergence"]:.3f}',
        })

# Final validation
final_stats = validate()
elapsed = (time.time() - start_time) / 3600

print("\n" + "="*80)
print("✅ DPO TERMINÉ")
print("="*80)

print(f"\n📊 STATS FINALES:")
print(f"   Val Loss:       {final_stats['loss']:.4f}")
print(f"   Reward Margin:  {final_stats['margin']:.4f}")
print(f"   KL Divergence:  {final_stats['kl']:.4f}")
print(f"   Temps:          {elapsed:.2f}h")

# Save
final_checkpoint = os.path.join(CONFIG['checkpoint_dir'], 'final.pt')
torch.save({
    'model_state_dict': policy_model.state_dict(),
    'config': CONFIG,
    'final_stats': final_stats,
    'training_history': training_history,
    'dataset_stats': dataset_stats,
}, final_checkpoint)

print(f"\n💾 Saved: {final_checkpoint}")

# History
history_file = os.path.join(CONFIG['checkpoint_dir'], 'history.json')
with open(history_file, 'w') as f:
    json.dump({
        'config': {k: v for k, v in CONFIG.items() if not callable(v)},
        'final_stats': final_stats,
        'training_history': training_history,
        'dataset_stats': dataset_stats,
    }, f, indent=2)

print(f"💾 History: {history_file}")
print("\n✅ DPO SUCCÈS!")
