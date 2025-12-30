#!/usr/bin/env python3
"""
🚀 HessGPT SFT - CONFIGURATION OPTIMALE
✅ Datasets recommandés par priorité
✅ Format ChatML pour conversations
✅ Protection catastrophic forgetting
✅ Multi-task + Safety
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
print("🚀 HessGPT SFT - OPTIMAL")
print("="*80)

# ============================================
# CONFIGURATION OPTIMALE
# ============================================
CONFIG = {
    # Modèle (aligné pre-training)
    'vocab_size': 50257,
    'embed_dim': 1280,
    'num_heads': 20,
    'num_layers': 20,
    'max_seq_len': 1024,  # ✅ ALIGNÉ avec pre-training
    'dropout': 0.1,
    
    # Training (protège pre-training)
    'batch_size': 16,      # ✅ Augmenté (1024 tokens)
    'gradient_accumulation': 4,
    'num_epochs': 2,       # ✅ 2-3 recommandé
    'learning_rate': 1e-5, # ✅ Recommandation (10x moins que pre-train)
    'warmup_ratio': 0.03,  # ✅ Recommandation
    'max_grad_norm': 0.5,  # Strict pour éviter forgetting
    'weight_decay': 0.01,  # ✅ Recommandation
    
    # Datasets par priorité (total ~350k)
    # 1️⃣ ESSENTIEL
    'openorca_samples': 80000,       # Multi-task ⭐⭐⭐
    'ultrachat_samples': 70000,      # Conversations ⭐⭐⭐
    'alpaca_samples': 50000,         # Instructions classiques ⭐⭐
    
    # 2️⃣ CHATBOT (use case)
    'wizard_samples': 50000,         # Instructions évoluées ⭐⭐
    'sharegpt_samples': 40000,       # Conversations réelles
    'dailydialog_samples': 30000,    # Dialogues quotidiens
    'oasst1_samples': 20000,         # Dialogues qualité
    'dolly_samples': 10000,          # Tâches variées
    
    # 3️⃣ SAFETY
    'hh_rlhf_samples': 10000,        # Helpful + Harmless
    
    # Custom
    'synthetic_samples': 1000,
    
    # Filtres qualité
    'min_length': 20,
    'max_length': 512,  # Tokens par réponse
    'min_turns': 1,     # Minimum 1 échange
    'max_turns': 5,     # Maximum 5 tours
    'min_unique_ratio': 0.6,
    'max_repetition_ratio': 0.15,
    
    # Validation
    'val_split': 0.05,
    'validate_every': 500,
    'patience': 3,
    'min_improvement': 0.001,
    
    # Scheduler
    'scheduler_type': 'cosine',
    'min_lr_ratio': 0.1,
    
    # Checkpoints
    'checkpoint_dir': './checkpoints/sft_optimal',
    'pretrain_checkpoint': './checkpoints/HessGpt_V5.pt',
}

os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("❌ GPU requise!")
    sys.exit(1)

print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# ============================================
# FORMAT ChatML
# ============================================
def format_chatml(messages):
    """
    Convertit messages au format ChatML
    
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello!"}
    ]
    """
    formatted = ""
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '').strip()
        
        if role == 'system':
            formatted += f"<|system|>\n{content}\n"
        elif role == 'user':
            formatted += f"<|user|>\n{content}\n"
        elif role == 'assistant':
            formatted += f"<|assistant|>\n{content}\n"
    
    return formatted

def format_alpaca(instruction, input_text="", output=""):
    """Format Alpaca classique"""
    if input_text:
        prompt = f"<|user|>\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n<|assistant|>\n{output}\n"
    else:
        prompt = f"<|user|>\n{instruction}\n<|assistant|>\n{output}\n"
    return prompt

# ============================================
# DATASET CLASS AVEC MASKING
# ============================================
class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format selon le type
        if 'messages' in item:
            # Format ChatML
            text = format_chatml(item['messages'])
        else:
            # Format Alpaca
            text = format_alpaca(
                item['instruction'],
                item.get('input', ''),
                item['output']
            )
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens.append(self.tokenizer.eos_token_id)
        
        # Truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        # ✅ MASKING: Loss uniquement sur assistant
        labels = torch.full_like(target_ids, -100)
        
        # Trouve les tokens <|assistant|>
        text_decoded = self.tokenizer.decode(tokens)
        assistant_marker = "<|assistant|>"
        
        if assistant_marker in text_decoded:
            assistant_tokens = self.tokenizer.encode(assistant_marker, add_special_tokens=False)
            
            # Cherche la position
            for i in range(len(tokens) - len(assistant_tokens)):
                if tokens[i:i+len(assistant_tokens)] == assistant_tokens:
                    # Applique loss après le marker
                    start_idx = i + len(assistant_tokens)
                    if start_idx < len(labels):
                        labels[start_idx:] = target_ids[start_idx:]
                    break
        
        # Padding
        pad_length = self.max_length - 1 - len(input_ids)
        if pad_length > 0:
            input_ids = F.pad(input_ids, (0, pad_length), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, (0, pad_length), value=-100)
        
        return input_ids, labels

# ============================================
# FILTRES QUALITÉ
# ============================================
def calculate_repetition_score(text):
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    if not trigrams:
        return 0.0
    
    unique = len(set(trigrams))
    return 1.0 - (unique / len(trigrams))

def is_valid_response(text, config):
    if not text or len(text) < 10:
        return False
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not (config['min_length'] <= len(tokens) <= config['max_length']):
        return False
    
    words = text.split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < config['min_unique_ratio']:
            return False
    
    repetition = calculate_repetition_score(text)
    if repetition > config['max_repetition_ratio']:
        return False
    
    return True

# ============================================
# LOAD DATASETS
# ============================================
print("\n" + "="*80)
print("📥 CHARGEMENT DATASETS (PRIORITÉ OPTIMALE)")
print("="*80)

def load_dataset_with_cache(name, loader_func, max_samples, config):
    cache_file = f"data/cache/{name}_optimal.pt"
    os.makedirs("data/cache", exist_ok=True)
    
    if os.path.exists(cache_file):
        print(f"✅ Cache: {cache_file}")
        return torch.load(cache_file)['data']
    
    print(f"⏳ Chargement {name}...")
    data = loader_func(max_samples, config)
    
    torch.save({'data': data}, cache_file)
    print(f"💾 Cache: {len(data):,} samples")
    
    return data

# ============================================
# 1️⃣ ESSENTIEL
# ============================================

def load_openorca(max_samples, config):
    """OpenOrca - Multi-task instructions ⭐⭐⭐"""
    dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    data = []
    
    for item in tqdm(dataset.take(max_samples * 2), desc="OpenOrca", total=max_samples):
        if len(data) >= max_samples:
            break
        
        system_prompt = item.get('system_prompt', 'You are a helpful assistant.').strip()
        question = item.get('question', '').strip()
        response = item.get('response', '').strip()
        
        if question and response and is_valid_response(response, config):
            data.append({
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': response}
                ],
                'source': 'openorca'
            })
    
    random.shuffle(data)
    return data

def load_ultrachat(max_samples, config):
    """UltraChat - Conversations naturelles ⭐⭐⭐"""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    data = []
    
    for item in tqdm(dataset, desc="UltraChat"):
        if len(data) >= max_samples:
            break
        
        messages = item.get('messages', [])
        if len(messages) < 2:
            continue
        
        # Vérifie format
        valid = True
        for msg in messages:
            if msg.get('role') not in ['system', 'user', 'assistant']:
                valid = False
                break
        
        if not valid:
            continue
        
        # Extrait dernière réponse pour validation
        last_assistant = None
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                last_assistant = msg.get('content', '')
                break
        
        if last_assistant and is_valid_response(last_assistant, config):
            data.append({
                'messages': messages,
                'source': 'ultrachat'
            })
    
    random.shuffle(data)
    return data

def load_alpaca(max_samples, config):
    """Alpaca - Instructions classiques ⭐⭐"""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    data = []
    
    for item in tqdm(dataset, desc="Alpaca"):
        if len(data) >= max_samples:
            break
        
        inst = item.get('instruction', '').strip()
        inp = item.get('input', '').strip()
        out = item.get('output', '').strip()
        
        if inst and out and is_valid_response(out, config):
            data.append({
                'instruction': inst,
                'input': inp,
                'output': out,
                'source': 'alpaca'
            })
    
    random.shuffle(data)
    return data

# ============================================
# 2️⃣ CHATBOT
# ============================================

def load_sharegpt(max_samples, config):
    """ShareGPT - Conversations réelles"""
    dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
    data = []
    
    for item in tqdm(dataset, desc="ShareGPT"):
        if len(data) >= max_samples:
            break
        
        conversations = item.get('conversations', [])
        if len(conversations) < 2:
            continue
        
        # Convertit au format messages
        messages = []
        for conv in conversations[:config['max_turns']*2]:
            role_map = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
            role = role_map.get(conv.get('from', ''), 'user')
            content = conv.get('value', '').strip()
            
            if content:
                messages.append({'role': role, 'content': content})
        
        # Valide dernière réponse
        if messages and messages[-1]['role'] == 'assistant':
            last_response = messages[-1]['content']
            if is_valid_response(last_response, config):
                data.append({
                    'messages': messages,
                    'source': 'sharegpt'
                })
    
    random.shuffle(data)
    return data

def load_wizard(max_samples, config):
    """WizardLM - Instructions évoluées ⭐⭐"""
    dataset = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train")
    data = []
    
    for item in tqdm(dataset, desc="WizardLM"):
        if len(data) >= max_samples:
            break
        
        conversations = item.get('conversations', [])
        if len(conversations) >= 2:
            user_msg = conversations[0].get('value', '').strip()
            assistant_msg = conversations[1].get('value', '').strip()
            
            if user_msg and assistant_msg and is_valid_response(assistant_msg, config):
                data.append({
                    'messages': [
                        {'role': 'user', 'content': user_msg},
                        {'role': 'assistant', 'content': assistant_msg}
                    ],
                    'source': 'wizard'
                })
    
    random.shuffle(data)
    return data

def load_dolly(max_samples, config):
    """Dolly - Tâches variées"""
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    data = []
    
    for item in tqdm(dataset, desc="Dolly"):
        if len(data) >= max_samples:
            break
        
        instruction = item.get('instruction', '').strip()
        context = item.get('context', '').strip()
        response = item.get('response', '').strip()
        
        if instruction and response and is_valid_response(response, config):
            # Combine instruction + context si présent
            user_content = f"{instruction}\n\nContext: {context}" if context else instruction
            
            data.append({
                'messages': [
                    {'role': 'user', 'content': user_content},
                    {'role': 'assistant', 'content': response}
                ],
                'source': 'dolly'
            })
    
    random.shuffle(data)
    return data

def load_dailydialog(max_samples, config):
    """DailyDialog - Dialogues quotidiens naturels"""
    try:
        dataset = load_dataset("daily_dialog", split="train")
        data = []
        
        for item in tqdm(dataset, desc="DailyDialog"):
            if len(data) >= max_samples:
                break
            
            dialog = item.get('dialog', [])
            if len(dialog) < 2:
                continue
            
            # Prend des paires user/assistant
            for i in range(0, len(dialog) - 1, 2):
                if len(data) >= max_samples:
                    break
                
                user_msg = dialog[i].strip()
                assistant_msg = dialog[i + 1].strip() if i + 1 < len(dialog) else ""
                
                if user_msg and assistant_msg and is_valid_response(assistant_msg, config):
                    data.append({
                        'messages': [
                            {'role': 'user', 'content': user_msg},
                            {'role': 'assistant', 'content': assistant_msg}
                        ],
                        'source': 'dailydialog'
                    })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"⚠️  DailyDialog non disponible: {e}")
        return []

# ✅ AJOUT DE LA LIGNE MANQUANTE
def load_oasst1(max_samples, config):
    """OpenAssistant - Dialogues qualité"""
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    
    messages_dict = {}
    for item in dataset:
        msg_id = item['message_id']
        messages_dict[msg_id] = item
    
    data = []
    processed = set()
    
    for msg_id, msg in tqdm(messages_dict.items(), desc="OASST1"):
        if len(data) >= max_samples:
            break
        
        if msg['role'] == 'assistant' and msg['parent_id'] and msg['parent_id'] not in processed:
            parent = messages_dict.get(msg['parent_id'])
            if parent and parent['role'] == 'prompter':
                user_text = parent['text'].strip()
                assistant_text = msg['text'].strip()
                
                if is_valid_response(assistant_text, config):
                    data.append({
                        'messages': [
                            {'role': 'user', 'content': user_text},
                            {'role': 'assistant', 'content': assistant_text}
                        ],
                        'source': 'oasst1'
                    })
                    processed.add(msg['parent_id'])
    
    random.shuffle(data)
    return data

    
# ============================================
# 3️⃣ SAFETY
# ============================================

def load_hh_rlhf(max_samples, config):
    """Anthropic HH-RLHF - Helpful + Harmless"""
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        data = []
        
        for item in tqdm(dataset, desc="HH-RLHF"):
            if len(data) >= max_samples:
                break
            
            # Prend uniquement les "chosen" (bonnes réponses)
            chosen = item.get('chosen', '').strip()
            
            if not chosen:
                continue
            
            # Parse le dialogue
            lines = chosen.split('\n\n')
            messages = []
            
            for line in lines:
                if line.startswith('Human:'):
                    content = line.replace('Human:', '').strip()
                    messages.append({'role': 'user', 'content': content})
                elif line.startswith('Assistant:'):
                    content = line.replace('Assistant:', '').strip()
                    messages.append({'role': 'assistant', 'content': content})
            
            # Valide
            if messages and messages[-1]['role'] == 'assistant':
                last_response = messages[-1]['content']
                if is_valid_response(last_response, config):
                    data.append({
                        'messages': messages,
                        'source': 'hh_rlhf'
                    })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"⚠️  HH-RLHF non disponible: {e}")
        return []

# ============================================
# CHARGEMENT
# ============================================
print("\n1️⃣ ESSENTIEL (Multi-task)")
print("─" * 80)

loaders = [
    ('openorca', load_openorca, CONFIG['openorca_samples']),
    ('ultrachat', load_ultrachat, CONFIG['ultrachat_samples']),
    ('alpaca', load_alpaca, CONFIG['alpaca_samples']),
]

all_data = []
dataset_stats = {}

for name, loader, max_samples in loaders:
    print(f"\n📚 {name.upper()}")
    data = load_dataset_with_cache(name, loader, max_samples, CONFIG)
    all_data.extend(data)
    dataset_stats[name] = len(data)

print("\n2️⃣ CHATBOT (Use case)")
print("─" * 80)

chatbot_loaders = [
    ('wizard', load_wizard, CONFIG['wizard_samples']),
    ('sharegpt', load_sharegpt, CONFIG['sharegpt_samples']),
    ('dailydialog', load_dailydialog, CONFIG['dailydialog_samples']),
    ('oasst1', load_oasst1, CONFIG['oasst1_samples']),
    ('dolly', load_dolly, CONFIG['dolly_samples']),
]

for name, loader, max_samples in chatbot_loaders:
    print(f"\n💬 {name.upper()}")
    data = load_dataset_with_cache(name, loader, max_samples, CONFIG)
    all_data.extend(data)
    dataset_stats[name] = len(data)

print("\n3️⃣ SAFETY")
print("─" * 80)

print("\n🛡️ HH-RLHF")
hh_data = load_dataset_with_cache('hh_rlhf', load_hh_rlhf, CONFIG['hh_rlhf_samples'], CONFIG)
all_data.extend(hh_data)
dataset_stats['hh_rlhf'] = len(hh_data)

# Synthetic
synthetic_file = "synthetic_10k.jsonl"
if os.path.exists(synthetic_file):
    print(f"\n✨ SYNTHETIC")
    synthetic_data = []
    with open(synthetic_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= CONFIG['synthetic_samples']:
                break
            try:
                obj = json.loads(line)
                u = obj['user'].strip()
                a = obj['assistant'].strip()
                if is_valid_response(a, CONFIG):
                    synthetic_data.append({
                        'messages': [
                            {'role': 'user', 'content': u},
                            {'role': 'assistant', 'content': a}
                        ],
                        'source': 'synthetic'
                    })
            except:
                continue
    all_data.extend(synthetic_data)
    dataset_stats['synthetic'] = len(synthetic_data)
    print(f"✅ {len(synthetic_data):,} samples")

# ============================================
# STATS
# ============================================
print("\n" + "="*80)
print("📊 COMPOSITION FINALE")
print("="*80)

total = len(all_data)
for name, count in dataset_stats.items():
    pct = (count / total * 100) if total > 0 else 0
    priority = "⭐⭐⭐" if name in ['openorca', 'ultrachat'] else "⭐⭐" if name == 'alpaca' else "⭐"
    print(f"  {name:12s} {count:>7,} ({pct:>5.1f}%) {priority}")
print(f"  {'─'*50}")
print(f"  {'TOTAL':12s} {total:>7,}")

# ============================================
# STRATIFIED SPLIT
# ============================================
print("\n📊 Stratified split...")

data_by_source = {}
for item in all_data:
    source = item.get('source', 'unknown')
    if source not in data_by_source:
        data_by_source[source] = []
    data_by_source[source].append(item)

train_data, val_data = [], []

for source, items in data_by_source.items():
    n = len(items)
    split_idx = int(n * (1 - CONFIG['val_split']))
    train_data.extend(items[:split_idx])
    val_data.extend(items[split_idx:])

random.shuffle(train_data)
random.shuffle(val_data)

print(f"✅ Train: {len(train_data):,}")
print(f"✅ Val:   {len(val_data):,}")

# ============================================
# DATALOADERS
# ============================================
train_dataset = ChatDataset(train_data, tokenizer, CONFIG['max_seq_len'])
val_dataset = ChatDataset(val_data, tokenizer, CONFIG['max_seq_len'])

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
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
# MODÈLE
# ============================================
print("\n" + "="*80)
print("🤖 MODÈLE")
print("="*80)

from HessGpt import HessGPT

model = HessGPT(
    vocab_size=CONFIG['vocab_size'],
    embed_dim=CONFIG['embed_dim'],
    num_heads=CONFIG['num_heads'],
    num_layers=CONFIG['num_layers'],
    max_seq_len=CONFIG['max_seq_len'],
    dropout=CONFIG['dropout']
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"✅ Paramètres: {num_params/1e6:.1f}M")

# Load pre-training
if os.path.exists(CONFIG['pretrain_checkpoint']):
    print(f"\n📂 Pre-training: {CONFIG['pretrain_checkpoint']}")
    checkpoint = torch.load(CONFIG['pretrain_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Chargé!")
else:
    print(f"\n⚠️  Pre-training non trouvé, training from scratch")

# ============================================
# OPTIMIZER & SCHEDULER
# ============================================
total_steps = (len(train_loader) * CONFIG['num_epochs']) // CONFIG['gradient_accumulation']
warmup_steps = int(CONFIG['warmup_ratio'] * total_steps)

print(f"\n📊 Training:")
print(f"   Steps:   {total_steps:,}")
print(f"   Warmup:  {warmup_steps:,}")
print(f"   LR:      {CONFIG['learning_rate']}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    betas=(0.9, 0.95),
    weight_decay=CONFIG['weight_decay'],
    fused=True
)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return CONFIG['min_lr_ratio'] + (1.0 - CONFIG['min_lr_ratio']) * 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================
# VALIDATION
# ============================================
@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for x, y in tqdm(val_loader, desc="Validation", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
                reduction='sum'
            )
        
        mask = (y != -100)
        total_loss += loss.item()
        total_tokens += mask.sum().item()
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 10))
    
    model.train()
    return avg_loss, perplexity

# ============================================
# TRAINING
# ============================================
print("\n" + "="*80)
print("🚀 TRAINING")
print("="*80)

scaler = torch.amp.GradScaler('cuda')
model.train()

start_epoch = 0
global_step = 0
best_val_loss = float('inf')
patience_counter = 0
training_history = []

resume_file = os.path.join(CONFIG['checkpoint_dir'], 'resume.pt')
if os.path.exists(resume_file):
    print(f"\n♻️  REPRISE: {resume_file}")
    resume_data = torch.load(resume_file, map_location=device)
    model.load_state_dict(resume_data['model_state_dict'])
    optimizer.load_state_dict(resume_data['optimizer_state_dict'])
    scheduler.load_state_dict(resume_data['scheduler_state_dict'])
    scaler.load_state_dict(resume_data['scaler_state_dict'])
    start_epoch = resume_data['epoch']
    global_step = resume_data['global_step']
    best_val_loss = resume_data.get('best_val_loss', float('inf'))
    patience_counter = resume_data.get('patience_counter', 0)
    training_history = resume_data.get('training_history', [])
    print(f"✅ Epoch {start_epoch}, step {global_step:,}")

start_time = time.time()

for epoch in range(start_epoch, CONFIG['num_epochs']):
    epoch_loss = 0
    valid_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    
    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100
            ) / CONFIG['gradient_accumulation']
        
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CONFIG['max_grad_norm']
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            global_step += 1
            
            if global_step % CONFIG['validate_every'] == 0:
                val_loss, val_ppl = validate(model, val_loader, device)
                
                print(f"\n{'─'*80}")
                print(f"📊 Step {global_step:,}")
                print(f"   Val Loss: {val_loss:.4f}")
                print(f"   Val PPL:  {val_ppl:.2f}")
                print(f"   LR:       {scheduler.get_last_lr()[0]:.2e}")
                print(f"{'─'*80}\n")
                
                training_history.append({
                    'step': global_step,
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'train_loss': loss.item() * CONFIG['gradient_accumulation'],
                    'lr': scheduler.get_last_lr()[0],
                })
        
        epoch_loss += loss.item() * CONFIG['gradient_accumulation']
        valid_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item() * CONFIG["gradient_accumulation"]:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
        })
    
    # Fin epoch
    avg_train_loss = epoch_loss / max(valid_batches, 1)
    val_loss, val_ppl = validate(model, val_loader, device)
    
    elapsed = (time.time() - start_time) / 3600
    
    print(f"\n{'='*80}")
    print(f"✅ EPOCH {epoch+1}/{CONFIG['num_epochs']} TERMINÉE")
    print(f"   Train Loss: {avg_train_loss:.4f}")
    print(f"   Val Loss:   {val_loss:.4f}")
    print(f"   Val PPL:    {val_ppl:.2f}")
    print(f"   Temps:      {elapsed:.2f}h")
    print(f"{'='*80}")
    
    # Sauvegarde resume
    torch.save({
        'epoch': epoch + 1,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': CONFIG,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'training_history': training_history,
        'dataset_stats': dataset_stats,
    }, resume_file)
    
    # Early stopping
    improvement = best_val_loss - val_loss
    
    if improvement >= CONFIG['min_improvement']:
        best_val_loss = val_loss
        patience_counter = 0
        
        # Sauvegarde BEST
        best_file = os.path.join(CONFIG['checkpoint_dir'], 'best.pt')
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'train_loss': avg_train_loss,
            'training_history': training_history,
            'dataset_stats': dataset_stats,
        }, best_file)
        
        print(f"🏆 MEILLEUR MODÈLE! Val Loss: {val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"⚠️  Patience: {patience_counter}/{CONFIG['patience']}")
        
        if patience_counter >= CONFIG['patience']:
            print(f"\n🛑 EARLY STOPPING")
            break

# ============================================
# RÉSUMÉ
# ============================================
total_time = (time.time() - start_time) / 3600

print("\n" + "="*80)
print("🎉 TRAINING TERMINÉ!")
print("="*80)

print(f"\n📊 STATISTIQUES:")
print(f"   Best Val Loss: {best_val_loss:.4f}")
print(f"   Temps total:   {total_time:.2f}h")
print(f"   Total samples: {len(all_data):,}")
print(f"   Steps:         {global_step:,}")

print(f"\n📊 Composition:")
for name, count in dataset_stats.items():
    pct = (count / total * 100) if total > 0 else 0
    print(f"   {name:12s} {count:>7,} ({pct:>5.1f}%)")

print(f"\n📁 Checkpoints:")
print(f"   Best:   {os.path.join(CONFIG['checkpoint_dir'], 'best.pt')}")
print(f"   Resume: {os.path.join(CONFIG['checkpoint_dir'], 'resume.pt')}")

# Historique JSON
history_file = os.path.join(CONFIG['checkpoint_dir'], 'history.json')
with open(history_file, 'w') as f:
    json.dump({
        'config': {k: v for k, v in CONFIG.items() if not callable(v)},
        'final_stats': {
            'best_val_loss': best_val_loss,
            'total_time_hours': total_time,
            'total_samples': len(all_data),
            'steps': global_step,
            'epochs_completed': epoch + 1,
        },
        'dataset_stats': dataset_stats,
        'training_history': training_history,
        'timestamp': datetime.now().isoformat(),
    }, f, indent=2)

print(f"   History: {history_file}")

print("\n" + "="*80)
print("✅ SUCCÈS!")
print("="*80)

print("\n💡 PROCHAINES ÉTAPES:")
print("\n1. 🧪 Tester:")
print("   python test_hessgpt.py")
print("\n2. 🎮 Interactif:")
print("   python inference.py --checkpoint checkpoints/sft_optimal/best.pt --interactive")
print("\n3. 📊 Format ChatML:")
print("   <|user|>")
print("   Hello!")
print("   <|assistant|>")
print("   Hi! How can I help you today?")

    