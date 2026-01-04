#!/usr/bin/env python3
"""
🚀 HessGPT SFT - PREMIUM HIGH-QUALITY DATASETS
✅ 1 epoch optimal
✅ 480k+ samples haute qualité
✅ Format simple User/Assistant
✅ Sans ChatML
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
print("🚀 HessGPT SFT - PREMIUM QUALITY")
print("="*80)

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    # Modèle (identique)
    'vocab_size': 50257,
    'embed_dim': 1280,
    'num_heads': 20,
    'num_layers': 20,
    'max_seq_len': 512,
    'dropout': 0.1,
    
    # Training (identique)
    'batch_size': 16,
    'gradient_accumulation': 4,
    'num_epochs': 1,  # ✅ 1 epoch suffit avec 800k
    'learning_rate': 1e-5,
    'warmup_ratio': 0.03,
    'max_grad_norm': 0.5,
    'weight_decay': 0.01,
    
    # ============================================
    # 🏆 TIER 1: ELITE (~300k) 
    # ============================================
    # Dans la section CONFIG, ajoute :
    'alpaca_samples': 52000,  # Stanford Alpaca Original
    'orca_math_samples': 100000,        # 200k dispo → 100k
    'metamath_samples': 80000,          # 395k dispo → 80k
    'code_alpaca_samples': 20000,       # 20k dispo → ALL
    'evol_instruct_samples': 50000,     # 143k dispo → 50k
    'platypus_samples': 25000,          # 25k dispo → ALL
    'oasst2_samples': 50000,            # 161k dispo → 50k (NOUVEAU)
    'ultrafeedback_samples': 40000,     # 64k dispo → 40k (NOUVEAU)
    'slimorca_samples': 100000,         # 518k dispo → 100k (NOUVEAU)
    'no_robots_samples': 10000,         # 10k dispo → ALL (NOUVEAU)
    
    # 🏆 TIER 1 Total: ~475k (au lieu de 71k)
    
    # ============================================
    # 🥇 TIER 2: PREMIUM (~250k)
    # ============================================
    'openorca_samples': 80000,          # 3.2M dispo → 80k
    'ultrachat_samples': 60000,         # 200k dispo → 60k
    'wizardlm_samples': 50000,          # 70k dispo → 50k
    'nous_instruct_samples': 40000,     # 52k dispo → 40k
    'openhermes_samples': 60000,        # 1M dispo → 60k
    'airoboros_samples': 40000,         # 58k dispo → 40k
    'orca_dpo_samples': 12000,          # 12.9k dispo → ALL (NOUVEAU)
    'capybara_samples': 16000,          # 16k dispo → ALL (NOUVEAU)
    'puffin_samples': 3000,             # 3k dispo → ALL (NOUVEAU)
    'tulu_v2_samples': 50000,           # 326k dispo → 50k (NOUVEAU)
    
    # 🥇 TIER 2 Total: ~411k (au lieu de 120k)
    
    # ============================================
    # 🥈 TIER 3: SOLID (~150k)
    # ============================================
    'alpaca_gpt4_samples': 40000,       # 52k dispo → 40k
    'oasst1_samples': 40000,            # 88k dispo → 40k
    'flan_samples': 60000,              # 1.8M dispo → 60k
    'dolly_samples': 15000,             # 15k dispo → ALL
    'code_contests_samples': 13000,     # 13k dispo → ALL (NOUVEAU)
    'sciq_samples': 11000,              # 11.6k dispo → ALL (NOUVEAU)
    'glaive_code_samples': 50000,       # 136k dispo → 50k (NOUVEAU)
    
    # 🥈 TIER 3 Total: ~229k (au lieu de 43k)
    
    # ============================================
    # 🛡️ TIER 4: SAFETY (~50k)
    # ============================================
    'hh_rlhf_samples': 30000,           # 161k dispo → 30k
    'prosocial_samples': 20000,         # 165k dispo → 20k
    'safe_rlhf_samples': 10000,         # 30k dispo → 10k
    
    # 🛡️ TIER 4 Total: ~60k (au lieu de 30k)
    
    # ============================================
    # ✨ SYNTHETIC
    # ============================================
    'synthetic_samples': 10,
    
    # ============================================
    # GRAND TOTAL: ~1,180k samples
    # ============================================
    # Avec filtres qualité: ~800-900k samples effectifs
    
    # Filtres qualité (identiques)
    'min_length': 30,
    'max_length': 512,
    'min_unique_ratio': 0.65,
    'max_repetition_ratio': 0.12,
    'min_quality_score': 0.7,
    
    # Validation
    'val_split': 0.03,
    'validate_every': 1000,
    'patience': 3,
    'min_improvement': 0.001,
    
    # Paths
    'checkpoint_dir': './checkpoint/sft_premium_800k',
    'pretrain_checkpoint': './checkpoint/HessGpt_V5.pt',
    'cache_dir': './data/cache_premium_800k',
}

os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
os.makedirs(CONFIG['cache_dir'], exist_ok=True)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("❌ GPU requise!")
    sys.exit(1)

print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# ============================================
# FORMAT SIMPLE
# ============================================
def format_conversation(user_text, assistant_text, system_text=None):
    """Format simple User/Assistant"""
    conversation = ""
    
    if system_text:
        conversation += f"System: {system_text}\n\n"
    
    conversation += f"User: {user_text}\n"
    conversation += f"Assistant: {assistant_text}"
    
    return conversation

# ============================================
# FILTRES QUALITÉ
# ============================================
def calculate_text_quality(text):
    """Score qualité 0-1"""
    if not text or len(text) < 20:
        return 0.0
    
    words = text.split()
    if len(words) < 5:
        return 0.0
    
    unique_ratio = len(set(words)) / len(words)
    
    length_score = 1.0
    if len(words) < 50:
        length_score = len(words) / 50
    elif len(words) > 400:
        length_score = max(0.5, 1.0 - (len(words) - 400) / 400)
    
    sentence_markers = text.count('.') + text.count('!') + text.count('?')
    has_structure = 1.0 if sentence_markers >= 2 else 0.5
    
    quality = (unique_ratio * 0.4 + length_score * 0.4 + has_structure * 0.2)
    return quality

def calculate_repetition_score(text):
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    if not trigrams:
        return 0.0
    
    unique = len(set(trigrams))
    return 1.0 - (unique / len(trigrams))

def is_high_quality(text, config):
    """Filtres stricts"""
    if not text or len(text) < 20:
        return False
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not (config['min_length'] <= len(tokens) <= config['max_length']):
        return False
    
    quality = calculate_text_quality(text)
    if quality < config['min_quality_score']:
        return False
    
    words = text.split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < config['min_unique_ratio']:
            return False
    
    repetition = calculate_repetition_score(text)
    if repetition > config['max_repetition_ratio']:
        return False
    
    low_quality_markers = ['...', 'TODO', 'FIXME', '[insert', 'Lorem ipsum']
    if any(marker.lower() in text.lower() for marker in low_quality_markers):
        return False
    
    return True

# ============================================
# DATASET CLASS
# ============================================
class ConversationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        text = format_conversation(
            item['user'],
            item['assistant'],
            item.get('system')
        )
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens.append(self.tokenizer.eos_token_id)
        
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Masking: Loss uniquement sur Assistant
        labels = torch.full_like(target_ids, -100)
        
        text_decoded = self.tokenizer.decode(tokens)
        assistant_marker = "Assistant:"
        
        if assistant_marker in text_decoded:
            marker_tokens = self.tokenizer.encode(assistant_marker, add_special_tokens=False)
            
            for i in range(len(tokens) - len(marker_tokens)):
                match = True
                for j, marker_token in enumerate(marker_tokens):
                    if i + j >= len(tokens) or tokens[i + j] != marker_token:
                        match = False
                        break
                
                if match:
                    start_idx = i + len(marker_tokens)
                    if start_idx < len(labels):
                        labels[start_idx:] = target_ids[start_idx:]
                    break
        
        pad_length = self.max_length - 1 - len(input_ids)
        if pad_length > 0:
            input_ids = F.pad(input_ids, (0, pad_length), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, (0, pad_length), value=-100)
        
        return input_ids, labels

# ============================================
# CACHE SYSTEM
# ============================================
def load_dataset_with_cache(name, loader_func, max_samples, config):
    cache_file = os.path.join(config['cache_dir'], f"{name}_v2.pt")
    
    if os.path.exists(cache_file):
        print(f"  ✅ Cache: {name}")
        cached = torch.load(cache_file)
        print(f"     {len(cached['data']):,} samples")
        return cached['data']
    
    print(f"  ⏳ Chargement {name}...")
    data = loader_func(max_samples, config)
    
    if data:
        torch.save({'data': data}, cache_file)
        print(f"  💾 {len(data):,} samples")
    
    return data

# ============================================
# 🏆 TIER 1: ELITE DATASETS
# ============================================

def load_alpaca_original(max_samples, config):
    """Stanford Alpaca - Original 52k"""
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Alpaca-Original"):
            if len(data) >= max_samples:
                break
            
            instruction = item.get('instruction', '').strip()
            input_text = item.get('input', '').strip()
            output = item.get('output', '').strip()
            
            # Combine instruction + input si présent
            user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
            
            if instruction and output and is_high_quality(output, config):
                data.append({
                    'user': user_content,
                    'assistant': output,
                    'source': 'alpaca_original'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_orca_math(max_samples, config):
    """Orca Math - Mathematical reasoning"""
    try:
        dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Orca-Math"):
            if len(data) >= max_samples:
                break
            
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            
            if question and answer and is_high_quality(answer, config):
                data.append({
                    'user': question,
                    'assistant': answer,
                    'source': 'orca_math'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_metamath(max_samples, config):
    """MetaMath - Math problems"""
    try:
        dataset = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
        data = []
        
        for item in tqdm(dataset.take(max_samples * 3), desc="MetaMath", total=max_samples):
            if len(data) >= max_samples:
                break
            
            query = item.get('query', '').strip()
            response = item.get('response', '').strip()
            
            if query and response and is_high_quality(response, config):
                data.append({
                    'user': query,
                    'assistant': response,
                    'source': 'metamath'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_code_alpaca(max_samples, config):
    """Code Alpaca - Code instructions"""
    try:
        dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Code-Alpaca"):
            if len(data) >= max_samples:
                break
            
            instruction = item.get('instruction', '').strip()
            output = item.get('output', '').strip()
            input_text = item.get('input', '').strip()
            
            user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
            
            if instruction and output and is_high_quality(output, config):
                data.append({
                    'user': user_content,
                    'assistant': output,
                    'source': 'code_alpaca'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_evol_instruct(max_samples, config):
    """WizardLM Evol Instruct"""
    try:
        dataset = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Evol-Instruct"):
            if len(data) >= max_samples:
                break
            
            conversations = item.get('conversations', [])
            if len(conversations) >= 2:
                user_msg = conversations[0].get('value', '').strip()
                assistant_msg = conversations[1].get('value', '').strip()
                
                if user_msg and assistant_msg and is_high_quality(assistant_msg, config):
                    data.append({
                        'user': user_msg,
                        'assistant': assistant_msg,
                        'source': 'evol_instruct'
                    })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_platypus(max_samples, config):
    """Open Platypus - STEM"""
    try:
        dataset = load_dataset("garage-bAInd/Open-Platypus", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Platypus"):
            if len(data) >= max_samples:
                break
            
            instruction = item.get('instruction', '').strip()
            output = item.get('output', '').strip()
            input_text = item.get('input', '').strip()
            
            user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
            
            if instruction and output and is_high_quality(output, config):
                data.append({
                    'user': user_content,
                    'assistant': output,
                    'source': 'platypus'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_camel_ai(max_samples, config):
    """CAMEL AI - Physics conversations"""
    try:
        dataset = load_dataset("camel-ai/physics", split="train")
        data = []
        
        for item in tqdm(dataset, desc="CAMEL-AI"):
            if len(data) >= max_samples:
                break
            
            message_1 = item.get('message_1', '').strip()
            message_2 = item.get('message_2', '').strip()
            
            if message_1 and message_2 and is_high_quality(message_2, config):
                data.append({
                    'user': message_1,
                    'assistant': message_2,
                    'source': 'camel_ai'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_lmsys_chat(max_samples, config):
    """LMSYS Chat - Real conversations"""
    try:
        dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        data = []
        
        for item in tqdm(dataset.take(max_samples * 4), desc="LMSYS-Chat", total=max_samples):
            if len(data) >= max_samples:
                break
            
            conversation = item.get('conversation', [])
            if len(conversation) >= 2:
                user_msg = conversation[0].get('content', '').strip()
                assistant_msg = conversation[1].get('content', '').strip()
                
                if user_msg and assistant_msg and is_high_quality(assistant_msg, config):
                    data.append({
                        'user': user_msg,
                        'assistant': assistant_msg,
                        'source': 'lmsys_chat'
                    })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_logic(max_samples, config):
    """LogiCoT - Logical reasoning"""
    try:
        dataset = load_dataset("INK-USC/LogiCoT", split="train")
        data = []
        
        for item in tqdm(dataset, desc="LogiCoT"):
            if len(data) >= max_samples:
                break
            
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            
            if question and answer and is_high_quality(answer, config):
                data.append({
                    'user': question,
                    'assistant': answer,
                    'source': 'logic'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

# ============================================
# 🥇 TIER 2: PREMIUM DATASETS
# ============================================

def load_openorca(max_samples, config):
    """OpenOrca - Multi-task"""
    try:
        dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        data = []
        
        for item in tqdm(dataset.take(max_samples * 2), desc="OpenOrca", total=max_samples):
            if len(data) >= max_samples:
                break
            
            system_prompt = item.get('system_prompt', '').strip()
            question = item.get('question', '').strip()
            response = item.get('response', '').strip()
            
            if question and response and is_high_quality(response, config):
                data.append({
                    'user': question,
                    'assistant': response,
                    'system': system_prompt if system_prompt else None,
                    'source': 'openorca'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_ultrachat(max_samples, config):
    """UltraChat - Conversations"""
    try:
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        data = []
        
        for item in tqdm(dataset, desc="UltraChat"):
            if len(data) >= max_samples:
                break
            
            messages = item.get('messages', [])
            if len(messages) >= 2:
                user_msg = None
                assistant_msg = None
                
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_msg = msg.get('content', '').strip()
                    elif msg.get('role') == 'assistant' and user_msg:
                        assistant_msg = msg.get('content', '').strip()
                
                if user_msg and assistant_msg and is_high_quality(assistant_msg, config):
                    data.append({
                        'user': user_msg,
                        'assistant': assistant_msg,
                        'source': 'ultrachat'
                    })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_wizardlm(max_samples, config):
    """WizardLM - Evolved instructions"""
    try:
        dataset = load_dataset("cognitivecomputations/WizardLM_alpaca_evol_instruct_70k_unfiltered", split="train")
        data = []
        
        for item in tqdm(dataset, desc="WizardLM"):
            if len(data) >= max_samples:
                break
            
            instruction = item.get('instruction', '').strip()
            output = item.get('output', '').strip()
            
            if instruction and output and is_high_quality(output, config):
                data.append({
                    'user': instruction,
                    'assistant': output,
                    'source': 'wizardlm'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_nous_instruct(max_samples, config):
    """Nous Research Instruct"""
    try:
        dataset = load_dataset("teknium/GPT4-LLM-Cleaned", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Nous-Instruct"):
            if len(data) >= max_samples:
                break
            
            instruction = item.get('instruction', '').strip()
            output = item.get('output', '').strip()
            
            if instruction and output and is_high_quality(output, config):
                data.append({
                    'user': instruction,
                    'assistant': output,
                    'source': 'nous_instruct'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_openhermes(max_samples, config):
    """OpenHermes - Diverse tasks"""
    try:
        dataset = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
        data = []
        
        for item in tqdm(dataset.take(max_samples * 2), desc="OpenHermes", total=max_samples):
            if len(data) >= max_samples:
                break
            
            conversations = item.get('conversations', [])
            if len(conversations) >= 2:
                user_msg = None
                assistant_msg = None
                
                for conv in conversations:
                    role = conv.get('from', '')
                    if role in ['human', 'user']:
                        user_msg = conv.get('value', '').strip()
                    elif role in ['gpt', 'assistant'] and user_msg:
                        assistant_msg = conv.get('value', '').strip()
                
                if user_msg and assistant_msg and is_high_quality(assistant_msg, config):
                    data.append({
                        'user': user_msg,
                        'assistant': assistant_msg,
                        'source': 'openhermes'
                    })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_airoboros(max_samples, config):
    """Airoboros - Creative tasks"""
    try:
        dataset = load_dataset("jondurbin/airoboros-3.2", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Airoboros"):
            if len(data) >= max_samples:
                break
            
            conversations = item.get('conversations', [])
            if len(conversations) >= 2:
                user_msg = conversations[0].get('value', '').strip()
                assistant_msg = conversations[1].get('value', '').strip()
                
                if user_msg and assistant_msg and is_high_quality(assistant_msg, config):
                    data.append({
                        'user': user_msg,
                        'assistant': assistant_msg,
                        'source': 'airoboros'
                    })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

# ============================================
# 🥈 TIER 3: SOLID DATASETS
# ============================================

def load_alpaca_gpt4(max_samples, config):
    """Alpaca GPT-4"""
    try:
        dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Alpaca-GPT4"):
            if len(data) >= max_samples:
                break
            
            instruction = item.get('instruction', '').strip()
            output = item.get('output', '').strip()
            input_text = item.get('input', '').strip()
            
            user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
            
            if instruction and output and is_high_quality(output, config):
                data.append({
                    'user': user_content,
                    'assistant': output,
                    'source': 'alpaca_gpt4'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_oasst1(max_samples, config):
    """OpenAssistant"""
    try:
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
                    
                    if is_high_quality(assistant_text, config):
                        data.append({
                            'user': user_text,
                            'assistant': assistant_text,
                            'source': 'oasst1'
                        })
                        processed.add(msg['parent_id'])
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_flan(max_samples, config):
    """Google FLAN"""
    try:
        dataset = load_dataset("Muennighoff/flan", split="train", streaming=True)
        data = []
        
        for item in tqdm(dataset.take(max_samples * 3), desc="FLAN", total=max_samples):
            if len(data) >= max_samples:
                break
            
            inputs = item.get('inputs', '').strip()
            targets = item.get('targets', '').strip()
            
            if inputs and targets and is_high_quality(targets, config):
                data.append({
                    'user': inputs,
                    'assistant': targets,
                    'source': 'flan'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_dolly(max_samples, config):
    """Databricks Dolly"""
    try:
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Dolly"):
            if len(data) >= max_samples:
                break
            
            instruction = item.get('instruction', '').strip()
            context = item.get('context', '').strip()
            response = item.get('response', '').strip()
            
            user_content = f"{instruction}\n\nContext: {context}" if context else instruction
            
            if instruction and response and is_high_quality(response, config):
                data.append({
                    'user': user_content,
                    'assistant': response,
                    'source': 'dolly'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

# ============================================
# 🛡️ TIER 4: SAFETY DATASETS
# ============================================

def load_hh_rlhf(max_samples, config):
    """Anthropic HH-RLHF"""
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        data = []
        
        for item in tqdm(dataset, desc="HH-RLHF"):
            if len(data) >= max_samples:
                break
            
            chosen = item.get('chosen', '').strip()
            
            if not chosen:
                continue
            
            lines = chosen.split('\n\n')
            user_msg = None
            assistant_msg = None
            
            for line in lines:
                if line.startswith('Human:'):
                    user_msg = line.replace('Human:', '').strip()
                elif line.startswith('Assistant:') and user_msg:
                    assistant_msg = line.replace('Assistant:', '').strip()
            
            if user_msg and assistant_msg and is_high_quality(assistant_msg, config):
                data.append({
                    'user': user_msg,
                    'assistant': assistant_msg,
                    'source': 'hh_rlhf'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_prosocial(max_samples, config):
    """Prosocial Dialogue"""
    try:
        dataset = load_dataset("allenai/prosocial-dialog", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Prosocial"):
            if len(data) >= max_samples:
                break
            
            context = item.get('context', '').strip()
            response = item.get('response', '').strip()
            
            if context and response and is_high_quality(response, config):
                data.append({
                    'user': context,
                    'assistant': response,
                    'source': 'prosocial'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_safe_rlhf(max_samples, config):
    """Safe RLHF"""
    try:
        dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train", streaming=True)
        data = []
        
        for item in tqdm(dataset.take(max_samples * 3), desc="Safe-RLHF", total=max_samples):
            if len(data) >= max_samples:
                break
            
            prompt = item.get('prompt', '').strip()
            response_0 = item.get('response_0', '').strip()
            is_response_0_safe = item.get('is_response_0_safe', True)
            
            if prompt and response_0 and is_response_0_safe and is_high_quality(response_0, config):
                data.append({
                    'user': prompt,
                    'assistant': response_0,
                    'source': 'safe_rlhf'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []
def load_oasst2(max_samples, config):
    """OpenAssistant 2 - Version améliorée"""
    try:
        dataset = load_dataset("OpenAssistant/oasst2", split="train")
        
        # Construire le graph des messages
        messages_dict = {}
        for item in dataset:
            msg_id = item['message_id']
            messages_dict[msg_id] = item
        
        data = []
        processed = set()
        
        for msg_id, msg in tqdm(messages_dict.items(), desc="OASST2"):
            if len(data) >= max_samples:
                break
            
            # Cherche paires assistant/prompter de qualité
            if msg['role'] == 'assistant' and msg['parent_id'] and msg['parent_id'] not in processed:
                parent = messages_dict.get(msg['parent_id'])
                if parent and parent['role'] == 'prompter':
                    # ✅ Filtre par qualité (rank, lang)
                    if msg.get('lang') != 'en':
                        continue
                    if msg.get('rank', 0) is not None and msg.get('rank', 0) > 2:
                        continue
                    
                    user_text = parent['text'].strip()
                    assistant_text = msg['text'].strip()
                    
                    if is_high_quality(assistant_text, config):
                        data.append({
                            'user': user_text,
                            'assistant': assistant_text,
                            'source': 'oasst2'
                        })
                        processed.add(msg['parent_id'])
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_ultrafeedback(max_samples, config):
    """UltraFeedback - Responses with high ratings"""
    try:
        dataset = load_dataset("openbmb/UltraFeedback", split="train")
        data = []
        
        for item in tqdm(dataset, desc="UltraFeedback"):
            if len(data) >= max_samples:
                break
            
            instruction = item.get('instruction', '').strip()
            completions = item.get('completions', [])
            
            if not instruction or not completions:
                continue
            
            # Prend la meilleure réponse (highest rating)
            best_completion = max(completions, 
                                key=lambda x: x.get('overall_score', 0))
            
            response = best_completion.get('response', '').strip()
            score = best_completion.get('overall_score', 0)
            
            # ✅ Garde uniquement score >= 8
            if score >= 8 and response and is_high_quality(response, config):
                data.append({
                    'user': instruction,
                    'assistant': response,
                    'source': 'ultrafeedback'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_slimorca(max_samples, config):
    """SlimOrca - Deduped high-quality Orca"""
    try:
        dataset = load_dataset("Open-Orca/SlimOrca-Dedup", split="train", streaming=True)
        data = []
        
        for item in tqdm(dataset.take(max_samples * 2), desc="SlimOrca", total=max_samples):
            if len(data) >= max_samples:
                break
            
            conversations = item.get('conversations', [])
            if len(conversations) >= 2:
                if conversations[0].get('from') == 'human':
                    user_msg = conversations[0].get('value', '').strip()
                    assistant_msg = conversations[1].get('value', '').strip()
                    
                    if user_msg and assistant_msg and is_high_quality(assistant_msg, config):
                        data.append({
                            'user': user_msg,
                            'assistant': assistant_msg,
                            'source': 'slimorca'
                        })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_no_robots(max_samples, config):
    """No Robots - Human-written instructions"""
    try:
        dataset = load_dataset("HuggingFaceH4/no_robots", split="train")
        data = []
        
        for item in tqdm(dataset, desc="NoRobots"):
            if len(data) >= max_samples:
                break
            
            messages = item.get('messages', [])
            if len(messages) >= 2:
                user_msg = messages[0].get('content', '').strip()
                assistant_msg = messages[1].get('content', '').strip()
                
                if user_msg and assistant_msg and is_high_quality(assistant_msg, config):
                    data.append({
                        'user': user_msg,
                        'assistant': assistant_msg,
                        'source': 'no_robots'
                    })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

# ============================================
# 🥇 TIER 2: SPECIALIZED HIGH-QUALITY (+80k)
# ============================================

def load_orca_dpo(max_samples, config):
    """Orca DPO Pairs - Take chosen responses"""
    try:
        dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Orca-DPO"):
            if len(data) >= max_samples:
                break
            
            system = item.get('system', '').strip()
            question = item.get('question', '').strip()
            chosen = item.get('chosen', '').strip()
            
            if question and chosen and is_high_quality(chosen, config):
                data.append({
                    'user': question,
                    'assistant': chosen,
                    'system': system if system else None,
                    'source': 'orca_dpo'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_capybara(max_samples, config):
    """Capybara - Multi-turn conversations"""
    try:
        dataset = load_dataset("LDJnr/Capybara", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Capybara"):
            if len(data) >= max_samples:
                break
            
            conversation = item.get('conversation', [])
            # Prend première paire user/assistant
            if len(conversation) >= 2:
                user_msg = None
                assistant_msg = None
                
                for msg in conversation:
                    if msg.get('input'):
                        user_msg = msg['input'].strip()
                    elif msg.get('output') and user_msg:
                        assistant_msg = msg['output'].strip()
                        break
                
                if user_msg and assistant_msg and is_high_quality(assistant_msg, config):
                    data.append({
                        'user': user_msg,
                        'assistant': assistant_msg,
                        'source': 'capybara'
                    })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_puffin(max_samples, config):
    """Puffin - Reasoning dataset"""
    try:
        dataset = load_dataset("LDJnr/Puffin", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Puffin"):
            if len(data) >= max_samples:
                break
            
            conversations = item.get('conversations', [])
            if len(conversations) >= 2:
                user_msg = conversations[0].get('value', '').strip()
                assistant_msg = conversations[1].get('value', '').strip()
                
                if user_msg and assistant_msg and is_high_quality(assistant_msg, config):
                    data.append({
                        'user': user_msg,
                        'assistant': assistant_msg,
                        'source': 'puffin'
                    })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_know_saqa(max_samples, config):
    """KnowSaqa - Medical Q&A"""
    try:
        dataset = load_dataset("medalpaca/medical_meadow_wikidoc", split="train")
        data = []
        
        for item in tqdm(dataset, desc="MedicalMeadow"):
            if len(data) >= max_samples:
                break
            
            instruction = item.get('instruction', '').strip()
            output = item.get('output', '').strip()
            
            if instruction and output and is_high_quality(output, config):
                data.append({
                    'user': instruction,
                    'assistant': output,
                    'source': 'medical_meadow'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_tulu_v2(max_samples, config):
    """Tulu V2 Mix - AllenAI instruction mix"""
    try:
        dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train", streaming=True)
        data = []
        
        for item in tqdm(dataset.take(max_samples * 2), desc="Tulu-V2", total=max_samples):
            if len(data) >= max_samples:
                break
            
            messages = item.get('messages', [])
            if len(messages) >= 2:
                if messages[0].get('role') == 'user':
                    user_msg = messages[0].get('content', '').strip()
                    assistant_msg = messages[1].get('content', '').strip()
                    
                    if user_msg and assistant_msg and is_high_quality(assistant_msg, config):
                        data.append({
                            'user': user_msg,
                            'assistant': assistant_msg,
                            'source': 'tulu_v2'
                        })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

# ============================================
# 🥈 TIER 3: DOMAIN-SPECIFIC (+50k)
# ============================================

def load_code_contests(max_samples, config):
    """Code Contests - Competitive programming"""
    try:
        dataset = load_dataset("deepmind/code_contests", split="train", streaming=True)
        data = []
        
        for item in tqdm(dataset.take(max_samples * 3), desc="CodeContests", total=max_samples):
            if len(data) >= max_samples:
                break
            
            description = item.get('description', '').strip()
            solutions = item.get('solutions', {})
            
            if not description or not solutions:
                continue
            
            python_sols = solutions.get('language', [])
            if 3 in python_sols:  # Python language code
                idx = python_sols.index(3)
                solution = solutions.get('solution', [])[idx]
                
                if solution and is_high_quality(solution, config):
                    data.append({
                        'user': f"Solve this coding problem:\n\n{description}",
                        'assistant': f"```python\n{solution}\n```",
                        'source': 'code_contests'
                    })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_sciq(max_samples, config):
    """SciQ - Science questions"""
    try:
        dataset = load_dataset("allenai/sciq", split="train")
        data = []
        
        for item in tqdm(dataset, desc="SciQ"):
            if len(data) >= max_samples:
                break
            
            question = item.get('question', '').strip()
            support = item.get('support', '').strip()
            correct = item.get('correct_answer', '').strip()
            
            if question and correct:
                instruction = f"{question}\n\nContext: {support}" if support else question
                
                if is_high_quality(correct, config):
                    data.append({
                        'user': instruction,
                        'assistant': correct,
                        'source': 'sciq'
                    })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def load_glaive_code(max_samples, config):
    """Glaive Code Assistant - Code help"""
    try:
        dataset = load_dataset("glaiveai/glaive-code-assistant", split="train")
        data = []
        
        for item in tqdm(dataset, desc="Glaive-Code"):
            if len(data) >= max_samples:
                break
            
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            
            if question and answer and is_high_quality(answer, config):
                data.append({
                    'user': question,
                    'assistant': answer,
                    'source': 'glaive_code'
                })
        
        random.shuffle(data)
        return data
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

# ============================================
# CHARGEMENT DE TOUS LES DATASETS
# ============================================
print("\n" + "="*80)
print("📥 CHARGEMENT DATASETS PREMIUM")
print("="*80)

ALL_LOADERS = [
    # 🏆 TIER 1
    ('orca_math', load_orca_math, CONFIG['orca_math_samples'], "🏆"),
    ('metamath', load_metamath, CONFIG['metamath_samples'], "🏆"),
    ('code_alpaca', load_code_alpaca, CONFIG['code_alpaca_samples'], "🏆"),
    ('evol_instruct', load_evol_instruct, CONFIG['evol_instruct_samples'], "🏆"),
    ('platypus', load_platypus, CONFIG['platypus_samples'], "🏆"),
    #('lmsys_chat', load_lmsys_chat, CONFIG['lmsys_chat_samples'], "🏆"),
    
    # 🥇 TIER 2
    ('openorca', load_openorca, CONFIG['openorca_samples'], "🥇"),
    ('ultrachat', load_ultrachat, CONFIG['ultrachat_samples'], "🥇"),
    ('wizardlm', load_wizardlm, CONFIG['wizardlm_samples'], "🥇"),
    ('nous_instruct', load_nous_instruct, CONFIG['nous_instruct_samples'], "🥇"),
    ('openhermes', load_openhermes, CONFIG['openhermes_samples'], "🥇"),
    ('airoboros', load_airoboros, CONFIG['airoboros_samples'], "🥇"),
    
    # 🥈 TIER 3
    # Dans ALL_LOADERS, ajoute dans 🥈 TIER 3 (avant alpaca_gpt4) :
    ('alpaca_original', load_alpaca_original, CONFIG['alpaca_samples'], "🥈"),
    ('alpaca_gpt4', load_alpaca_gpt4, CONFIG['alpaca_gpt4_samples'], "🥈"),
    ('oasst1', load_oasst1, CONFIG['oasst1_samples'], "🥈"),
    ('flan', load_flan, CONFIG['flan_samples'], "🥈"),
    ('dolly', load_dolly, CONFIG['dolly_samples'], "🥈"),
    
    # 🛡️ TIER 4
    ('hh_rlhf', load_hh_rlhf, CONFIG['hh_rlhf_samples'], "🛡️"),
    ('prosocial', load_prosocial, CONFIG['prosocial_samples'], "🛡️"),
    ('safe_rlhf', load_safe_rlhf, CONFIG['safe_rlhf_samples'], "🛡️"),
    ('oasst2', load_oasst2, CONFIG['oasst2_samples'], "🏆"),
    ('ultrafeedback', load_ultrafeedback, CONFIG['ultrafeedback_samples'], "🏆"),
    ('slimorca', load_slimorca, CONFIG['slimorca_samples'], "🏆"),
    ('no_robots', load_no_robots, CONFIG['no_robots_samples'], "🏆"),
    
    # 🥇 TIER 2 (après airoboros)
    ('orca_dpo', load_orca_dpo, CONFIG['orca_dpo_samples'], "🥇"),
    ('capybara', load_capybara, CONFIG['capybara_samples'], "🥇"),
    ('puffin', load_puffin, CONFIG['puffin_samples'], "🥇"),
    #('medical_meadow', load_know_saqa, CONFIG['medical_meadow_samples'], "🥇"),
    ('tulu_v2', load_tulu_v2, CONFIG['tulu_v2_samples'], "🥇"),
    
    # 🥈 TIER 3 (après dolly)
    ('code_contests', load_code_contests, CONFIG['code_contests_samples'], "🥈"),
    ('sciq', load_sciq, CONFIG['sciq_samples'], "🥈"),
    ('glaive_code', load_glaive_code, CONFIG['glaive_code_samples'], "🥈"),
]

all_data = []
dataset_stats = {}

for name, loader, max_samples, tier in ALL_LOADERS:
    print(f"\n{tier} {name.upper()}")
    data = load_dataset_with_cache(name, loader, max_samples, CONFIG)
    
    if data:
        all_data.extend(data)
        dataset_stats[name] = len(data)
    else:
        dataset_stats[name] = 0

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
                user_text = obj.get('user', '').strip()
                assistant_text = obj.get('assistant', '').strip()
                
                if user_text and assistant_text and is_high_quality(assistant_text, CONFIG):
                    synthetic_data.append({
                        'user': user_text,
                        'assistant': assistant_text,
                        'source': 'synthetic'
                    })
            except:
                continue
    
    all_data.extend(synthetic_data)
    dataset_stats['synthetic'] = len(synthetic_data)
    print(f"  ✅ {len(synthetic_data):,} samples")

# ============================================
# STATS & SPLIT
# ============================================
print("\n" + "="*80)
print("📊 STATISTIQUES FINALES")
print("="*80)

total = len(all_data)
print(f"\nTotal samples: {total:,}")

for tier_emoji in ["🏆", "🥇", "🥈", "🛡️"]:
    tier_name = {"🏆": "ELITE", "🥇": "PREMIUM", "🥈": "SOLID", "🛡️": "SAFETY"}[tier_emoji]
    print(f"\n{tier_emoji} TIER {tier_name}:")
    tier_total = 0
    
    for name, _, _, tier in ALL_LOADERS:
        if tier == tier_emoji and name in dataset_stats:
            count = dataset_stats[name]
            tier_total += count
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {name:18s} {count:>7,} ({pct:>5.1f}%)")
    
    print(f"  {'─'*50}\n  Subtotal: {tier_total:,}")

# Split stratifié
print("\n📊 Création train/val split...")

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

# DataLoaders
train_dataset = ConversationDataset(train_data, tokenizer, CONFIG['max_seq_len'])
val_dataset = ConversationDataset(val_data, tokenizer, CONFIG['max_seq_len'])

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

# Charge pre-training
if os.path.exists(CONFIG['pretrain_checkpoint']):
    print(f"\n📂 Pre-training: {CONFIG['pretrain_checkpoint']}")
    checkpoint = torch.load(CONFIG['pretrain_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Chargé!")
else:
    print(f"\n⚠️  Pre-training non trouvé")

# ============================================
# OPTIMIZER & SCHEDULER
# ============================================
total_steps = (len(train_loader) * CONFIG['num_epochs']) // CONFIG['gradient_accumulation']
warmup_steps = int(CONFIG['warmup_ratio'] * total_steps)

print(f"\n📊 Training:")
print(f"   Epochs:  {CONFIG['num_epochs']}")
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
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================
# VALIDATION FUNCTION
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
# TRAINING LOOP
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

# Resume si existe
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
# RÉSUMÉ FINAL
# ============================================
total_time = (time.time() - start_time) / 3600

print("\n" + "="*80)
print("🎉 TRAINING TERMINÉ!")
print("="*80)

print(f"\n📊 RÉSULTATS:")
print(f"   Best Val Loss: {best_val_loss:.4f}")
print(f"   Temps total:   {total_time:.2f}h")
print(f"   Total samples: {len(all_data):,}")
print(f"   Steps:         {global_step:,}")

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
print("\n1. 🧪 Tester le modèle:")
print("   python test_hessgpt.py --checkpoint checkpoints/sft_premium/best.pt")

print("\n2. 🎮 Mode interactif:")
print("   python inference.py --checkpoint checkpoints/sft_premium/best.pt --interactive")

print("\n3. 📝 Format attendu:")
print("   User: Bonjour!")
print("   Assistant: Bonjour! Comment puis-je vous aider aujourd'hui?")

print("\n4. 📊 Analyser l'historique:")
print(f"   cat {history_file}")

print("\n" + "="*80)
print("🎯 QUALITÉ DES DATASETS")
print("="*80)

print("\n🏆 Elite Quality (200k samples):")
print("   - Orca-Math: Raisonnement mathématique avancé")
print("   - MetaMath: Problèmes mathématiques variés")
print("   - Code Alpaca: Instructions de code haute qualité")
print("   - Evol Instruct: Instructions complexes évoluées")
print("   - Platypus: Focus STEM et sciences")
print("   - CAMEL-AI: Conversations multi-domaines")
print("   - LMSYS Chat: Conversations réelles d'utilisateurs")
print("   - LogiCoT: Raisonnement logique structuré")

print("\n🥇 Premium Quality (150k samples):")
print("   - OpenOrca: Multi-task instructions")
print("   - UltraChat: Conversations naturelles longues")
print("   - WizardLM: Instructions évoluées complexes")
print("   - Nous Instruct: Instructions curées GPT-4")
print("   - OpenHermes: Tâches diverses haute qualité")
print("   - Airoboros: Tâches créatives et roleplay")

print("\n🥈 Solid Quality (100k samples):")
print("   - Alpaca GPT-4: Instructions générées par GPT-4")
print("   - ShareGPT: Vraies conversations utilisateurs")
print("   - OASST1: Dialogues communauté OpenAssistant")
print("   - FLAN: Google FLAN instructions variées")
print("   - Dolly: Databricks instructions métier")

print("\n🛡️ Safety & Ethics (30k samples):")
print("   - HH-RLHF: Anthropic Helpful & Harmless")
print("   - Prosocial: Dialogues prosociaux")
print("   - Safe RLHF: Alignement sécurité")

print("\n" + "="*80)
print("⚙️ OPTIMISATIONS APPLIQUÉES")
print("="*80)

print("\n✅ Architecture:")
print("   - Gradient accumulation: 4 steps")
print("   - Mixed precision (FP16)")
print("   - Gradient clipping: 0.5")
print("   - AdamW optimizer (fused)")
print("   - Cosine LR schedule avec warmup")

print("\n✅ Qualité:")
print("   - Filtrage strict des réponses")
print("   - Score qualité minimum: 0.7")
print("   - Diversité lexicale: 65% minimum")
print("   - Anti-répétition: 12% maximum")
print("   - Longueur optimale: 30-512 tokens")

print("\n✅ Training:")
print("   - 1 epoch optimal (évite overfitting)")
print("   - Learning rate: 1e-5 (10x moins que pretrain)")
print("   - Warmup: 3% des steps")
print("   - Validation tous les 1000 steps")
print("   - Early stopping avec patience=3")

print("\n✅ Data:")
print("   - 480k+ samples haute qualité")
print("   - Split stratifié par source")
print("   - Cache pour chargement rapide")
print("   - Format simple User/Assistant")
print("   - Loss masking (uniquement sur Assistant)")

print("\n" + "="*80)
print("📈 MÉTRIQUES À SURVEILLER")
print("="*80)

print("\n1. Validation Loss:")
print("   - Doit diminuer régulièrement")
print("   - Target: < 2.0 pour bonne qualité")
print("   - Si augmente: overfitting possible")

print("\n2. Perplexity:")
print("   - Mesure la \"surprise\" du modèle")
print("   - Target: < 10 pour conversations naturelles")
print("   - Plus bas = meilleures prédictions")

print("\n3. Learning Rate:")
print("   - Commence à 1e-5")
print("   - Warmup durant 3% des steps")
print("   - Cosine decay vers 1e-6")

print("\n4. Gradient Norm:")
print("   - Clipping à 0.5")
print("   - Si souvent clippé: learning rate trop élevé")
print("   - Si jamais clippé: peut augmenter LR")

print("\n" + "="*80)
print("🔧 TROUBLESHOOTING")
print("="*80)

print("\n❌ Out of Memory (OOM):")
print("   - Réduire batch_size (16 → 8 → 4)")
print("   - Réduire max_seq_len (512 → 384 → 256)")
print("   - Augmenter gradient_accumulation")

print("\n❌ Loss NaN:")
print("   - Réduire learning rate (1e-5 → 5e-6)")
print("   - Vérifier gradient clipping activé")
print("   - Vérifier données pas corrompues")

print("\n❌ Pas d'amélioration:")
print("   - Augmenter learning rate (1e-5 → 2e-5)")
print("   - Vérifier pre-training chargé correctement")
print("   - Augmenter warmup_ratio")

print("\n❌ Overfitting rapide:")
print("   - Déjà avec 1 epoch, difficile à overfit")
print("   - Augmenter dropout si nécessaire")
print("   - Ajouter plus de données diverses")

print("\n❌ Validation lente:")
print("   - Réduire val_split (3% → 1%)")
print("   - Augmenter validate_every (1000 → 2000)")
print("   - Utiliser moins de workers DataLoader")

print("\n" + "="*80)
print("🚀 AMÉLIORATIONS FUTURES")
print("="*80)

print("\n1. Augmenter capacité:")
print("   - Passer à 2-3 epochs si pas d'overfitting")
print("   - Augmenter batch size si VRAM disponible")
print("   - Tester learning rates plus élevés")

print("\n2. Ajouter datasets:")
print("   - SlimOrca pour plus de diversité")
print("   - Code contests pour coding avancé")
print("   - Math reasoning datasets supplémentaires")

print("\n3. Techniques avancées:")
print("   - LoRA pour fine-tuning efficace")
print("   - QLoRA pour réduire VRAM")
print("   - DPO/RLHF pour alignment")

print("\n4. Évaluation:")
print("   - MT-Bench pour qualité conversations")
print("   - HumanEval pour capacités code")
print("   - MMLU pour connaissances générales")
print("   - TruthfulQA pour véracité")

print("\n" + "="*80)
print("📚 RESSOURCES")
print("="*80)

print("\n📖 Documentation:")
print("   - Transformers: https://huggingface.co/docs/transformers")
print("   - Datasets: https://huggingface.co/docs/datasets")
print("   - PyTorch: https://pytorch.org/docs")

print("\n🤖 Modèles similaires:")
print("   - Llama 2: Meta's instruction-tuned model")
print("   - Mistral: Efficient 7B model")
print("   - Zephyr: DPO-tuned Mistral")
print("   - Nous Hermes: High-quality instruction model")

print("\n📊 Benchmarks:")
print("   - Open LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard")
print("   - AlpacaEval: https://tatsu-lab.github.io/alpaca_eval")
print("   - MT-Bench: https://github.com/lm-sys/FastChat")

print("\n💬 Communauté:")
print("   - HuggingFace Discord")
print("   - r/LocalLLaMA")
print("   - EleutherAI Discord")

print("\n" + "="*80)
print("🎊 FIN DU SCRIPT")
print("="*80)

print("\n📊 Statistiques finales:")
for tier_emoji in ["🏆", "🥇", "🥈", "🛡️"]:
    tier_total = sum(dataset_stats.get(name, 0) for name, _, _, tier in ALL_LOADERS if tier == tier_emoji)
    tier_pct = (tier_total / total * 100) if total > 0 else 0
    tier_name = {"🏆": "Elite", "🥇": "Premium", "🥈": "Solid", "🛡️": "Safety"}[tier_emoji]
    print(f"   {tier_emoji} {tier_name:10s}: {tier_total:>7,} samples ({tier_pct:>5.1f}%)")

print(f"\n   {'─'*50}")
print(f"   {'TOTAL':14s}: {total:>7,} samples")

print("\n✨ Le modèle est prêt pour l'inférence!")
print("   Checkpoint: checkpoints/sft_premium/best.pt")
print("\n🎯 Bonne utilisation de HessGPT SFT Premium!")
print("="*80)