#!/usr/bin/env python3
"""
🚀 HessGPT V5 - FIXED VERSION
✅ Datasets corrigés (pas de doublons)
✅ Gestion mémoire améliorée
✅ Retry logic robuste
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
import time
import math
import json
import gc
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer
from datetime import datetime
import traceback

sys.path.append('./Core/Model')

print("="*80)
print("🚀 HessGPT V5 - FIXED VERSION")
print("="*80)

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    'vocab_size': 50257,
    'embed_dim': 1280,
    'num_heads': 20,
    'num_layers': 20,
    'max_seq_len': 1024,
    'dropout': 0.1,
    
    'batch_size': 32,
    'num_epochs_per_dataset': 2,
    'learning_rate': 3e-4,
    'warmup_steps': 2000,
    'gradient_accumulation': 4,
    'max_grad_norm': 1.0,
    
    'tokens_per_dataset': 1_000_000_000,
    'min_text_length': 200,
    'max_text_length': 5000,
    
    'val_tokens': 10_000_000,
    'validate_every': 1000,
    
    'scheduler_reset_per_dataset': True,
    'scheduler_type': 'cosine',
    'min_lr_ratio': 0.1,
    
    'use_compile': True,
    'compile_mode': 'default',
    'checkpoint_file': './checkpoints/HessGpt_V5.pt',
    
    'max_retries': 3,
    'skip_on_error': True,
}

# ============================================
# DATASETS CORRIGÉS - 5 datasets, 10B tokens
# ============================================
DATASETS_SEQUENCE = [
    {
        'name': 'fineweb_edu',
        'source': 'HuggingFaceFW/fineweb-edu',
        'config': 'sample-10BT',
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '🎓 Educational web content',
        'tokens': 1_000_000_000,
    },
    {
        'name': 'wikipedia',
        'source': 'wikipedia',
        'config': '20220301.en',
        'split': 'train',
        'text_key': 'text',
        'streaming': False,
        'description': '📚 Encyclopedia',
        'tokens': 1_000_000_000,
    },
    {
        'name': 'c4',
        'source': 'allenai/c4',
        'config': 'en',
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '🌐 Common Crawl',
        'tokens': 1_000_000_000,
    },
    {
        'name': 'openwebtext',
        'source': 'openwebtext',
        'config': None,
        'split': 'train',
        'text_key': 'text',
        'streaming': False,
        'description': '🌍 Reddit content',
        'tokens': 1_000_000_000,
    },
    {
        'name': 'pile_books',
        'source': 'monology/pile-uncopyrighted',
        'config': 'all',
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '📖 Books from Pile',
        'tokens': 1_000_000_000,
    },
]

total_tokens = sum(ds['tokens'] for ds in DATASETS_SEQUENCE)
total_seen = total_tokens * 2

print(f"\n📊 CONFIGURATION")
print(f"   Datasets: {len(DATASETS_SEQUENCE)}")
print(f"   Input tokens: {total_tokens/1e9:.1f}B")
print(f"   Tokens seen: {total_seen/1e9:.1f}B (×2 epochs)")
print(f"\n📚 DATASETS:")
for i, ds in enumerate(DATASETS_SEQUENCE, 1):
    print(f"   {i}. {ds['name']:18} - {ds['description']}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n✅ Device: {device}")

if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ============================================
# CLASSES INCHANGÉES (CheckpointManager, etc.)
# ============================================
class CheckpointManager:
    def __init__(self, checkpoint_path):
        self.path = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    def save(self, model, optimizer, scheduler, metadata):
        save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        checkpoint = {
            'model_state_dict': save_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': metadata['global_step'],
            'current_dataset_idx': metadata['current_dataset_idx'],
            'current_epoch': metadata['current_epoch'],
            'current_batch': metadata.get('current_batch', 0),
            'training_history': metadata['training_history'],
            'config': CONFIG,
            'last_save_time': datetime.now().isoformat(),
            'total_training_time': metadata.get('total_training_time', 0),
        }
        
        tmp_path = self.path + '.tmp'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, self.path)
        
        return checkpoint
    
    def load(self):
        if not os.path.exists(self.path):
            return None
        
        print(f"\n📂 Chargement checkpoint: {self.path}")
        checkpoint = torch.load(self.path, map_location='cpu')
        
        print(f"   ✅ Step: {checkpoint['global_step']:,}")
        print(f"   ✅ Dataset: {checkpoint['current_dataset_idx']}/{len(DATASETS_SEQUENCE)}")
        print(f"   ✅ Epoch: {checkpoint['current_epoch']}")
        
        return checkpoint
    
    def exists(self):
        return os.path.exists(self.path)

def tokenize_texts(texts, tokenizer):
    tokens = []
    filtered = 0
    
    for text in texts:
        if len(text) < CONFIG['min_text_length']:
            filtered += 1
            continue
        
        alpha_ratio = sum(c.isalnum() or c.isspace() for c in text[:1000]) / min(len(text), 1000)
        if alpha_ratio < 0.7:
            filtered += 1
            continue
        
        encoded = tokenizer.encode(
            text[:CONFIG['max_text_length']],
            add_special_tokens=False
        )
        tokens.extend(encoded)
        tokens.append(tokenizer.eos_token_id)
    
    return tokens, filtered

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.num_samples = len(tokens) // (seq_len + 1)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        return chunk[:-1], chunk[1:]

@torch.no_grad()
def calculate_perplexity(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            _, loss = model(x, targets=y)
        
        mask = (y != tokenizer.pad_token_id)
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 10))
    
    model.train()
    return perplexity, avg_loss

# ============================================
# LOAD DATASET - AVEC MEILLEURES ERREURS
# ============================================
def load_single_dataset(dataset_info):
    print(f"\n{'='*80}")
    print(f"📥 DATASET: {dataset_info['name'].upper()}")
    print(f"   {dataset_info['description']}")
    print(f"   Target: {dataset_info['tokens']/1e9:.1f}B tokens")
    print(f"{'='*80}")
    
    cache_dir = "data/cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{dataset_info['name']}_{dataset_info['tokens']//1e9:.0f}B_cache.pt"
    
    # Check cache
    if os.path.exists(cache_file):
        print(f"✅ Cache trouvé: {cache_file}")
        try:
            print(f"⏳ Chargement du cache...")
            cached = torch.load(cache_file)
            print(f"✅ Cache chargé: {len(cached['train_tokens'])/1e6:.1f}M train tokens")
            return cached['train_tokens'], cached['val_tokens']
        except Exception as e:
            print(f"⚠️  Erreur chargement cache: {e}")
            print(f"🔄 Suppression et re-téléchargement...")
            os.remove(cache_file)
    
    # Download avec retries
    for attempt in range(CONFIG['max_retries']):
        try:
            print(f"⏳ Téléchargement (tentative {attempt+1}/{CONFIG['max_retries']})...")
            start_time = time.time()
            
            if dataset_info['streaming']:
                dataset = load_dataset(
                    dataset_info['source'],
                    name=dataset_info['config'],
                    split=dataset_info['split'],
                    streaming=True,
                    trust_remote_code=False  # ✅ Plus sécurisé
                )
            else:
                dataset = load_dataset(
                    dataset_info['source'],
                    dataset_info['config'],
                    split=dataset_info['split'],
                    trust_remote_code=False
                )
            
            # Tokenize
            all_tokens = []
            batch_texts = []
            doc_count = 0
            total_filtered = 0
            
            max_tokens = dataset_info['tokens']
            estimated_docs = int(max_tokens / 500)
            
            pbar = tqdm(
                dataset if hasattr(dataset, '__iter__') else iter(dataset),
                total=estimated_docs,
                desc=f"📥 {dataset_info['name']}"
            )
            
            for item in pbar:
                if len(all_tokens) >= max_tokens:
                    break
                
                text = item.get(dataset_info['text_key'], '')
                
                if not text or not isinstance(text, str):
                    continue
                
                batch_texts.append(text)
                doc_count += 1
                
                if len(batch_texts) >= 1000:
                    tokens, filtered = tokenize_texts(batch_texts, tokenizer)
                    all_tokens.extend(tokens)
                    total_filtered += filtered
                    batch_texts = []
                    
                    pbar.set_postfix({
                        'docs': f'{doc_count:,}',
                        'tokens': f'{len(all_tokens)/1e6:.1f}M',
                        'progress': f'{len(all_tokens)/max_tokens*100:.1f}%'
                    })
            
            if batch_texts:
                tokens, filtered = tokenize_texts(batch_texts, tokenizer)
                all_tokens.extend(tokens)
                total_filtered += filtered
            
            if len(all_tokens) > max_tokens:
                all_tokens = all_tokens[:max_tokens]
            
            elapsed = time.time() - start_time
            
            print(f"\n✅ Téléchargement réussi!")
            print(f"   Documents: {doc_count:,}")
            print(f"   Tokens: {len(all_tokens)/1e9:.2f}B")
            print(f"   Filtered: {total_filtered:,}")
            print(f"   Temps: {elapsed/60:.1f} min")
            
            # Split train/val
            val_size = CONFIG['val_tokens']
            val_tokens = all_tokens[:val_size]
            train_tokens = all_tokens[val_size:]
            
            print(f"💾 Sauvegarde cache...")
            torch.save({
                'train_tokens': train_tokens,
                'val_tokens': val_tokens,
                'dataset_info': dataset_info,
            }, cache_file)
            print(f"✅ Cache sauvegardé: {cache_file}")
            
            return train_tokens, val_tokens
            
        except Exception as e:
            print(f"❌ Erreur tentative {attempt+1}: {str(e)[:200]}")
            
            if attempt < CONFIG['max_retries'] - 1:
                wait_time = 30 * (attempt + 1)
                print(f"⏳ Attente {wait_time}s avant retry...")
                time.sleep(wait_time)
            else:
                print(f"❌ Échec après {CONFIG['max_retries']} tentatives")
                print(traceback.format_exc())
                if CONFIG['skip_on_error']:
                    print(f"⏭️  Skip dataset {dataset_info['name']}")
                    return None, None
                else:
                    raise

# ============================================
# TRAINING - AVEC CLEANUP MÉMOIRE
# ============================================
def train_on_dataset(
    model, train_tokens, val_tokens, optimizer, scheduler,
    dataset_info, dataset_idx, checkpoint_manager,
    training_history, resume_epoch=0, resume_batch=0,
    global_step=0, total_training_time=0
):
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING: {dataset_info['name'].upper()}")
    print(f"{'='*80}")
    
    train_dataset = TokenDataset(train_tokens, CONFIG['max_seq_len'])
    val_dataset = TokenDataset(val_tokens, CONFIG['max_seq_len'])
    
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
        num_workers=2,
        pin_memory=True,
    )
    
    num_batches = len(train_loader)
    print(f"📊 {num_batches:,} batches × {CONFIG['num_epochs_per_dataset']} epochs")
    
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    start_time = time.time()
    
    for epoch in range(resume_epoch, CONFIG['num_epochs_per_dataset']):
        epoch_loss = 0
        valid_batches = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"[{dataset_info['name']}] Epoch {epoch+1}/{CONFIG['num_epochs_per_dataset']}"
        )
        
        for batch_idx, (x, y) in enumerate(pbar):
            if epoch == resume_epoch and batch_idx < resume_batch:
                continue
            
            try:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    logits, loss = model(x, targets=y)
                    loss = loss / CONFIG['gradient_accumulation']
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️  NaN/Inf au batch {batch_idx}")
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
                        val_ppl, val_loss = calculate_perplexity(model, val_loader, device)
                        
                        print(f"\n{'─'*80}")
                        print(f"📊 VALIDATION - Step {global_step:,}")
                        print(f"   Perplexity: {val_ppl:7.2f}")
                        print(f"   Loss:       {val_loss:7.4f}")
                        print(f"{'─'*80}\n")
                        
                        training_history['validations'].append({
                            'step': global_step,
                            'dataset': dataset_info['name'],
                            'epoch': epoch + 1,
                            'perplexity': val_ppl,
                            'val_loss': val_loss,
                            'train_loss': loss.item() * CONFIG['gradient_accumulation'],
                            'lr': scheduler.get_last_lr()[0],
                        })
                
                epoch_loss += loss.item() * CONFIG['gradient_accumulation']
                valid_batches += 1
                
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item() * CONFIG["gradient_accumulation"]:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    })
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n❌ CUDA OOM au batch {batch_idx}!")
                    print(f"🧹 Cleanup mémoire...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    continue
                else:
                    raise
        
        # Fin epoch
        avg_loss = epoch_loss / max(valid_batches, 1)
        val_ppl, val_loss = calculate_perplexity(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        total_training_time += epoch_time
        
        print(f"\n{'='*80}")
        print(f"✅ EPOCH {epoch+1} TERMINÉE")
        print(f"   Train Loss: {avg_loss:.4f}")
        print(f"   Val PPL:    {val_ppl:.2f}")
        print(f"   Temps:      {epoch_time/60:.1f} min")
        print(f"{'='*80}")
        
        training_history['epochs'].append({
            'dataset': dataset_info['name'],
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'global_step': global_step,
        })
        
        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata={
                'global_step': global_step,
                'current_dataset_idx': dataset_idx,
                'current_epoch': epoch + 1,
                'current_batch': 0,
                'training_history': training_history,
                'total_training_time': total_training_time,
            }
        )
        
        start_time = time.time()
    
    return global_step, total_training_time

# ============================================
# MAIN
# ============================================
def main():
    from HessGpt import HessGPT
    
    print("\n" + "="*80)
    print("🚀 HessGPT V5 - FIXED VERSION")
    print("="*80)
    
    if device == 'cpu':
        print("\n❌ GPU requise!")
        return
    
    checkpoint_manager = CheckpointManager(CONFIG['checkpoint_file'])
    
    print("\n🤖 Création modèle 500M...")
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
    
    if CONFIG['use_compile']:
        print("⚡ Compilation...")
        try:
            model = torch.compile(model, mode=CONFIG['compile_mode'])
            print("✅ Compilé")
        except Exception as e:
            print(f"⚠️  Compilation échouée: {e}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True
    )
    
    def create_scheduler(optimizer, dataset_tokens):
        steps = (dataset_tokens // (CONFIG['batch_size'] * CONFIG['max_seq_len'] * CONFIG['gradient_accumulation'])) * CONFIG['num_epochs_per_dataset']
        
        def lr_lambda(step):
            if step < CONFIG['warmup_steps']:
                return step / CONFIG['warmup_steps']
            progress = (step - CONFIG['warmup_steps']) / max(steps - CONFIG['warmup_steps'], 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return CONFIG['min_lr_ratio'] + (1.0 - CONFIG['min_lr_ratio']) * cosine
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = create_scheduler(optimizer, DATASETS_SEQUENCE[0]['tokens'])
    
    training_history = {
        'config': CONFIG,
        'datasets_completed': [],
        'epochs': [],
        'validations': [],
        'start_time': datetime.now().isoformat(),
    }
    
    global_step = 0
    start_dataset_idx = 0
    resume_epoch = 0
    total_training_time = 0
    
    checkpoint = checkpoint_manager.load()
    if checkpoint:
        print("\n♻️  REPRISE DU TRAINING")
        
        unwrapped_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        global_step = checkpoint['global_step']
        start_dataset_idx = checkpoint['current_dataset_idx']
        resume_epoch = checkpoint['current_epoch']
        training_history = checkpoint['training_history']
        total_training_time = checkpoint.get('total_training_time', 0)
    
    for dataset_idx in range(start_dataset_idx, len(DATASETS_SEQUENCE)):
        dataset_info = DATASETS_SEQUENCE[dataset_idx]
        
        print(f"\n\n{'#'*80}")
        print(f"# DATASET {dataset_idx+1}/{len(DATASETS_SEQUENCE)}: {dataset_info['name'].upper()}")
        print(f"{'#'*80}")
        
        train_tokens, val_tokens = load_single_dataset(dataset_info)
        
        if train_tokens is None:
            print(f"⏭️  Skip {dataset_info['name']}")
            continue
        
        if CONFIG['scheduler_reset_per_dataset'] and resume_epoch == 0:
            print(f"\n🔄 Reset scheduler pour {dataset_info['name']}")
            scheduler = create_scheduler(optimizer, dataset_info['tokens'])
        
        global_step, total_training_time = train_on_dataset(
            model, train_tokens, val_tokens, optimizer, scheduler,
            dataset_info, dataset_idx, checkpoint_manager,
            training_history, resume_epoch, 0,
            global_step, total_training_time
        )
        
        training_history['datasets_completed'].append(dataset_info['name'])
        
        resume_epoch = 0
        
        # ✅ CLEANUP MÉMOIRE AGRESSIF
        print(f"\n🧹 Cleanup mémoire...")
        del train_tokens, val_tokens
        gc.collect()
        torch.cuda.empty_cache()
        print(f"✅ Mémoire libérée")
    
    total_time = time.time()
    
    print("\n" + "="*80)
    print("🎉 TRAINING TERMINÉ!")
    print("="*80)
    print(f"\n📊 STATS FINALES:")
    print(f"   Datasets:    {len(training_history['datasets_completed'])}")
    print(f"   Steps:       {global_step:,}")
    print(f"   Temps:       {total_training_time/3600:.2f}h")
    
    if training_history['validations']:
        final_ppl = training_history['validations'][-1]['perplexity']
        print(f"   PPL final:   {final_ppl:.2f}")
    
    print(f"\n💾 Checkpoint: {checkpoint_manager.path}")
    print("\n✅ SUCCÈS!")

if __name__ == "__main__":
    main()
