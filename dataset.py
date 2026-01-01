#!/usr/bin/env python3
"""
📥 Dataset Downloader pour HessGPT V5
Télécharge et met en cache tous les datasets
"""

import torch
import os
import time
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer
import traceback

print("="*80)
print("📥 DATASET DOWNLOADER - HessGPT V5")
print("="*80)

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    'tokens_per_dataset': 1_000_000_000,  # 1B tokens par dataset
    'min_text_length': 200,
    'max_text_length': 5000,
    'val_tokens': 10_000_000,  # 10M tokens pour validation
    'max_retries': 3,
}

DATASETS = [
    {
        'name': 'wikipedia',
        'source': 'wikimedia/wikipedia',
        'config': '20231101.en',  # ← Corrigé !
        'split': 'train',
        'text_key': 'text',
        'streaming': True,  # ← Streaming activé
        'description': '📚 Encyclopedia',
    },
    {
        'name': 'openwebtext',
        'source': 'openwebtext',
        'config': None,
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '🌍 Reddit content',
    },
    {
        'name': 'pile_books',
        'source': 'monology/pile-uncopyrighted',
        'config': 'all',
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '📖 Books from Pile',
    },
]

# ============================================
# SETUP
# ============================================
cache_dir = "data/cache"
os.makedirs(cache_dir, exist_ok=True)

print(f"\n📂 Cache directory: {cache_dir}")
print(f"📊 Datasets à télécharger: {len(DATASETS)}")
print(f"🎯 Target par dataset: {CONFIG['tokens_per_dataset']/1e9:.1f}B tokens")
print(f"\n{'='*80}\n")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ============================================
# FONCTIONS
# ============================================
def tokenize_texts(texts, tokenizer):
    """Tokenize et filtre les textes"""
    tokens = []
    filtered = 0
    
    for text in texts:
        # Filtre longueur
        if len(text) < CONFIG['min_text_length']:
            filtered += 1
            continue
        
        # Filtre qualité (ratio caractères alphanumériques)
        sample = text[:1000]
        alpha_ratio = sum(c.isalnum() or c.isspace() for c in sample) / len(sample)
        if alpha_ratio < 0.7:
            filtered += 1
            continue
        
        # Tokenize (limite longueur)
        encoded = tokenizer.encode(
            text[:CONFIG['max_text_length']],
            add_special_tokens=False
        )
        tokens.extend(encoded)
        tokens.append(tokenizer.eos_token_id)
    
    return tokens, filtered

def download_dataset(dataset_info):
    """Télécharge et met en cache un dataset"""
    print(f"{'='*80}")
    print(f"📥 DATASET: {dataset_info['name'].upper()}")
    print(f"   {dataset_info['description']}")
    print(f"   Target: {CONFIG['tokens_per_dataset']/1e9:.1f}B tokens")
    print(f"{'='*80}")
    
    # Vérifie si déjà en cache
    cache_file = f"{cache_dir}/{dataset_info['name']}_{CONFIG['tokens_per_dataset']//1e9:.0f}B_cache.pt"
    
    if os.path.exists(cache_file):
        print(f"✅ Cache déjà existant: {cache_file}")
        try:
            cached = torch.load(cache_file)
            print(f"   Train tokens: {len(cached['train_tokens'])/1e6:.1f}M")
            print(f"   Val tokens: {len(cached['val_tokens'])/1e6:.1f}M")
            print(f"⏭️  Skip (déjà téléchargé)\n")
            return True
        except Exception as e:
            print(f"⚠️  Cache corrompu: {e}")
            print(f"🗑️  Suppression et re-téléchargement...")
            os.remove(cache_file)
    
    # Téléchargement avec retries
    for attempt in range(CONFIG['max_retries']):
        try:
            print(f"\n⏳ Téléchargement (tentative {attempt+1}/{CONFIG['max_retries']})...")
            start_time = time.time()
            
            # Load dataset
            if dataset_info['streaming']:
                dataset = load_dataset(
                    dataset_info['source'],
                    name=dataset_info['config'],
                    split=dataset_info['split'],
                    streaming=True,
                    trust_remote_code=False
                )
            else:
                dataset = load_dataset(
                    dataset_info['source'],
                    dataset_info['config'],
                    split=dataset_info['split'],
                    trust_remote_code=False
                )
            
            # Tokenization
            all_tokens = []
            batch_texts = []
            doc_count = 0
            total_filtered = 0
            
            max_tokens = CONFIG['tokens_per_dataset']
            estimated_docs = int(max_tokens / 500)  # ~500 tokens par doc en moyenne
            
            pbar = tqdm(
                dataset if hasattr(dataset, '__iter__') else iter(dataset),
                total=estimated_docs,
                desc=f"📝 Tokenizing"
            )
            
            for item in pbar:
                # Stop si assez de tokens
                if len(all_tokens) >= max_tokens:
                    break
                
                text = item.get(dataset_info['text_key'], '')
                
                if not text or not isinstance(text, str):
                    continue
                
                batch_texts.append(text)
                doc_count += 1
                
                # Process par batch de 1000
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
            
            # Process dernier batch
            if batch_texts:
                tokens, filtered = tokenize_texts(batch_texts, tokenizer)
                all_tokens.extend(tokens)
                total_filtered += filtered
            
            # Truncate si trop de tokens
            if len(all_tokens) > max_tokens:
                all_tokens = all_tokens[:max_tokens]
            
            elapsed = time.time() - start_time
            
            print(f"\n✅ Téléchargement réussi!")
            print(f"   Documents traités: {doc_count:,}")
            print(f"   Documents filtrés: {total_filtered:,}")
            print(f"   Tokens collectés: {len(all_tokens)/1e9:.3f}B")
            print(f"   Temps: {elapsed/60:.1f} min")
            print(f"   Vitesse: {len(all_tokens)/elapsed/1000:.1f}K tokens/s")
            
            # Split train/val
            val_size = CONFIG['val_tokens']
            val_tokens = all_tokens[:val_size]
            train_tokens = all_tokens[val_size:]
            
            print(f"\n💾 Sauvegarde du cache...")
            print(f"   Train: {len(train_tokens)/1e6:.1f}M tokens")
            print(f"   Val:   {len(val_tokens)/1e6:.1f}M tokens")
            
            torch.save({
                'train_tokens': train_tokens,
                'val_tokens': val_tokens,
                'dataset_info': dataset_info,
                'config': CONFIG,
                'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            }, cache_file)
            
            print(f"✅ Cache sauvegardé: {cache_file}")
            print(f"   Taille fichier: {os.path.getsize(cache_file)/1e9:.2f} GB\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Erreur tentative {attempt+1}:")
            print(f"   {str(e)[:300]}")
            
            if attempt < CONFIG['max_retries'] - 1:
                wait_time = 30 * (attempt + 1)
                print(f"⏳ Attente {wait_time}s avant nouvelle tentative...")
                time.sleep(wait_time)
            else:
                print(f"\n❌ ÉCHEC après {CONFIG['max_retries']} tentatives")
                print(f"⚠️  Dataset {dataset_info['name']} non téléchargé")
                print(f"\n🔍 Traceback complet:")
                print(traceback.format_exc())
                return False

# ============================================
# MAIN
# ============================================
def main():
    print("🚀 Début du téléchargement...\n")
    
    start_time = time.time()
    success_count = 0
    failed_datasets = []
    
    for i, dataset_info in enumerate(DATASETS, 1):
        print(f"\n{'#'*80}")
        print(f"# DATASET {i}/{len(DATASETS)}")
        print(f"{'#'*80}\n")
        
        success = download_dataset(dataset_info)
        
        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset_info['name'])
    
    total_time = time.time() - start_time
    
    # Résumé final
    print("\n" + "="*80)
    print("🎉 TÉLÉCHARGEMENT TERMINÉ!")
    print("="*80)
    print(f"\n📊 RÉSUMÉ:")
    print(f"   Réussis: {success_count}/{len(DATASETS)}")
    if failed_datasets:
        print(f"   Échoués: {', '.join(failed_datasets)}")
    print(f"   Temps total: {total_time/60:.1f} min")
    print(f"\n📂 Cache directory: {cache_dir}")
    
    # Liste les fichiers créés
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('_cache.pt')]
    if cache_files:
        print(f"\n💾 Fichiers créés:")
        total_size = 0
        for f in sorted(cache_files):
            path = os.path.join(cache_dir, f)
            size = os.path.getsize(path) / 1e9
            total_size += size
            print(f"   - {f:40} ({size:.2f} GB)")
        print(f"\n   Total: {total_size:.2f} GB")
    
    print("\n✅ Les datasets sont prêts pour le training!")
    print("   Lance maintenant train_hessgpt_v5.py pour commencer l'entraînement")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Téléchargement interrompu par l'utilisateur")
        print("Les datasets déjà téléchargés sont sauvegardés dans le cache")
    except Exception as e:
        print(f"\n\n❌ ERREUR FATALE:")
        print(traceback.format_exc())