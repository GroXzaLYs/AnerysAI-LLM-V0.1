"""
Main training script for AneryzAI LLM
Demonstrates how to train the model on English and Indonesian data
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json

from src.config import ModelConfig, TrainingConfig, LanguageConfig
from src.model import TransformerModel
from src.tokenizer import BaseTokenizer
from src.training import (
    Trainer,
    LanguageModelDataset,
    create_optimizer,
    create_scheduler,
)
from src.utils import (
    CheckpointManager,
    ConfigManager,
    print_model_info,
    set_seed,
    get_device,
)


def create_sample_data():
    """
    Load comprehensive training data from JSON files in data/ directory
    Data yang lengkap dan bermakna untuk membuat AI lebih pintar
    """

    english_texts = []
    indonesian_texts = []

    try:
        # Load English training data
        with open('data/training_data_english.json', 'r', encoding='utf-8') as f:
            english_data = json.load(f)

        # Extract content from structured data
        for category, items in english_data.items():
            for item in items:
                if 'content' in item:
                    english_texts.append(item['content'])
                if 'title' in item and item['title'] not in english_texts:
                    english_texts.append(item['title'])

        # Load Indonesian training data
        with open('data/training_data_indonesian.json', 'r', encoding='utf-8') as f:
            indonesian_data = json.load(f)

        # Extract content from structured data
        for category, items in indonesian_data.items():
            for item in items:
                if 'content' in item:
                    indonesian_texts.append(item['content'])
                if 'title' in item and item['title'] not in indonesian_texts:
                    indonesian_texts.append(item['title'])

        # Load Q&A training data
        with open('data/qa_training_data.json', 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        # Add Q&A pairs
        for qa_pair in qa_data.get('qa_pairs', []):
            english_texts.append(f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}")

        # Add conversational scenarios
        for scenario in qa_data.get('conversational_scenarios', []):
            conversation_text = f"Scenario: {scenario['scenario']}\n"
            for dialogue in scenario['dialogue']:
                conversation_text += dialogue + "\n"
            english_texts.append(conversation_text)

    except FileNotFoundError as e:
        print(f"Warning: Training data file not found: {e}")
        print("Falling back to minimal sample data...")
        # Fallback to minimal data if files not found
        english_texts = [
            "Artificial intelligence is transforming our world. Machine learning algorithms can learn patterns from data and make predictions.",
            "Deep learning uses neural networks with multiple layers to process complex information and solve difficult problems."
        ]
        indonesian_texts = [
            "Kecerdasan buatan mengubah dunia kita. Algoritma pembelajaran mesin dapat belajar pola dari data dan membuat prediksi.",
            "Pembelajaran mendalam menggunakan jaringan neural dengan banyak lapisan untuk memproses informasi kompleks dan memecahkan masalah sulit."
        ]
    except json.JSONDecodeError as e:
        print(f"Error parsing training data JSON: {e}")
        print("Falling back to minimal sample data...")
        english_texts = ["Artificial intelligence helps solve complex problems."]
        indonesian_texts = ["Kecerdasan buatan membantu memecahkan masalah kompleks."]

    print(f"Loaded {len(english_texts)} English texts and {len(indonesian_texts)} Indonesian texts")

    return english_texts, indonesian_texts


def prepare_training_data(texts, tokenizer, language):
    """
    Prepare training data by tokenizing texts
    """
    tokenized_texts = []
    for text in texts:
        # Add special tokens untuk setiap document
        tokens = tokenizer.encode(text, language=language, add_special_tokens=True)
        if len(tokens) > 0:
            tokenized_texts.append(tokens)
    
    # Concatenate all texts to create longer sequences for better training
    if tokenized_texts:
        all_tokens = []
        for tokens in tokenized_texts:
            all_tokens.extend(tokens)
        # Split into chunks of reasonable size
        chunk_size = 256  # Shorter than max_length to allow for sliding windows
        concatenated_tokens = []
        for i in range(0, len(all_tokens), chunk_size):
            chunk = all_tokens[i:i + chunk_size + 1]  # +1 for next token prediction
            if len(chunk) >= 128:  # Minimum length
                concatenated_tokens.append(chunk)
        return concatenated_tokens
    
    return tokenized_texts


def main(args):
    """
    Main training loop
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = args.device if args.device else get_device()
    print(f"Using device: {device}")
    
    # Create model config
    model_config = ModelConfig(
        vocab_size=1000,  # Temporary, will be updated
        max_sequence_length=256,  # Reduced for our data size
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    
    # Create training config  
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        device=device,
    )
    
    print(f"\n{'='*60}")
    print("AneryzAI LLM Training Configuration")
    print(f"{'='*60}")
    print(f"Model Config:")
    print(f"  Vocab Size: {model_config.vocab_size}")
    print(f"  Max Sequence Length: {model_config.max_sequence_length}")
    print(f"  Embedding Dim: {model_config.embedding_dim}")
    print(f"  Num Heads: {model_config.num_heads}")
    print(f"  Num Layers: {model_config.num_layers}")
    print(f"\nTraining Config:")
    print(f"  Physical Batch Size: {training_config.batch_size}")
    print(f"  Effective Batch Size: {args.effective_batch_size}")
    print(f"  Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"  Learning Rate: {training_config.learning_rate:.2e}")
    print(f"  Num Epochs: {training_config.num_epochs}")
    print(f"  Max Steps: {training_config.max_steps}")
    print(f"{'='*60}\n")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = BaseTokenizer(vocab_size=model_config.vocab_size)
    
    # Get sample data
    english_texts, indonesian_texts = create_sample_data()
    
    # Train tokenizer on both languages
    print("\nTraining tokenizer on English texts...")
    tokenizer.train(english_texts, language="en")
    
    print("Training tokenizer on Indonesian texts...")
    tokenizer.train(indonesian_texts, language="id")
    
    # Update model config with actual vocab size
    actual_vocab_size = len(tokenizer.idx2word)
    print(f"Actual vocabulary size: {actual_vocab_size}")
    model_config.vocab_size = actual_vocab_size
    
    # Save tokenizer
    tokenizer_path = Path("checkpoints/tokenizer.json")
    tokenizer_path.parent.mkdir(exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    
    # Prepare training data
    print("\nPreparing training data...")
    en_tokens = prepare_training_data(english_texts, tokenizer, "en")
    id_tokens = prepare_training_data(indonesian_texts, tokenizer, "id")
    
    # Combine data
    all_tokens = en_tokens + id_tokens
    
    if len(all_tokens) == 0:
        print("Error: No training data available!")
        return
    
    # Create dataset
    dataset = LanguageModelDataset(
        token_ids_list=all_tokens,
        max_length=model_config.max_sequence_length,
        stride=model_config.max_sequence_length // 2,  # Overlapping windows for more data
    )
    
    print(f"Dataset size: {len(dataset)} examples")
    
    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = TransformerModel(model_config).to(device)
    print_model_info(model)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    
    scheduler = create_scheduler(
        optimizer,
        warmup_steps=training_config.warmup_steps,
        max_steps=training_config.max_steps,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=training_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Save configuration
    config_dict = {
        "model_config": {
            "vocab_size": model_config.vocab_size,
            "max_sequence_length": model_config.max_sequence_length,
            "embedding_dim": model_config.embedding_dim,
            "num_heads": model_config.num_heads,
            "num_layers": model_config.num_layers,
            "ffn_dim": model_config.ffn_dim,
            "dropout_rate": model_config.dropout_rate,
        },
        "training_config": {
            "physical_batch_size": training_config.batch_size,
            "effective_batch_size": args.effective_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": training_config.learning_rate,
            "num_epochs": training_config.num_epochs,
            "max_steps": training_config.max_steps,
        },
        "supported_languages": ["en", "id"],
    }
    
    ConfigManager.save_config(
        config_dict,
        Path("checkpoints/config.json")
    )
    
    # Train model
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    trainer.train(
        num_epochs=training_config.num_epochs,
        save_path="checkpoints/best_model.pt"
    )
    
    # Save final model
    final_model_path = "checkpoints/final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    print(f"\n{'='*60}")
    print("Training Completed Successfully!")
    print(f"Checkpoints saved in: checkpoints/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AnerysAI LLM")
    
    # Model arguments
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Vocabulary size")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--embedding_dim", type=int, default=768,
                        help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Physical batch size (will be adjusted for gradient accumulation)")
    parser.add_argument("--effective_batch_size", type=int, default=9000,
                        help="Effective batch size (simulated with gradient accumulation)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Gradient accumulation steps (auto-calculated if not specified)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Maximum training steps")
    
    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, cpu, mps)")
    
    args = parser.parse_args()
    
    # Calculate gradient accumulation steps for effective batch size
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = max(1, args.effective_batch_size // args.batch_size)
    
    # Adjust learning rate for larger effective batch size (linear scaling rule)
    effective_lr = args.learning_rate * (args.effective_batch_size / 256)  # Scale from base batch size of 256
    args.learning_rate = effective_lr
    
    main(args)
