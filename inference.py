"""
Inference script for AneryzAI LLM
Demonstrates text generation in English and Indonesian
"""

import torch
from pathlib import Path
import argparse
import json

from src.config import ModelConfig
from src.model import TransformerModel
from src.tokenizer import BaseTokenizer
from src.inference import TextGenerator, GenerationType, BatchGenerator
from src.utils import get_device, LanguageDetector, ConfigManager


def load_model_and_tokenizer(checkpoint_dir="checkpoints", device="cpu"):
    """
    Load pre-trained model and tokenizer
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load config
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        print(f"Config not found at {config_path}")
        return None, None
    
    config_dict = ConfigManager.load_config(str(config_path))
    
    # Create model config
    model_config = ModelConfig(
        vocab_size=config_dict["model_config"]["vocab_size"],
        max_sequence_length=config_dict["model_config"]["max_sequence_length"],
        embedding_dim=config_dict["model_config"]["embedding_dim"],
        num_heads=config_dict["model_config"]["num_heads"],
        num_layers=config_dict["model_config"]["num_layers"],
        ffn_dim=config_dict["model_config"]["ffn_dim"],
        dropout_rate=config_dict["model_config"]["dropout_rate"],
    )
    
    # Load model
    model = TransformerModel(model_config).to(device)
    
    # Load weights
    model_path = checkpoint_dir / "best_model.pt"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model not found at {model_path}, using random initialization")
    
    # Load tokenizer
    tokenizer = BaseTokenizer(vocab_size=model_config.vocab_size)
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer.load(str(tokenizer_path))
        print(f"Tokenizer loaded from {tokenizer_path}")
    else:
        print(f"Tokenizer not found at {tokenizer_path}")
        return model, None
    
    model.eval()
    return model, tokenizer


def demonstrate_single_generation(model, tokenizer, device):
    """
    Demonstrate single prompt generation
    """
    print(f"\n{'='*70}")
    print("Single Prompt Generation Demo")
    print(f"{'='*70}\n")
    
    # Create text generator
    generator = TextGenerator(model, tokenizer, device=device)
    
    # English examples
    english_prompts = [
        "The future of artificial intelligence is",
        "Machine learning enables computers to",
        "Natural language processing helps us understand",
    ]
    
    print("English Generation Examples:")
    print("-" * 70)
    
    for prompt in english_prompts:
        print(f"\nPrompt: {prompt}")
        
        # Detect language
        detected_lang = LanguageDetector.detect_language(prompt)
        
        # Generate with different strategies
        try:
            # Top-K sampling
            generated, score = generator.generate(
                prompt=prompt,
                language=detected_lang,
                max_new_tokens=30,
                generation_type=GenerationType.TOP_K,
                top_k=20,
                temperature=0.7,
                return_scores=True,
            )
            print(f"Generated (Top-K): {generated}")
            print(f"Score: {score:.4f}")
        except Exception as e:
            print(f"Generation error: {e}")
    
    # Indonesian examples
    indonesian_prompts = [
        "Kecerdasan buatan adalah",
        "Pembelajaran mesin memungkinkan komputer untuk",
        "Pemrosesan bahasa alami sangat penting untuk",
    ]
    
    print("\n\n" + "="*70)
    print("Indonesian Generation Examples:")
    print("-" * 70)
    
    for prompt in indonesian_prompts:
        print(f"\nPrompt: {prompt}")
        
        # Detect language
        detected_lang = LanguageDetector.detect_language(prompt)
        
        try:
            # Top-P sampling
            generated, score = generator.generate(
                prompt=prompt,
                language=detected_lang,
                max_new_tokens=30,
                generation_type=GenerationType.TOP_P,
                top_p=0.8,
                temperature=0.7,
                return_scores=True,
            )
            print(f"Generated (Top-P): {generated}")
            print(f"Score: {score:.4f}")
        except Exception as e:
            print(f"Generation error: {e}")


def demonstrate_batch_generation(model, tokenizer, device):
    """
    Demonstrate batch generation for multiple prompts
    """
    print(f"\n{'='*70}")
    print("Batch Generation Demo")
    print(f"{'='*70}\n")
    
    generator = TextGenerator(model, tokenizer, device=device)
    batch_gen = BatchGenerator(generator)
    
    # Mix of English and Indonesian prompts
    prompts = [
        "Artificial intelligence and machine learning",
        "Pembelajaran mendalam adalah teknologi yang",
        "The transformer architecture has revolutionized",
        "Jaringan neural konvolusional digunakan untuk",
    ]
    
    print("Generating completions for batch of prompts...")
    print("-" * 70)
    
    try:
        results = batch_gen.generate_batch(
            prompts,
            language="en",  # Default, will be auto-detected
            max_new_tokens=40,
            generation_type=GenerationType.TOP_K,
            top_k=50,
        )
        
        for prompt, (generated, score) in zip(prompts, results):
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated}")
            if score:
                print(f"Score: {score:.4f}")
    
    except Exception as e:
        print(f"Batch generation error: {e}")


def demonstrate_thinking_and_searching(model, tokenizer, device):
    """
    Demonstrate thinking (chain-of-thought) and searching capabilities
    """
    print(f"\n{'='*70}")
    print("Thinking and Searching Demo")
    print(f"{'='*70}\n")
    
    generator = TextGenerator(model, tokenizer, device=device)
    
    # Chain-of-thought examples
    print("Chain-of-Thought Reasoning Examples:")
    print("-" * 70)
    
    thinking_prompts = [
        "Why does the sky appear blue during the day?",
        "How does photosynthesis work in plants?",
        "Apa itu kecerdasan buatan dan bagaimana cara kerjanya?",
    ]
    
    for prompt in thinking_prompts:
        print(f"\nPrompt: {prompt}")
        try:
            thought_process = generator.think_and_generate(
                prompt,
                language="en" if "blue" in prompt or "photosynthesis" in prompt else "id",
                max_new_tokens=80,
                num_steps=2,
            )
            print(f"Reasoning: {thought_process}")
        except Exception as e:
            print(f"Thinking error: {e}")
    
    # Search and generate examples
    print(f"\n{'='*70}")
    print("Search-Augmented Generation Examples:")
    print("-" * 70)
    
    search_queries = [
        "What is the capital of France?",
        "Berapa populasi Indonesia saat ini?",
        "What are the benefits of renewable energy?",
    ]
    
    for query in search_queries:
        print(f"\nQuery: {query}")
        try:
            search_response = generator.search_and_generate(
                query,
                language="en" if "France" in query or "renewable" in query else "id",
                max_new_tokens=100,
            )
            print(f"Response: {search_response}")
        except Exception as e:
            print(f"Search error: {e}")


def interactive_generation(model, tokenizer, device):
    """
    Interactive text generation - user can input prompts
    """
    print(f"\n{'='*70}")
    print("Interactive Generation Mode")
    print(f"{'='*70}")
    print("Type 'quit' to exit")
    print("Commands:")
    print("  /en - Force English language")
    print("  /id - Force Indonesian language")
    print("  /think - Enable chain-of-thought reasoning")
    print("  /search - Enable search-augmented generation")
    print("  /normal - Normal generation mode")
    print("  /config - Show generation config")
    print("-" * 70)
    
    generator = TextGenerator(model, tokenizer, device=device)
    
    # Default generation config
    gen_config = {
        "max_new_tokens": 50,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 0.95,
        "generation_type": GenerationType.TOP_P.value,
    }
    
    forced_language = None
    generation_mode = "normal"  # "normal", "think", "search"
    
    while True:
        try:
            user_input = input("\nPrompt: ").strip()
            
            if user_input.lower() == "quit":
                print("Exiting interactive mode...")
                break
            
            if user_input.startswith("/"):
                # Process commands
                if user_input == "/en":
                    forced_language = "en"
                    print("Language forced to English")
                elif user_input == "/id":
                    forced_language = "id"
                    print("Language forced to Indonesian")
                elif user_input == "/think":
                    generation_mode = "think"
                    print("Chain-of-thought reasoning enabled")
                elif user_input == "/search":
                    generation_mode = "search"
                    print("Search-augmented generation enabled")
                elif user_input == "/normal":
                    generation_mode = "normal"
                    print("Normal generation mode")
                elif user_input == "/config":
                    print("\nCurrent Generation Config:")
                    for key, value in gen_config.items():
                        print(f"  {key}: {value}")
                    print(f"Mode: {generation_mode}")
                else:
                    print("Unknown command")
                continue
            
            if not user_input:
                continue
            
            # Detect language
            language = forced_language or LanguageDetector.detect_language(user_input)
            
            print(f"\nDetected Language: {language}")
            print(f"Mode: {generation_mode}")
            print("Generating...")
            
            # Generate based on mode
            if generation_mode == "think":
                generated = generator.think_and_generate(
                    prompt=user_input,
                    language=language,
                    max_new_tokens=gen_config["max_new_tokens"],
                    temperature=gen_config["temperature"],
                    top_k=gen_config["top_k"],
                    top_p=gen_config["top_p"],
                    num_steps=3,
                )
                score = None
            elif generation_mode == "search":
                generated = generator.search_and_generate(
                    query=user_input,
                    language=language,
                    max_new_tokens=gen_config["max_new_tokens"],
                    temperature=gen_config["temperature"],
                    top_k=gen_config["top_k"],
                    top_p=gen_config["top_p"],
                )
                score = None
            else:
                generated, score = generator.generate(
                    prompt=user_input,
                    language=language,
                    max_new_tokens=gen_config["max_new_tokens"],
                    temperature=gen_config["temperature"],
                    top_k=gen_config["top_k"],
                    top_p=gen_config["top_p"],
                    generation_type=GenerationType[gen_config["generation_type"].upper()],
                    return_scores=True,
                )
            
            print(f"\nGenerated: {generated}")
            if score:
                print(f"Score: {score:.4f}")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main(args):
    """
    Main inference function
    """
    # Get device
    device = args.device if args.device else get_device()
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        checkpoint_dir=args.checkpoint_dir,
        device=device
    )
    
    if model is None or tokenizer is None:
        print("Error: Failed to load model or tokenizer")
        return
    
    print(f"\nModel and tokenizer loaded successfully!")
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")
    
    # Run demonstrations or interactive mode
    if args.interactive:
        interactive_generation(model, tokenizer, device)
    else:
        # Run demo
        demonstrate_single_generation(model, tokenizer, device)
        
        if args.batch:
            demonstrate_batch_generation(model, tokenizer, device)
        
        if args.thinking:
            demonstrate_thinking_and_searching(model, tokenizer, device)
    
    print(f"\n{'='*70}")
    print("Inference completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for AnerysAI LLM")
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing model checkpoints"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, mps)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Also run batch generation demo"
    )
    
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Also run thinking and searching demo"
    )
    
    args = parser.parse_args()
    main(args)
