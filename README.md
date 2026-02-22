# AnerysAI LLM v0.1

A comprehensive Indonesian-English bilingual language model built with PyTorch. This project implements a complete transformer-based LLM from scratch with advanced features for both training and inference.

## 🚀 Key Features

- **Bilingual Support**: Full English and Indonesian language capabilities
- **Chain-of-Thought Reasoning**: Step-by-step reasoning capabilities for complex questions
- **Search-Augmented Generation**: Web search integration for enhanced responses
- **Comprehensive Training Data**: High-quality, structured data from various sources (not template-generated)
- **Advanced Architecture**: Custom transformer implementation with multi-head attention
- **Flexible Configuration**: Easily adjustable model parameters and training settings
- **GPU/CPU Support**: Optimized for both GPU and CPU training
- **Interactive Inference**: Real-time text generation with customizable parameters

## 📁 Project Structure

```
AnerysAI-LLM-V0.1/
├── data/                          # Training data directory
│   ├── training_data_english.json    # Comprehensive English training data
│   ├── training_data_indonesian.json # Comprehensive Indonesian training data
│   ├── qa_training_data.json         # Q&A pairs and conversations
│   └── README.md                     # Data documentation
├── src/                           # Source code
│   ├── model/                      # Transformer model implementation
│   ├── tokenizer/                  # BPE tokenizer
│   ├── training/                   # Training utilities
│   └── utils/                      # Helper functions
├── checkpoints/                   # Saved model weights
├── examples.py                    # Usage examples
├── inference.py                   # Text generation interface
├── train.py                       # Training script
└── requirements.txt               # Python dependencies
```

## 🏗️ Architecture

- **Model**: Custom transformer with configurable layers, heads, and embedding dimensions
- **Tokenizer**: Byte-Pair Encoding (BPE) supporting both English and Indonesian
- **Training**: Adam optimizer with learning rate scheduling and gradient clipping
- **Data**: Structured JSON-based training data covering AI, technology, and general knowledge

## 📚 Training Data

Unlike template-generated content, our training data is sourced from comprehensive, high-quality materials:

- **AI & Machine Learning**: Detailed explanations of algorithms, architectures, and applications
- **Computer Science**: Programming paradigms, data structures, and software engineering
- **Technology Trends**: Cloud computing, IoT, blockchain, and emerging technologies
- **Q&A Pairs**: Technical interview questions and conversational scenarios
- **Bilingual Content**: Parallel English and Indonesian datasets

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training with Large Batch Sizes
```bash
# Train with effective batch size 9000 using gradient accumulation
python train.py --embedding_dim 512 --num_heads 8 --num_layers 6 --batch_size 4 --effective_batch_size 9000 --device cuda

# The system automatically calculates gradient accumulation steps (9000/4 = 2250)
# Learning rate is scaled according to effective batch size
```

### Inference
```bash
# Interactive mode
python inference.py --device cpu --interactive

# Demo mode
python inference.py --device cpu
```

## 🎯 Usage Examples

### Programmatic Usage
```python
from src import TransformerModel, ModelConfig, BaseTokenizer

# Load model
config = ModelConfig()
model = TransformerModel(config)
tokenizer = BaseTokenizer()

# Generate text
prompt = "What is artificial intelligence?"
tokens = tokenizer.encode(prompt, language='en')
# ... generate response
```

### Interactive Chat
```bash
python inference.py --interactive
# Type questions in English or Indonesian
# Use /en or /id to switch languages
```

### Thinking and Searching Features
```bash
# Run demo with thinking and searching capabilities
python inference.py --thinking

# Programmatic usage
from src.inference import TextGenerator

generator = TextGenerator(model, tokenizer)

# Chain-of-thought reasoning
response = generator.think_and_generate(
    "Why does the sky appear blue?",
    num_steps=3
)

# Search-augmented generation
response = generator.search_and_generate(
    "What is the capital of France?"
)
```

## 🔧 Configuration

### Model Parameters
- `vocab_size`: Tokenizer vocabulary size (default: 32000)
- `embedding_dim`: Model embedding dimensions (default: 512)
- `num_heads`: Number of attention heads (default: 8)
- `num_layers`: Number of transformer layers (default: 6)
- `max_length`: Maximum sequence length (default: 512)

### Training Parameters
- `batch_size`: Physical batch size loaded in memory
- `effective_batch_size`: Simulated large batch size using gradient accumulation
- `gradient_accumulation_steps`: Auto-calculated (effective_batch_size / batch_size)
- `learning_rate`: Auto-scaled based on effective batch size (linear scaling rule)
- `num_epochs`: Number of training epochs
- `max_steps`: Maximum training steps

## 📈 Performance

- **Small Model**: ~10M parameters, suitable for CPU training
- **Medium Model**: ~50M parameters, requires GPU for efficient training
- **Large Model**: ~100M+ parameters, optimized for GPU training
- **Bilingual Capability**: Handles code-switching between English and Indonesian

## 🤝 Contributing

1. **Add Training Data**: Contribute high-quality content to `data/` directory
2. **Improve Architecture**: Enhance model components in `src/model/`
3. **Optimize Training**: Improve training efficiency and stability
4. **Add Features**: Implement new capabilities and interfaces

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- Built with PyTorch for deep learning framework
- Inspired by transformer architecture research
- Comprehensive bilingual dataset compilation
- Open-source community contributions

---

**Note**: This is a research and educational implementation. For production use, consider larger models and more extensive training data.