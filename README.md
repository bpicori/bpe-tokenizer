# BPE Tokenizer

A **Byte Pair Encoding (BPE)** tokenizer implementation in Go that follows GPT-4's tokenization approach. Build custom vocabularies, encode text to tokens, and decode tokens back to text.

## Quick Start

### Installation
```bash
go mod download
make build
```

### Usage
```bash
# 1. Get training data (optional - uses Simple Wikipedia) or provide your own in training_text.txt
make download-dataset

# 2. Train the tokenizer
./bpe-tokenizer train

# 3. Encode text to tokens
./bpe-tokenizer encode --text="hello world"
# Output: [104 9349 1294]

# 4. Decode tokens back to text
./bpe-tokenizer decode -ids="104 9349 1294"
# Output: hello world
```

## Configuration

Modify constants in `bpe/bpe.go`:
- `VOCAB_SIZE`: Total vocabulary size (default: 10,000)
- `GPT4_SPLIT_PATTERN`: Text splitting regex

## References
* https://www.youtube.com/watch?v=zduSFxRajkE
* https://github.com/karpathy/minbpe