# BPE Tokenizer

A **byte-level Byte Pair Encoding (BPE)** tokenizer in Go, following GPT-4's
pre-tokenization regex and supporting reserved special tokens (e.g.
`<|endoftext|>`) that are never split or merged.

## Quick start

```bash
go mod download
make build
```

```bash
./bin/bpe-tokenizer train -file=<path>
./bin/bpe-tokenizer encode -text="hello world"   # → [104 9349 1294]
./bin/bpe-tokenizer decode -ids="104 9349 1294"  # → hello world
```

`encode` and `decode` load `vocab.model` from the current directory, so run
`train` first.

## Datasets

```bash
make download-simple-wikipedia # Simple Wikipedia → data/simple-wikipedia.txt
make download-wikitext2       # WikiText-2 v1 → data/wikitext2-{train,validation,test}.txt
make download-tinystories     # TinyStories, ~2 GB → data/TinyStoriesV2-GPT4-*.txt
make download-openwebtext    # OpenWebText sample, ~12 GB raw → data/owt_*.txt
```

## Special tokens

`<|endoftext|>` is reserved by default (`main.go`) and gets a fixed ID right
after the byte range (`256`). To register a different set:

```go
bpe.NewBPETokenizer([]string{"<|endoftext|>", "<|other|>", ...})
```

IDs are assigned in order (`256`, `257`, …) and persisted in `vocab.model`.

## Configuration

In `bpe/bpe.go`:

- `VOCAB_SIZE` — total vocabulary size (default `10_000`). Merge count is
  `VOCAB_SIZE − 256 − len(specialTokens)`.
- `GPT4_SPLIT_PATTERN` — pre-tokenization regex (mirrors tiktoken's
  `cl100k_base`).

## References

- Karpathy, *minbpe* — https://github.com/karpathy/minbpe
- *Let's build the GPT Tokenizer* — https://www.youtube.com/watch?v=zduSFxRajkE
