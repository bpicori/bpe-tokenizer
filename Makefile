
.PHONY: build run test download-simple-wikipedia download-wikitext2 download-tinystories download-openwebtext clean

DATA_DIR := data

# Build the BPE tokenizer
build:
	mkdir -p bin
	go build -o bin/bpe-tokenizer

# Run the tokenizer
run:
	./bin/bpe-tokenizer

# Run tests
test:
	go test ./...

# Toy dataset (Simple Wikipedia) — small, good for quick sanity checks
download-simple-wikipedia:
	mkdir -p $(DATA_DIR)
	python3 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet datasets
	.venv/bin/python -c "from datasets import load_dataset; ds = load_dataset('rahular/simple-wikipedia'); open('$(DATA_DIR)/simple-wikipedia.txt', 'w').write('\\n'.join(ds['train']['text']))"

# WikiText-2 v1 — same corpus as https://huggingface.co/datasets/mindchain/wikitext2 (HF loads reliably via glue/wikitext).
download-wikitext2:
	mkdir -p $(DATA_DIR)
	python3 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet datasets
	.venv/bin/python -c "from pathlib import Path; from datasets import load_dataset; d = load_dataset('wikitext', 'wikitext-2-v1'); b = Path('$(DATA_DIR)'); [(b / ('wikitext2-%s.txt' % s)).write_text('\\n'.join(d[s]['text'])) for s in ('train', 'validation', 'test')]"

# CS336 Assignment 1 — TinyStories (GPT-4 generated)
# Source: https://huggingface.co/datasets/roneneldan/TinyStories
download-tinystories:
	mkdir -p $(DATA_DIR)
	[ -f $(DATA_DIR)/TinyStoriesV2-GPT4-train.txt ] || curl -L -o $(DATA_DIR)/TinyStoriesV2-GPT4-train.txt \
		https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
	[ -f $(DATA_DIR)/TinyStoriesV2-GPT4-valid.txt ] || curl -L -o $(DATA_DIR)/TinyStoriesV2-GPT4-valid.txt \
		https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# CS336 Assignment 1 — OpenWebText subsample
# Source: https://huggingface.co/datasets/stanford-cs336/owt-sample
download-openwebtext:
	mkdir -p $(DATA_DIR)
	[ -f $(DATA_DIR)/owt_train.txt ] || ( curl -L -o $(DATA_DIR)/owt_train.txt.gz \
		https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz \
		&& gunzip $(DATA_DIR)/owt_train.txt.gz )
	[ -f $(DATA_DIR)/owt_valid.txt ] || ( curl -L -o $(DATA_DIR)/owt_valid.txt.gz \
		https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz \
		&& gunzip $(DATA_DIR)/owt_valid.txt.gz )

# Clean build artifacts
clean:
	rm -rf bin
	rm -f vocab.model
	rm -f training_text.txt
	rm -rf .venv
	rm -rf $(DATA_DIR)


