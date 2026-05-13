
.PHONY: build run test download-dataset download-tinystories download-openwebtext clean

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
download-dataset:
	python3 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet datasets
	.venv/bin/python -c "from datasets import load_dataset; ds = load_dataset('rahular/simple-wikipedia'); f = open('training_text.txt', 'w'); f.write('\n'.join(ds['train']['text'])); f.close()"

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


