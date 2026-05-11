
.PHONY: build run test download-dataset clean

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

download-dataset:
	python3 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet datasets
	.venv/bin/python -c "from datasets import load_dataset; ds = load_dataset('rahular/simple-wikipedia'); f = open('training_text.txt', 'w'); f.write('\n'.join(ds['train']['text'])); f.close()"

# Clean build artifacts
clean:
	rm -rf bin
	rm -f vocab.model
	rm -f training_text.txt
	rm -rf .venv


