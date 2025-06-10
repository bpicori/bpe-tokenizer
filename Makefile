
.PHONY: build run test download-dataset clean

# Build the BPE tokenizer
build:
	go build -o bpe-tokenizer

# Run the tokenizer
run:
	./bpe-tokenizer

# Run tests
test:
	go test ./...

download-dataset:
	mkdir -p wiki_dataset
	huggingface-cli download rahular/simple-wikipedia --repo-type dataset --local-dir wiki_dataset
	python -c "from datasets import load_dataset; ds = load_dataset('rahular/simple-wikipedia'); f = open('training_text.txt', 'w'); f.write('\n'.join(ds['train']['text'])); f.close()"
	rm -rf wiki_dataset

# Clean build artifacts
clean:
	rm -f bpe-tokenizer
	rm -f vocab.model
	rm -f training_text.txt
	rm -rf wiki_dataset


