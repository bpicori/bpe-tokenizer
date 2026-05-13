package main

import (
	"bpicori/bpe-tokenizer/bpe"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

var SpecialTokens = []string{"<|endoftext|>"}

func loadTrainingText(path string) string {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Get file size 
	stat, err := file.Stat()
	if err != nil {
		panic(err)
	}
	
	data := make([]byte, stat.Size())
	
	_, err = io.ReadFull(file, data)
	if err != nil {
		panic(err)
	}
	
	return string(data)
}


func main() {
	trainCmd := flag.NewFlagSet("train", flag.ExitOnError)
	encodeCmd := flag.NewFlagSet("encode", flag.ExitOnError)
	decodeCmd := flag.NewFlagSet("decode", flag.ExitOnError)

	trainFile := trainCmd.String("file", "", "Path to the training-text file (required)")
	encodeInput := encodeCmd.String("text", "", "Text to encode")
	decodeInput := decodeCmd.String("ids", "", "Space-separated list of token IDs to decode")

	if len(os.Args) < 2 {
		fmt.Println("Usage: bpe-tokenizer <command> [arguments]")
		fmt.Println("Commands:")
		fmt.Println("  train  -file=\"<path>\"           Train on the given text file (required)")
		fmt.Println("  encode -text=\"<text>\"           Encode a string into token IDs")
		fmt.Println("  decode -ids=\"<id1 id2 ...>\"     Decode a space-separated list of IDs back to text")
		return
	}

	command := os.Args[1]
	bpe := bpe.NewBPETokenizer(SpecialTokens)

	switch command {
	case "train":
		trainCmd.Parse(os.Args[2:])
		if *trainFile == "" {
			fmt.Println("Usage: bpe-tokenizer train -file=\"<path>\"")
			os.Exit(1)
		}
		trainingText := loadTrainingText(*trainFile)
		bpe.Train(trainingText)
		bpe.Save()
		fmt.Println("Training completed and model saved.")

	case "encode":
		encodeCmd.Parse(os.Args[2:])
		if *encodeInput == "" {
			fmt.Println("Usage: bpe-tokenizer encode -text=\"<text>\"")
			return
		}
		bpe.Load()
		fmt.Println(bpe.Encode(*encodeInput))

	case "decode":
		decodeCmd.Parse(os.Args[2:])
		if *decodeInput == "" {
			fmt.Println("Usage: bpe-tokenizer decode -ids=\"<id1 id2 ...>\"")
			return
		}
		bpe.Load()
		var ids []int
		for _, idStr := range strings.Fields(*decodeInput) {
			var id int
			_, err := fmt.Sscanf(idStr, "%d", &id)
			if err != nil {
				fmt.Println("Invalid ID:", idStr)
				return
			}
			ids = append(ids, id)
		}
		fmt.Println(bpe.Decode(ids))

	default:
		fmt.Println("Unknown command:", command)
		fmt.Println("Commands: train, encode -text=\"<text>\", decode -ids=\"<id1 id2 ...>\"")
	}
}
