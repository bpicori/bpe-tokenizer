package main

import (
	"bpicori/bpe-tokenizer/bpe"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

func loadTrainingText() string {
	file, err := os.Open("training_text.txt")
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

	encodeInput := encodeCmd.String("text", "", "Text to encode")
	decodeInput := decodeCmd.String("ids", "", "Space-separated list of token IDs to decode")

	if len(os.Args) < 2 {
		fmt.Println("Usage: bpe-tokenizer <command> [arguments]")
		fmt.Println("Commands: train, encode -text=\"<text>\", decode -ids=\"<id1 id2 ...>\"")
		return
	}

	command := os.Args[1]
	bpe := bpe.NewBPETokenizer()

	switch command {
	case "train":
		trainCmd.Parse(os.Args[2:])
		trainingText := loadTrainingText()
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
