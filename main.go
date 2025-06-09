package main

import (
	"bpicori/bpe-tokenizer/bpe"
	"fmt"
)

func main() {
	trainingText := `Byte-pair encoding[1][2] (also known as BPE, or digram coding)[3] is an algorithm, first described in 1994 by Philip Gage, for encoding strings of text into smaller strings by creating and using a translation table.[4] A slightly modified version of the algorithm is used in large language model tokenizers.The original version of the algorithm focused on compression. It replaces the highest-frequency pair of bytes with a new byte that was not contained in the initial dataset. A lookup table of the replacements is required to rebuild the initial dataset. The modified version builds "tokens" (units of recognition) that match varying amounts of source text, from single characters (including single digits or single punctuation marks) to whole words (even long compound words).[5][6][7]`

	bpe := bpe.NewBPETokenizer()
	bpe.Train(trainingText)

	fmt.Println(bpe.Decode(bpe.Encode("hello world")))
}
