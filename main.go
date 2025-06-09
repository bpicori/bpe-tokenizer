package main

import (
	"bytes"
	"fmt"
)

type Pair struct {
	First  int
	Second int
}

func (p Pair) String() string {
	return fmt.Sprintf("%d-%d", p.First, p.Second)
}

type Merge struct {
	Pair  Pair
	Index int
}

const VOCAB_SIZE = 279

func merge(list []int, pair Pair, index int) []int {
	newList := []int{}

	for i := 0; i < len(list); i++ {
		if i < len(list)-1 && list[i] == pair.First && list[i+1] == pair.Second {
			newList = append(newList, index)
			i++ // skip the next element as we merged
		} else {
			newList = append(newList, list[i])
		}
	}

	return newList
}

func stats(tokens []int) map[Pair]int {
	m := make(map[Pair]int)

	for i := range len(tokens) - 1 {
		curr := tokens[i]
		next := tokens[i+1]
		pair := Pair{curr, next}
		m[pair]++
	}

	return m
}

func mostFrequentPair(m map[Pair]int) Pair {
	max := 0
	maxPair := Pair{}

	for pair, count := range m {
		if count > max {
			max = count
			maxPair = pair
		}
	}

	return maxPair
}

func decode(tokens []int, merges []Merge) string {
	vocab := make(map[int][]byte)
	for i := 0; i < 256; i++ {
		vocab[i] = []byte{byte(i)}
	}

	for _, merge := range merges {
		vocab[merge.Index] = append(vocab[merge.Pair.First], vocab[merge.Pair.Second]...)
	}

	var buffer bytes.Buffer
	for _, token := range tokens {
		buffer.Write(vocab[token])
	}
	byteSequence := buffer.Bytes()

	// Decode to UTF-8
	text := string(bytes.Runes(byteSequence))

	return text
}

func encode(text string, merges []Merge) []int {
	tokens := make([]int, len(text))
	for i, b := range []byte(text) {
		tokens[i] = int(b)
	}

	for _, m := range merges {
		tokens = merge(tokens, m.Pair, m.Index)
	}

	return tokens
}

func buildVocab(text string) ([]int, []Merge) {
	tokens := make([]int, len(text))
	for i, b := range []byte(text) {
		tokens[i] = int(b)
	}

	numOfMerges := VOCAB_SIZE - 256

	merges := []Merge{}
	for i := 0; i < numOfMerges; i++ {
		stats := stats(tokens)
		maxUsedPair := mostFrequentPair(stats)
		idx := 256 + i
		tokens = merge(tokens, maxUsedPair, idx)
		merges = append(merges, Merge{maxUsedPair, idx})
	}

	return tokens, merges
}

func main() {
	trainingText := `Byte-pair encoding[1][2] (also known as BPE, or digram coding)[3] is an algorithm, first described in 1994 by Philip Gage, for encoding strings of text into smaller strings by creating and using a translation table.[4] A slightly modified version of the algorithm is used in large language model tokenizers.The original version of the algorithm focused on compression. It replaces the highest-frequency pair of bytes with a new byte that was not contained in the initial dataset. A lookup table of the replacements is required to rebuild the initial dataset. The modified version builds "tokens" (units of recognition) that match varying amounts of source text, from single characters (including single digits or single punctuation marks) to whole words (even long compound words).[5][6][7]`

	_, merges := buildVocab(trainingText)

	fmt.Println(decode(encode("hello world", merges), merges))
}
