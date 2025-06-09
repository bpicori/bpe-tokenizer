package main

import (
	"fmt"

	"github.com/dlclark/regexp2"
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
const GPT4_SPLIT_PATTERN = `(?i:'[sdmt]|'ll|'ve|'re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+`


type BPETokenizer struct {
	vocab     map[string]int
	idToToken map[int]string
	vocabSize int
	merges    []Merge
}

func NewBPETokenizer() *BPETokenizer {
	return &BPETokenizer{
		vocab:     make(map[string]int),
		idToToken: make(map[int]string),
		vocabSize: 0,
		merges:    []Merge{},
	}
}

func (bpe *BPETokenizer) Tokenize(text string) []int {
	re := regexp2.MustCompile(GPT4_SPLIT_PATTERN, regexp2.None)
	tokens := []int{}
	start := 0

	for start < len(text) {
		match, err := re.FindStringMatch(text[start:])
		if err != nil || match == nil {
			break
		}

		matched := match.String()
		if _, exists := bpe.vocab[matched]; !exists {
			bpe.vocab[matched] = bpe.vocabSize
			bpe.idToToken[bpe.vocabSize] = matched
			bpe.vocabSize++
		}
		tokens = append(tokens, bpe.vocab[matched])
		start += match.Index + len(matched)
	}

	return tokens
}

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

func (bpe *BPETokenizer) Decode(tokens []int) string {
	localVocab := make(map[int]string)

	for id, tok := range bpe.idToToken {
		localVocab[id] = tok
	}

	for _, merge := range bpe.merges {
		first := localVocab[merge.Pair.First]
		second := localVocab[merge.Pair.Second]
		localVocab[merge.Index] = first + second
	}

	var result string
	for _, token := range tokens {
		result += localVocab[token]
	}
	return result
}

func (bpe *BPETokenizer) Encode(text string) []int {
	tokens := bpe.Tokenize(text)

	for _, m := range bpe.merges {
		tokens = merge(tokens, m.Pair, m.Index)
	}

	return tokens
}

func (bpe *BPETokenizer) BuildVocab(text string) {
	tokens := bpe.Tokenize(text)
	numOfMerges := VOCAB_SIZE - 256
	for i := 0; i < numOfMerges; i++ {
		statsMap := stats(tokens)
		maxUsedPair := mostFrequentPair(statsMap)
		idx := 256 + i
		tokens = merge(tokens, maxUsedPair, idx)
		bpe.merges = append(bpe.merges, Merge{maxUsedPair, idx})
	}
}

func main() {
	trainingText := `Byte-pair encoding[1][2] (also known as BPE, or digram coding)[3] is an algorithm, first described in 1994 by Philip Gage, for encoding strings of text into smaller strings by creating and using a translation table.[4] A slightly modified version of the algorithm is used in large language model tokenizers.The original version of the algorithm focused on compression. It replaces the highest-frequency pair of bytes with a new byte that was not contained in the initial dataset. A lookup table of the replacements is required to rebuild the initial dataset. The modified version builds "tokens" (units of recognition) that match varying amounts of source text, from single characters (including single digits or single punctuation marks) to whole words (even long compound words).[5][6][7]`

	bpe := NewBPETokenizer()
	bpe.BuildVocab(trainingText)

	fmt.Println(bpe.Decode(bpe.Encode("hello world")))
}
