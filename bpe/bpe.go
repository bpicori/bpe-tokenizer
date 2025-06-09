package bpe

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
	vocab     map[string]int // {hello: 0, world: 1, ...} - used to check if a word is already tokenized
	idToToken map[int]string // {0: hello, 1: world, ...} - used to decode tokens
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

func (bpe *BPETokenizer) merge(list []int, pair Pair, index int) []int {
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

func (bpe *BPETokenizer) stats(tokens []int) map[Pair]int {
	m := make(map[Pair]int)

	for i := range len(tokens) - 1 {
		curr := tokens[i]
		next := tokens[i+1]
		pair := Pair{curr, next}
		m[pair]++
	}

	return m
}

func (bpe *BPETokenizer) mostFrequentPair(m map[Pair]int) Pair {
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
		tokens = bpe.merge(tokens, m.Pair, m.Index)
	}

	return tokens
}

func (bpe *BPETokenizer) Train(text string) {
	tokens := bpe.Tokenize(text)
	numOfMerges := VOCAB_SIZE - 256
	for i := 0; i < numOfMerges; i++ {
		statsMap := bpe.stats(tokens)
		maxUsedPair := bpe.mostFrequentPair(statsMap)
		idx := 256 + i
		tokens = bpe.merge(tokens, maxUsedPair, idx)
		bpe.merges = append(bpe.merges, Merge{maxUsedPair, idx})
	}
}
