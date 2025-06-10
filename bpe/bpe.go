package bpe

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"sync"

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

const VOCAB_SIZE = 1000 * 10
const GPT4_SPLIT_PATTERN = `(?i:'[sdmt]|'ll|'ve|'re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+`

type BPETokenizer struct {
	vocab     map[string]int // {hello: 0, world: 1, ...} - used to check if a word is already tokenized
	idToToken map[int]string // {0: hello, 1: world, ...} - used to decode tokens
	vocabSize int
	Merges    []Merge
}

func NewBPETokenizer() *BPETokenizer {
	tokenizer := &BPETokenizer{
		Merges:    []Merge{},
		vocab:     make(map[string]int),
		idToToken: make(map[int]string),
		vocabSize: 256,
	}

	for i := 0; i < 256; i++ {
		byteStr := string([]byte{byte(i)})
		tokenizer.vocab[byteStr] = i
		tokenizer.idToToken[i] = byteStr
	}

	return tokenizer
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

	for i := 0; i < len(tokens)-1; i++ {
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
	if text == "" {
		return []int{}
	}

	lines := strings.Split(text, "\n")
	
	resultChan := make(chan []int, len(lines))
	var wg sync.WaitGroup
	
	for i, line := range lines {
		wg.Add(1)
		go func(lineNum int, lineText string) {
			defer wg.Done()
			
			if lineText == "" {
				resultChan <- []int{} 
				return
			}
			
			var lineTokens []int
			re := regexp2.MustCompile(GPT4_SPLIT_PATTERN, regexp2.None)
			start := 0
			
			for start < len(lineText) {
				match, err := re.FindStringMatch(lineText[start:])
				if err != nil || match == nil {
					break
				}
				
				matched := match.String()
				chunkBytes := []byte(matched)
				for _, b := range chunkBytes {
					lineTokens = append(lineTokens, int(b))
				}
				
				start += match.Index + len(matched)
			}
			
			// Add newline token if not the last line
			if lineNum < len(lines)-1 {
				lineTokens = append(lineTokens, int('\n'))
			}
			
			resultChan <- lineTokens
			fmt.Printf("Tokenized line %d/%d\n", lineNum+1, len(lines))
		}(i, line)
	}
	
	go func() {
		wg.Wait()
		close(resultChan)
	}()
	
	var allTokens []int
	for tokens := range resultChan {
		allTokens = append(allTokens, tokens...)
	}
	
	return allTokens
}

func (bpe *BPETokenizer) Decode(tokens []int) string {
	if len(tokens) == 0 {
		return ""
	}

	localVocab := make(map[int]string)
	for id, tok := range bpe.idToToken {
		localVocab[id] = tok
	}

	for _, merge := range bpe.Merges {
		first := localVocab[merge.Pair.First]
		second := localVocab[merge.Pair.Second]
		localVocab[merge.Index] = first + second
	}

	var result []byte
	for _, token := range tokens {
		if tokenStr, exists := localVocab[token]; exists {
			result = append(result, []byte(tokenStr)...)
		}
	}

	return string(result)
}

func (bpe *BPETokenizer) Encode(text string) []int {
	tokens := bpe.Tokenize(text)

	for _, m := range bpe.Merges {
		tokens = bpe.merge(tokens, m.Pair, m.Index)
	}

	return tokens
}

func (bpe *BPETokenizer) Train(text string) {
	tokens := bpe.Tokenize(text)
	numOfMerges := VOCAB_SIZE - 256
	for i := 0; i < numOfMerges; i++ {
		fmt.Println("Training progress:", i, "/", numOfMerges)
		statsMap := bpe.stats(tokens)
		if len(statsMap) == 0 {
			for j := i; j < numOfMerges; j++ {
				dummyPair := Pair{First: 0, Second: 0}
				idx := 256 + j
				bpe.Merges = append(bpe.Merges, Merge{dummyPair, idx})
			}
			break
		}

		idx := 256 + i
		maxUsedPair := bpe.mostFrequentPair(statsMap)

		firstToken := bpe.idToToken[maxUsedPair.First]
		secondToken := bpe.idToToken[maxUsedPair.Second]
		mergedToken := firstToken + secondToken

		bpe.vocab[mergedToken] = idx
		bpe.idToToken[idx] = mergedToken

		tokens = bpe.merge(tokens, maxUsedPair, idx)
		bpe.Merges = append(bpe.Merges, Merge{maxUsedPair, idx})
	}
}

func (bpe *BPETokenizer) Save() {
	fileName := "vocab.model"

	file, err := os.Create(fmt.Sprintf("./%s", fileName)) // creates or truncates
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	for _, m := range bpe.Merges {
		fmt.Fprintln(file, m.Pair.String(), m.Index)
	}

	fmt.Println("Vocab saved to", fileName)
}

func (bpe *BPETokenizer) Load() {
	fileName := "vocab.model"

	file, err := os.Open(fmt.Sprintf("./%s", fileName))
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	// Reset and reinitialize base vocabulary
	bpe.vocab = make(map[string]int)
	bpe.idToToken = make(map[int]string)
	bpe.vocabSize = 256
	bpe.Merges = []Merge{}

	// Initialize base vocabulary with all 256 possible bytes
	for i := 0; i < 256; i++ {
		byteStr := string([]byte{byte(i)})
		bpe.vocab[byteStr] = i
		bpe.idToToken[i] = byteStr
	}

	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		var first, second, index int
		_, err := fmt.Sscanf(line, "%d-%d %d", &first, &second, &index)
		if err != nil {
			panic(err)
		}

		merge := Merge{
			Pair:  Pair{First: first, Second: second},
			Index: index,
		}
		bpe.Merges = append(bpe.Merges, merge)

		if first < len(bpe.idToToken) && second < len(bpe.idToToken) {
			firstToken := bpe.idToToken[first]
			secondToken := bpe.idToToken[second]
			mergedToken := firstToken + secondToken
			bpe.vocab[mergedToken] = index
			bpe.idToToken[index] = mergedToken
		}
	}
}
