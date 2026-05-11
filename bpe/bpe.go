package bpe

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"

	"github.com/dlclark/regexp2"
)

var splitRegex = regexp2.MustCompile(GPT4_SPLIT_PATTERN, regexp2.None)

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

const VOCAB_SIZE = 256 + 1000
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
	w := 0
	n := len(list)
	for i := 0; i < n; i++ {
		if i < n-1 && list[i] == pair.First && list[i+1] == pair.Second {
			list[w] = index
			w++
			i++ 
		} else {
			list[w] = list[i]
			w++
		}
	}
	return list[:w]
}

func (bpe *BPETokenizer) stats(tokens []int) map[Pair]int {
	n := len(tokens)
	if n < 2 {
		return map[Pair]int{}
	}

	workers := runtime.GOMAXPROCS(0)
	// Below ~64k tokens the sequential path wins (sync overhead dominates).
	if workers < 2 || n < 64*1024 {
		m := make(map[Pair]int, n/8)
		for i := 0; i < n-1; i++ {
			m[Pair{tokens[i], tokens[i+1]}]++
		}
		return m
	}

	chunk := n / workers
	partials := make([]map[Pair]int, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if w == workers-1 {
			end = n
		}
		// Include the boundary pair (tokens[end-1], tokens[end]) so we don't miss
		// pairs that straddle chunk boundaries.
		if end < n {
			end++
		}
		wg.Add(1)
		go func(idx, s, e int) {
			defer wg.Done()
			local := make(map[Pair]int, (e-s)/8)
			for i := s; i < e-1; i++ {
				local[Pair{tokens[i], tokens[i+1]}]++
			}
			partials[idx] = local
		}(w, start, end)
	}
	wg.Wait()

	out := partials[0]
	for i := 1; i < workers; i++ {
		for p, c := range partials[i] {
			out[p] += c
		}
	}
	return out
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

/**
 * Tokenize text into bytes
 * 1. Split text into lines
 * 2. For each line, split into chunks using GPT4_SPLIT_PATTERN
 * 3. For each chunk, convert bytes to int
 * 4. Add newline token if not the last line
**/
func (bpe *BPETokenizer) Tokenize(text string) []int {
	if text == "" {
		fmt.Println("Tokenize: empty input, nothing to do")
		return []int{}
	}

	lines := strings.Split(text, "\n")
	lineTokens := make([][]int, len(lines))

	workers := runtime.GOMAXPROCS(0)
	if workers > len(lines) {
		workers = len(lines)
	}

	fmt.Printf("Tokenize: starting (%d lines, %d workers)\n", len(lines), workers)

	jobs := make(chan int, workers*2)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				lineTokens[i] = tokenizeLine(lines[i], i < len(lines)-1)
			}
		}()
	}
	for i := range lines {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	total := 0
	for _, t := range lineTokens {
		total += len(t)
	}
	allTokens := make([]int, 0, total)
	for _, t := range lineTokens {
		allTokens = append(allTokens, t...)
	}

	fmt.Printf("Tokenize: finished (%d lines -> %d tokens)\n", len(lines), len(allTokens))
	return allTokens
}

func tokenizeLine(lineText string, appendNewline bool) []int {
	if lineText == "" {
		if appendNewline {
			return []int{int('\n')}
		}
		return nil
	}

	out := make([]int, 0, len(lineText)+1)
	start := 0
	for start < len(lineText) {
		match, err := splitRegex.FindStringMatch(lineText[start:])
		if err != nil || match == nil {
			break
		}
		matched := match.String()
		for i := 0; i < len(matched); i++ {
			out = append(out, int(matched[i]))
		}
		start += match.Index + len(matched)
	}
	if appendNewline {
		out = append(out, int('\n'))
	}
	return out
}

/**
 * Decode tokens into text
 * 1. Create a local copy of idToToken
 * 2. For each merge, update local copy with merged tokens
 * 3. For each token, convert to byte and append to result
**/
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

/**
 * Encode text into tokens
 * 1. Tokenize text into bytes
 * 2. For each merge, apply merge to tokens
**/
func (bpe *BPETokenizer) Encode(text string) []int {
	tokens := bpe.Tokenize(text)

	for _, m := range bpe.Merges {
		tokens = bpe.merge(tokens, m.Pair, m.Index)
	}

	return tokens
}

func (bpe *BPETokenizer) Train(text string) {
	fmt.Println("Starting Training")
	tokens := bpe.Tokenize(text)
	numOfMerges := VOCAB_SIZE - 256
	for i := 0; i < numOfMerges; i++ {
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

		fmt.Printf("Merge %d/%d: (%d, %d) %q + %q -> %d %q (count=%d)\n",
			i+1, numOfMerges,
			maxUsedPair.First, maxUsedPair.Second,
			firstToken, secondToken,
			idx, mergedToken,
			statsMap[maxUsedPair])

		tokens = bpe.merge(tokens, maxUsedPair, idx)
		bpe.Merges = append(bpe.Merges, Merge{maxUsedPair, idx})
	}
	fmt.Println("Finished Training")
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
