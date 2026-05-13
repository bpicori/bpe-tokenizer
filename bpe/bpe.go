package bpe

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"sort"
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

// VOCAB_SIZE is the target *total* vocabulary size: 256 base byte tokens, plus
// any special tokens, plus learned BPE merges. Train computes the number of
// merges as VOCAB_SIZE - mergeStart, where mergeStart already accounts for
// the byte range and the special-token range (set in NewBPETokenizer).
const VOCAB_SIZE = 10_000
const GPT4_SPLIT_PATTERN = `(?i:'[sdmt]|'ll|'ve|'re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+`

type BPETokenizer struct {
	vocab         map[string]int     // {hello: 0, world: 1, ...} - used to check if a word is already tokenized
	idToToken     map[int]string     // {0: hello, 1: world, ...} - used to decode tokens
	vocabSize     int
	Merges        []Merge
	specialTokens []string           // reserved tokens with fixed IDs starting at 256; never merged
	mergeStart    int                // first ID available to learned BPE merges (256 + len(specialTokens))
	splitRegex    *regexp2.Regexp    // pre-tokenization regex, with specials prepended as alternatives
}

// NewBPETokenizer constructs a tokenizer with the 256 raw-byte base vocabulary.
// Special tokens get fixed IDs 256, 257, ... and are excluded from BPE merging.
// Learned merges start at mergeStart and grow upward.
func NewBPETokenizer(specialTokens []string) *BPETokenizer {
	tokenizer := &BPETokenizer{
		Merges:        []Merge{},
		vocab:         make(map[string]int),
		idToToken:     make(map[int]string),
		vocabSize:     256 + len(specialTokens),
		specialTokens: append([]string{}, specialTokens...),
		mergeStart:    256 + len(specialTokens),
		splitRegex:    buildSplitRegex(specialTokens),
	}

	for i := 0; i < 256; i++ {
		byteStr := string([]byte{byte(i)})
		tokenizer.vocab[byteStr] = i
		tokenizer.idToToken[i] = byteStr
	}

	for i, tok := range specialTokens {
		id := 256 + i
		tokenizer.vocab[tok] = id
		tokenizer.idToToken[id] = tok
	}

	return tokenizer
}

// buildSplitRegex prepends each special token as a top-priority alternative to
// the GPT-4 split pattern. Longer specials come first so that overlapping
// tokens (e.g. "<|end|>" vs "<|endoftext|>") prefer the longer match — regexp2
// is .NET-flavored, so alternation is leftmost-first, not POSIX-longest.
func buildSplitRegex(specialTokens []string) *regexp2.Regexp {
	if len(specialTokens) == 0 {
		return regexp2.MustCompile(GPT4_SPLIT_PATTERN, regexp2.None)
	}
	sorted := append([]string{}, specialTokens...)
	sort.SliceStable(sorted, func(i, j int) bool { return len(sorted[i]) > len(sorted[j]) })
	alts := make([]string, 0, len(sorted))
	for _, t := range sorted {
		if t != "" {
			alts = append(alts, regexp2.Escape(t))
		}
	}
	pattern := "(?:" + strings.Join(alts, "|") + ")|" + GPT4_SPLIT_PATTERN
	return regexp2.MustCompile(pattern, regexp2.None)
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

// stats counts adjacent-pair frequencies *within* each chunk.
// Pairs that would span a chunk boundary are never counted — that's the whole
// reason Tokenize returns [][]int instead of a flat []int. See Tokenize's doc.
func (bpe *BPETokenizer) stats(chunks [][]int) map[Pair]int {
	totalLen := 0
	for _, c := range chunks {
		totalLen += len(c)
	}
	if totalLen < 2 {
		return map[Pair]int{}
	}

	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || totalLen < 64*1024 || len(chunks) < workers {
		m := make(map[Pair]int, totalLen/8)
		for _, chunk := range chunks {
			for i := 0; i < len(chunk)-1; i++ {
				m[Pair{chunk[i], chunk[i+1]}]++
			}
		}
		return m
	}

	n := len(chunks)
	per := n / workers
	partials := make([]map[Pair]int, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * per
		end := start + per
		if w == workers-1 {
			end = n
		}
		wg.Add(1)
		go func(idx, s, e int) {
			defer wg.Done()
			local := make(map[Pair]int)
			for i := s; i < e; i++ {
				chunk := chunks[i]
				for j := 0; j < len(chunk)-1; j++ {
					local[Pair{chunk[j], chunk[j+1]}]++
				}
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

// Tokenize splits text into byte-level chunks bounded by GPT4_SPLIT_PATTERN.
//
// Returns [][]int where each inner slice is one regex chunk, expanded into its
// raw bytes (0–255). Chunk boundaries are preserved on purpose: downstream BPE
// (stats, merge, Encode) iterates *inside* each chunk only, so a pair can
// never span the boundary the regex drew.
//
// Why this matters — worked example. Input: "cat, cat"
//
//	regex chunks:    «cat»          «,»     « cat»
//	returned:        {99,97,116}    {44}    {32,99,97,116}
//
// During training, stats counts pairs *within* each chunk:
//
//	from «cat»:     (c,a)=1, (a,t)=1
//	from «,»:       (nothing — single byte)
//	from « cat»:    ( ,c)=1, (c,a)=1, (a,t)=1
//
// Numbers: the regex caps digit runs at 3, so "12345" becomes two chunks
// {49,50,51} and {52,53} — BPE can never produce one giant "12345" token.
//
// Newlines: text is split on '\n' first, and a single-byte chunk {10} is
// inserted between lines so newlines act as hard boundaries too.
// Tokenize splits text into byte-level chunks bounded by GPT4_SPLIT_PATTERN.
// Special tokens are emitted as singleton chunks holding their reserved ID:
// the regex matches them as standalone alternatives (they sit in front of the
// pattern), and the per-chunk post-check below recognises the matched text
// and substitutes the reserved ID instead of decomposing it to bytes.
//
// Because each special lives in its own length-1 chunk, `stats` can never
// pair it with anything → BPE cannot merge across or into a special token.
func (bpe *BPETokenizer) Tokenize(text string) [][]int {
	if text == "" {
		fmt.Println("Tokenize: empty input, nothing to do")
		return [][]int{}
	}

	lines := strings.Split(text, "\n")
	lineChunks := make([][][]int, len(lines))

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
				lineChunks[i] = bpe.tokenizeLine(lines[i], i < len(lines)-1)
			}
		}()
	}
	for i := range lines {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	totalChunks := 0
	totalBytes := 0
	for _, lc := range lineChunks {
		totalChunks += len(lc)
		for _, c := range lc {
			totalBytes += len(c)
		}
	}
	all := make([][]int, 0, totalChunks)
	for _, lc := range lineChunks {
		all = append(all, lc...)
	}

	fmt.Printf("Tokenize: finished (%d lines -> %d chunks, %d bytes)\n", len(lines), len(all), totalBytes)
	return all
}

func (bpe *BPETokenizer) tokenizeLine(lineText string, appendNewline bool) [][]int {
	if lineText == "" {
		if appendNewline {
			return [][]int{{int('\n')}}
		}
		return nil
	}

	var chunks [][]int
	start := 0
	for start < len(lineText) {
		match, err := bpe.splitRegex.FindStringMatch(lineText[start:])
		if err != nil || match == nil {
			break
		}
		matched := match.String()
		// If the regex matched a special token, emit its reserved ID as a
		// singleton chunk. Special IDs occupy [256, mergeStart); merge IDs
		// (>= mergeStart) cannot end up here because learned merges are never
		// added to the split regex.
		if id, ok := bpe.vocab[matched]; ok && id >= 256 && id < bpe.mergeStart {
			chunks = append(chunks, []int{id})
		} else {
			chunk := make([]int, 0, len(matched))
			for i := 0; i < len(matched); i++ {
				chunk = append(chunk, int(matched[i]))
			}
			chunks = append(chunks, chunk)
		}
		start += match.Index + len(matched)
	}
	if appendNewline {
		chunks = append(chunks, []int{int('\n')})
	}
	return chunks
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

// Encode text into tokens.
//
// Pre-split into chunks via Tokenize, apply every learned merge *inside* each
// chunk, then concatenate the chunks into one flat []int. Merges never glue
// bytes across chunk boundaries because each call to merge sees only one
// chunk's slice.
func (bpe *BPETokenizer) Encode(text string) []int {
	chunks := bpe.Tokenize(text)

	total := 0
	for i := range chunks {
		for _, m := range bpe.Merges {
			chunks[i] = bpe.merge(chunks[i], m.Pair, m.Index)
		}
		total += len(chunks[i])
	}

	out := make([]int, 0, total)
	for _, c := range chunks {
		out = append(out, c...)
	}
	return out
}

func (bpe *BPETokenizer) Train(text string) {
	fmt.Println("Starting Training")
	chunks := bpe.Tokenize(text)
	numOfMerges := VOCAB_SIZE - bpe.mergeStart
	for i := 0; i < numOfMerges; i++ {
		statsMap := bpe.stats(chunks)
		if len(statsMap) == 0 {
			for j := i; j < numOfMerges; j++ {
				dummyPair := Pair{First: 0, Second: 0}
				idx := bpe.mergeStart + j
				bpe.Merges = append(bpe.Merges, Merge{dummyPair, idx})
			}
			break
		}

		idx := bpe.mergeStart + i
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

		for j := range chunks {
			chunks[j] = bpe.merge(chunks[j], maxUsedPair, idx)
		}
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

	for _, tok := range bpe.specialTokens {
		fmt.Fprintf(file, "S %s\n", tok)
	}
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
	bpe.specialTokens = nil
	bpe.mergeStart = 256
	bpe.splitRegex = buildSplitRegex(nil)

	// Initialize base vocabulary with all 256 possible bytes
	for i := 0; i < 256; i++ {
		byteStr := string([]byte{byte(i)})
		bpe.vocab[byteStr] = i
		bpe.idToToken[i] = byteStr
	}

	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "S ") {
			tok := line[2:]
			id := 256 + len(bpe.specialTokens)
			bpe.specialTokens = append(bpe.specialTokens, tok)
			bpe.vocab[tok] = id
			bpe.idToToken[id] = tok
			bpe.vocabSize = id + 1
			bpe.mergeStart = id + 1
			bpe.splitRegex = buildSplitRegex(bpe.specialTokens)
			continue
		}
		var first, second, index int
		_, err := fmt.Sscanf(line, "%d-%d %d", &first, &second, &index)
		if err != nil {
			panic(err)
		}

		bpe.Merges = append(bpe.Merges, Merge{Pair{first, second}, index})

		if firstTok, ok1 := bpe.idToToken[first]; ok1 {
			if secondTok, ok2 := bpe.idToToken[second]; ok2 {
				merged := firstTok + secondTok
				bpe.vocab[merged] = index
				bpe.idToToken[index] = merged
			}
		}
	}
}
