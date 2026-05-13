package bpe

import (
	"bufio"
	"fmt"
	"io"
	"maps"
	"os"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
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

// Word is a unique chunk and its frequency. Training counts each pair once
// and weights by Count, e.g. " the" x 1_000_000 contributes (' ', 't')+=1M.
type Word struct {
	Tokens []int
	Count  int
}

// VocabSize = 256 bytes + specials + learned merges.
const VocabSize = 20_000

// GPT4SplitPattern pre-tokenization regex. RE2 has no lookaround, so the original
// `\s+(?!\S)` is rewritten as `\s+$` — equivalent because we tokenize one
// line at a time.
const GPT4SplitPattern = `(?i:'[sdmt]|'ll|'ve|'re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+$|\s+`

type BPETokenizer struct {
	vocab         map[string]int // "the" -> 1234
	idToToken     map[int]string // 1234 -> "the"
	vocabSize     int
	Merges        []Merge
	specialTokens []string // fixed IDs [256, mergeStart); never merged
	mergeStart    int      // first ID for learned merges
	splitRegex    *regexp.Regexp
}

// NewBPETokenizer seeds the vocab with 256 byte tokens and the given specials.
// IDs: bytes [0,256), specials [256, 256+len(specials)), merges [mergeStart, ...).
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

	for i := range 256 {
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

// Tokenize splits text into byte-level chunks using the GPT-4 regex.
// Each chunk is bytes (0–255); pairs never cross chunk boundaries.
//
// Example "cat, cat" -> [{99,97,116}, {44}, {32,99,97,116}]
//
//	(«cat»)       («,») (« cat»)
//
// Specials are emitted as singleton chunks holding their reserved ID, so
// BPE can never merge into or across them. Digit runs cap at 3 ("12345"
// -> {49,50,51}+{52,53}). Newlines are inserted as chunk {10}.
func (bpe *BPETokenizer) Tokenize(text string) [][]int {
	if text == "" {
		fmt.Println("Tokenize: empty input, nothing to do")
		return [][]int{}
	}

	lines := strings.Split(text, "\n")
	lineChunks := make([][][]int, len(lines))

	workers := min(runtime.GOMAXPROCS(0), len(lines))

	fmt.Printf("Tokenize: starting (%d lines, %d workers)\n", len(lines), workers)

	jobs := make(chan int, workers*2)
	var wg sync.WaitGroup
	for range workers {
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

// Encode pre-splits via Tokenize, then replays merges within each chunk.
// Example "cat" with merges {(c,a)->500, (500,t)->501} -> [501].
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

// Decode reconstructs text by replaying merges to materialise each ID's bytes.
// Example: idToToken = {99:"c", 97:"a", 116:"t", 500:"ca", 501:"cat"} (501 came
// from merge (500,116)) -> Decode([501]) == "cat".
func (bpe *BPETokenizer) Decode(tokens []int) string {
	if len(tokens) == 0 {
		return ""
	}

	localVocab := make(map[int]string)
	maps.Copy(localVocab, bpe.idToToken)

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

// Train learns BPE merges from corpus r: each step picks the corpus-wide most
// frequent pair, replaces it with a new token ID, and repeats until vocab is
// full. Pair stats are maintained incrementally (see pairStats).
func (bpe *BPETokenizer) Train(r io.Reader) {
	fmt.Println("Starting Training")
	words := bpe.countWordFrequencies(r)
	fmt.Printf("Train: %d unique chunks after collapse\n", len(words))

	stats := newPairStats(words)
	numOfMerges := VocabSize - bpe.mergeStart

	for i := range numOfMerges {
		newID := bpe.mergeStart + i

		bestPair, count, ok := stats.best()
		if !ok {
			fmt.Printf("Train: corpus exhausted after %d merges (target %d)\n", i, numOfMerges)
			break
		}

		bpe.recordMerge(bestPair, newID, count, i+1, numOfMerges)

		// Rewrite every word that contains bestPair. Snapshot indices first
		// because removeWord mutates stats.members[bestPair]. Subtract-then-add
		// handles overlapping pairs like (a,a) in "aaaa" naturally.
		for _, wi := range stats.wordsContaining(bestPair) {
			stats.removeWord(wi, words[wi])
			words[wi].Tokens = bpe.merge(words[wi].Tokens, bestPair, newID)
			stats.addWord(wi, words[wi])
		}
	}
	fmt.Println("Finished Training")
}

// recordMerge adds the merged token to the vocab and Merges list, and logs it.
// Example: pair=(99,97) "c"+"a" -> newID=500 "ca".
func (bpe *BPETokenizer) recordMerge(p Pair, newID, count, step, total int) {
	first := bpe.idToToken[p.First]
	second := bpe.idToToken[p.Second]
	merged := first + second

	bpe.vocab[merged] = newID
	bpe.idToToken[newID] = merged
	bpe.Merges = append(bpe.Merges, Merge{p, newID})

	fmt.Printf("Merge %d/%d: (%d, %d) %q + %q -> %d %q (count=%d)\n",
		step, total, p.First, p.Second, first, second, newID, merged, count)
}

// Save writes specials and learned merges to ./vocab.model.
// Format: "S <token>" lines first, then "<first>-<second> <index>" per merge.
func (bpe *BPETokenizer) Save() {
	fileName := "vocab.model"

	file, err := os.Create(fmt.Sprintf("./%s", fileName))
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

// Load restores the tokenizer from ./vocab.model, replaying each merge to
// rebuild idToToken (e.g. merge (99,97)->500 sets idToToken[500] = "ca").
func (bpe *BPETokenizer) Load() {
	fileName := "vocab.model"

	file, err := os.Open(fmt.Sprintf("./%s", fileName))
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	bpe.vocab = make(map[string]int)
	bpe.idToToken = make(map[int]string)
	bpe.vocabSize = 256
	bpe.Merges = []Merge{}
	bpe.specialTokens = nil
	bpe.mergeStart = 256
	bpe.splitRegex = buildSplitRegex(nil)

	for i := range 256 {
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

// --- internals ---

// buildSplitRegex prepends specials as top-priority alternatives, longest
// first so "<|endoftext|>" wins over "<|end|>" (Go regexp is leftmost-first).
func buildSplitRegex(specialTokens []string) *regexp.Regexp {
	if len(specialTokens) == 0 {
		return regexp.MustCompile(GPT4SplitPattern)
	}
	sorted := append([]string{}, specialTokens...)
	sort.SliceStable(sorted, func(i, j int) bool { return len(sorted[i]) > len(sorted[j]) })
	alts := make([]string, 0, len(sorted))
	for _, t := range sorted {
		if t != "" {
			alts = append(alts, regexp.QuoteMeta(t))
		}
	}
	pattern := "(?:" + strings.Join(alts, "|") + ")|" + GPT4SplitPattern
	return regexp.MustCompile(pattern)
}

func (bpe *BPETokenizer) tokenizeLine(lineText string, appendNewline bool) [][]int {
	if lineText == "" {
		if appendNewline {
			return [][]int{{int('\n')}}
		}
		return nil
	}

	matches := bpe.splitRegex.FindAllStringIndex(lineText, -1)
	chunks := make([][]int, 0, len(matches)+1)
	for _, m := range matches {
		matched := lineText[m[0]:m[1]]
		// Special token matched -> emit singleton chunk with its reserved ID.
		if id, ok := bpe.vocab[matched]; ok && id >= 256 && id < bpe.mergeStart {
			chunks = append(chunks, []int{id})
			continue
		}
		chunk := make([]int, len(matched))
		for i := 0; i < len(matched); i++ {
			chunk[i] = int(matched[i])
		}
		chunks = append(chunks, chunk)
	}
	if appendNewline {
		chunks = append(chunks, []int{int('\n')})
	}
	return chunks
}

// merge replaces every adjacent (pair.First, pair.Second) with index, in place.
// Example: merge([1,2,3,1,2], {1,2}, 99) -> [99,3,99].
// Overlaps don't double-count: merge([1,1,1], {1,1}, 99) -> [99,1].
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

// countWordFrequencies streams r line-by-line and returns each unique chunk
// with how many times it occurred. Drops length-1 chunks (no pairs, also
// filters specials). Memory is O(unique chunks), not O(corpus bytes).
// Example corpus "cat cat hat" -> [{" cat",2}, {"cat",1}, {" hat",1}].
func (bpe *BPETokenizer) countWordFrequencies(r io.Reader) []Word {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 1<<16), 64<<20)

	workers := max(runtime.GOMAXPROCS(0), 1)

	type local struct {
		index map[string]int
		words []Word
	}
	locals := make([]local, workers)
	for i := range locals {
		locals[i] = local{index: make(map[string]int, 1<<14)}
	}

	lineCh := make(chan string, workers*4)
	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func(slot int) {
			defer wg.Done()
			keyBuf := make([]byte, 0, 32)
			l := &locals[slot]
			for line := range lineCh {
				for _, c := range bpe.tokenizeLine(line, false) {
					if len(c) < 2 {
						continue
					}
					keyBuf = keyBuf[:0]
					for _, id := range c {
						keyBuf = append(keyBuf, byte(id>>24), byte(id>>16), byte(id>>8), byte(id))
					}
					key := string(keyBuf)
					if idx, ok := l.index[key]; ok {
						l.words[idx].Count++
					} else {
						l.index[key] = len(l.words)
						l.words = append(l.words, Word{Tokens: c, Count: 1})
					}
				}
			}
		}(w)
	}

	for scanner.Scan() {
		lineCh <- scanner.Text()
	}
	close(lineCh)
	wg.Wait()
	if err := scanner.Err(); err != nil {
		panic(err)
	}

	idx0 := locals[0].index
	words := locals[0].words
	for i := 1; i < workers; i++ {
		for key, j := range locals[i].index {
			if dst, ok := idx0[key]; ok {
				words[dst].Count += locals[i].words[j].Count
			} else {
				idx0[key] = len(words)
				words = append(words, locals[i].words[j])
			}
		}
	}
	return words
}

// pairStats tracks adjacent-pair frequencies across the corpus and which
// words each pair lives in, so Train can update both incrementally.
//
//	counts[(' ','t')]  = 1_000_000           // total weighted count
//	members[(' ','t')] = {wi1, wi2, ...}     // words containing the pair
type pairStats struct {
	counts  map[Pair]int
	members map[Pair]map[int]struct{}
}

func newPairStats(words []Word) *pairStats {
	s := &pairStats{
		counts:  make(map[Pair]int, len(words)),
		members: make(map[Pair]map[int]struct{}, len(words)),
	}
	for wi := range words {
		s.addWord(wi, words[wi])
	}
	return s
}

// addWord registers every adjacent pair in w as a contribution from word wi.
func (s *pairStats) addWord(wi int, w Word) {
	for j := 0; j < len(w.Tokens)-1; j++ {
		p := Pair{w.Tokens[j], w.Tokens[j+1]}
		s.counts[p] += w.Count
		set, ok := s.members[p]
		if !ok {
			set = make(map[int]struct{})
			s.members[p] = set
		}
		set[wi] = struct{}{}
	}
}

// removeWord undoes addWord: subtracts w's pair contributions and drops the
// word from each pair's member set. Empty entries are pruned.
func (s *pairStats) removeWord(wi int, w Word) {
	for j := 0; j < len(w.Tokens)-1; j++ {
		p := Pair{w.Tokens[j], w.Tokens[j+1]}
		s.counts[p] -= w.Count
		if s.counts[p] <= 0 {
			delete(s.counts, p)
		}
		if set, ok := s.members[p]; ok {
			delete(set, wi)
			if len(set) == 0 {
				delete(s.members, p)
			}
		}
	}
}

// best returns the most frequent pair, its count, and ok=false if empty.
func (s *pairStats) best() (Pair, int, bool) {
	var best Pair
	max := 0
	for p, c := range s.counts {
		if c > max {
			max = c
			best = p
		}
	}
	return best, max, max > 0
}

// wordsContaining returns a snapshot of the word indices that hold pair p.
func (s *pairStats) wordsContaining(p Pair) []int {
	set := s.members[p]
	out := make([]int, 0, len(set))
	for wi := range set {
		out = append(out, wi)
	}
	return out
}
