package main

import "fmt"

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

func merge(list []int, pair Pair, index int) []int {
	newList := []int{}

	for i := 0; i < len(list)-1; i++ {
		first := list[i]
		second := list[i+1]
		if first == pair.First && second == pair.Second {
			newList = append(newList, index)
			i++
		} else {
			newList = append(newList, first)
		}
	}

	return newList
}

func stats(tokens []int) map[Pair]int {
	m := make(map[Pair]int)

	for i := 0; i < len(tokens)-1; i++ {
		curr := tokens[i]
		next := tokens[i+1]
		pair := Pair{curr, next}
		m[pair]++
	}

	return m
}

func max(m map[Pair]int) Pair {
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
	vocab := make(map[int]string)
	for i := 0; i < 256; i++ {
		vocab[i] = string(rune(i))
	}
	for _, merge := range merges {
		vocab[merge.Index] = fmt.Sprintf("%s%s", vocab[merge.Pair.First], vocab[merge.Pair.Second])
	}

	decoded := ""
	for _, token := range tokens {
		decoded = decoded + vocab[token]
	}

	return decoded
}

func main() {
	text := `Byte-pair encoding[1][2] (also known as BPE, or digram coding)[3] is an algorithm, first described in 1994 by Philip Gage, for encoding strings of text into smaller strings by creating and using a translation table.[4] A slightly modified version of the algorithm is used in large language model tokenizers.The original version of the algorithm focused on compression. It replaces the highest-frequency pair of bytes with a new byte that was not contained in the initial dataset. A lookup table of the replacements is required to rebuild the initial dataset. The modified version builds "tokens" (units of recognition) that match varying amounts of source text, from single characters (including single digits or single punctuation marks) to whole words (even long compound words).[5][6][7]`
	tokens := make([]int, len(text))
	for i, b := range []byte(text) {
		tokens[i] = int(b)
	}

	vocabSize := 279
	numOfMerges := vocabSize - 256

	// copy original
	original := make([]int, len(tokens))
	copy(original, tokens)

	merges := []Merge{}
	for i := 0; i < numOfMerges; i++ {
		stats := stats(tokens)
		maxUsedPair := max(stats)
		idx := 256 + i
		fmt.Printf("Merging pair (%d, %d) into token %d \n", maxUsedPair.First, maxUsedPair.Second, idx)
		tokens = merge(tokens, maxUsedPair, idx)
		merges = append(merges, Merge{maxUsedPair, idx})
	}

	fmt.Println("Original Length:", len(original))
	fmt.Println("New Length:", len(tokens))
	fmt.Println("Compress rate:", float64(len(original))/float64(len(tokens)))

	decoded := decode(tokens, merges)
	fmt.Println(decoded)

	fmt.Println("==================")
	fmt.Println(decode([]int{128}, []Merge{}))
}
