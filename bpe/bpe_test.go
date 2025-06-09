package bpe

import (
	"reflect"
	"testing"
)

func TestNewBPETokenizer(t *testing.T) {
	tokenizer := NewBPETokenizer()

	if tokenizer == nil {
		t.Fatal("NewBPETokenizer() returned nil")
	}

	if tokenizer.vocab == nil {
		t.Error("vocab map should be initialized")
	}

	if tokenizer.idToToken == nil {
		t.Error("idToToken map should be initialized")
	}

	if tokenizer.vocabSize != 256 {
		t.Errorf("vocabSize should be 256 (base byte vocabulary), got %d", tokenizer.vocabSize)
	}

	// Check that base vocabulary is properly initialized with all 256 bytes
	if len(tokenizer.vocab) != 256 {
		t.Errorf("vocab should contain 256 base tokens, got %d", len(tokenizer.vocab))
	}

	if len(tokenizer.idToToken) != 256 {
		t.Errorf("idToToken should contain 256 base tokens, got %d", len(tokenizer.idToToken))
	}

	// Check a few specific byte mappings
	if tokenizer.vocab[string([]byte{0})] != 0 {
		t.Error("byte 0 should map to token 0")
	}
	if tokenizer.vocab[string([]byte{255})] != 255 {
		t.Error("byte 255 should map to token 255")
	}

	if tokenizer.Merges == nil {
		t.Error("merges slice should be initialized")
	}

	if len(tokenizer.Merges) != 0 {
		t.Errorf("merges slice should be empty, got length %d", len(tokenizer.Merges))
	}
}

func TestPairString(t *testing.T) {
	tests := []struct {
		name     string
		pair     Pair
		expected string
	}{
		{
			name:     "simple pair",
			pair:     Pair{First: 1, Second: 2},
			expected: "1-2",
		},
		{
			name:     "zero values",
			pair:     Pair{First: 0, Second: 0},
			expected: "0-0",
		},
		{
			name:     "large numbers",
			pair:     Pair{First: 256, Second: 257},
			expected: "256-257",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.pair.String()
			if result != tt.expected {
				t.Errorf("Pair.String() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestMerge(t *testing.T) {
	tokenizer := NewBPETokenizer()

	tests := []struct {
		name     string
		list     []int
		pair     Pair
		index    int
		expected []int
	}{
		{
			name:     "simple merge",
			list:     []int{1, 2, 3, 4},
			pair:     Pair{First: 1, Second: 2},
			index:    10,
			expected: []int{10, 3, 4},
		},
		{
			name:     "multiple merges",
			list:     []int{1, 2, 3, 1, 2, 5},
			pair:     Pair{First: 1, Second: 2},
			index:    10,
			expected: []int{10, 3, 10, 5},
		},
		{
			name:     "no merge found",
			list:     []int{1, 3, 4, 5},
			pair:     Pair{First: 1, Second: 2},
			index:    10,
			expected: []int{1, 3, 4, 5},
		},
		{
			name:     "merge at end",
			list:     []int{3, 4, 1, 2},
			pair:     Pair{First: 1, Second: 2},
			index:    10,
			expected: []int{3, 4, 10},
		},
		{
			name:     "empty list",
			list:     []int{},
			pair:     Pair{First: 1, Second: 2},
			index:    10,
			expected: []int{},
		},
		{
			name:     "single element",
			list:     []int{1},
			pair:     Pair{First: 1, Second: 2},
			index:    10,
			expected: []int{1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tokenizer.merge(tt.list, tt.pair, tt.index)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("merge() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestStats(t *testing.T) {
	tokenizer := NewBPETokenizer()

	tests := []struct {
		name     string
		tokens   []int
		expected map[Pair]int
	}{
		{
			name:   "simple sequence",
			tokens: []int{1, 2, 3, 4},
			expected: map[Pair]int{
				{1, 2}: 1,
				{2, 3}: 1,
				{3, 4}: 1,
			},
		},
		{
			name:   "repeated pairs",
			tokens: []int{1, 2, 1, 2, 3},
			expected: map[Pair]int{
				{1, 2}: 2,
				{2, 1}: 1,
				{2, 3}: 1,
			},
		},
		{
			name:     "single token",
			tokens:   []int{1},
			expected: map[Pair]int{},
		},
		{
			name:     "empty tokens",
			tokens:   []int{},
			expected: map[Pair]int{},
		},
		{
			name:   "two tokens",
			tokens: []int{1, 2},
			expected: map[Pair]int{
				{1, 2}: 1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tokenizer.stats(tt.tokens)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("stats() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestMostFrequentPair(t *testing.T) {
	tokenizer := NewBPETokenizer()

	tests := []struct {
		name     string
		stats    map[Pair]int
		expected Pair
	}{
		{
			name: "single most frequent",
			stats: map[Pair]int{
				{1, 2}: 3,
				{2, 3}: 1,
				{3, 4}: 2,
			},
			expected: Pair{1, 2},
		},
		{
			name: "tie - returns one of them",
			stats: map[Pair]int{
				{1, 2}: 2,
				{3, 4}: 2,
			},
			expected: Pair{}, // Will be one of the pairs, but we can't predict which
		},
		{
			name:     "empty stats",
			stats:    map[Pair]int{},
			expected: Pair{},
		},
		{
			name: "single pair",
			stats: map[Pair]int{
				{5, 6}: 1,
			},
			expected: Pair{5, 6},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tokenizer.mostFrequentPair(tt.stats)
			if tt.name == "tie - returns one of them" {
				// For ties, just check that the result is one of the valid pairs
				if (result != Pair{1, 2}) && (result != Pair{3, 4}) {
					t.Errorf("mostFrequentPair() = %v, want either {1, 2} or {3, 4}", result)
				}
			} else {
				if result != tt.expected {
					t.Errorf("mostFrequentPair() = %v, want %v", result, tt.expected)
				}
			}
		})
	}
}

func TestTokenize(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		validate func(*testing.T, *BPETokenizer, []int)
	}{
		{
			name: "simple text",
			text: "hello world",
			validate: func(t *testing.T, tokenizer *BPETokenizer, tokens []int) {
				if len(tokens) == 0 {
					t.Error("Expected non-empty tokens")
				}
				// Check that vocab was populated
				if len(tokenizer.vocab) == 0 {
					t.Error("Expected vocab to be populated")
				}
				// Check that idToToken was populated
				if len(tokenizer.idToToken) == 0 {
					t.Error("Expected idToToken to be populated")
				}
				// Check that vocabSize matches
				if tokenizer.vocabSize != len(tokenizer.vocab) {
					t.Errorf("vocabSize (%d) should match vocab length (%d)", tokenizer.vocabSize, len(tokenizer.vocab))
				}
			},
		},
		{
			name: "empty text",
			text: "",
			validate: func(t *testing.T, tokenizer *BPETokenizer, tokens []int) {
				if len(tokens) != 0 {
					t.Errorf("Expected empty tokens for empty text, got %v", tokens)
				}
			},
		},
		{
			name: "text with numbers",
			text: "hello 123 world",
			validate: func(t *testing.T, tokenizer *BPETokenizer, tokens []int) {
				if len(tokens) == 0 {
					t.Error("Expected non-empty tokens")
				}
				// Should have multiple tokens due to the pattern
				if len(tokens) < 2 {
					t.Error("Expected multiple tokens for mixed text")
				}
			},
		},
		{
			name: "repeated text",
			text: "hello hello",
			validate: func(t *testing.T, tokenizer *BPETokenizer, tokens []int) {
				if len(tokens) < 2 {
					t.Error("Expected at least 2 tokens")
				}
				// The same word should get the same token ID
				// We can't predict exact structure due to regex complexity, but vocab should be consistent
				if len(tokenizer.vocab) == 0 {
					t.Error("Expected vocab to be populated")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer := NewBPETokenizer()
			tokens := tokenizer.Tokenize(tt.text)
			tt.validate(t, tokenizer, tokens)
		})
	}
}

func TestDecode(t *testing.T) {
	tests := []struct {
		name     string
		setup    func(*BPETokenizer) []int
		validate func(*testing.T, string)
	}{
		{
			name: "simple decode",
			setup: func(tokenizer *BPETokenizer) []int {
				// Manually set up vocab
				tokenizer.vocab["hello"] = 0
				tokenizer.vocab[" "] = 1
				tokenizer.vocab["world"] = 2
				tokenizer.idToToken[0] = "hello"
				tokenizer.idToToken[1] = " "
				tokenizer.idToToken[2] = "world"
				tokenizer.vocabSize = 3
				return []int{0, 1, 2}
			},
			validate: func(t *testing.T, result string) {
				expected := "hello world"
				if result != expected {
					t.Errorf("Decode() = %q, want %q", result, expected)
				}
			},
		},
		{
			name: "empty tokens",
			setup: func(tokenizer *BPETokenizer) []int {
				return []int{}
			},
			validate: func(t *testing.T, result string) {
				if result != "" {
					t.Errorf("Decode() = %q, want empty string", result)
				}
			},
		},
		{
			name: "with merges",
			setup: func(tokenizer *BPETokenizer) []int {
				// Set up base vocab
				tokenizer.vocab["h"] = 0
				tokenizer.vocab["e"] = 1
				tokenizer.idToToken[0] = "h"
				tokenizer.idToToken[1] = "e"
				tokenizer.vocabSize = 2

				// Add a merge
				merge := Merge{
					Pair:  Pair{First: 0, Second: 1},
					Index: 256,
				}
				tokenizer.Merges = append(tokenizer.Merges, merge)

				return []int{256} // Use the merged token
			},
			validate: func(t *testing.T, result string) {
				expected := "he"
				if result != expected {
					t.Errorf("Decode() = %q, want %q", result, expected)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer := NewBPETokenizer()
			tokens := tt.setup(tokenizer)
			result := tokenizer.Decode(tokens)
			tt.validate(t, result)
		})
	}
}

func TestEncode(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		validate func(*testing.T, *BPETokenizer, []int)
	}{
		{
			name: "simple encode",
			text: "hello",
			validate: func(t *testing.T, tokenizer *BPETokenizer, tokens []int) {
				if len(tokens) == 0 {
					t.Error("Expected non-empty tokens")
				}
				// Encode should apply merges, so result might be different from Tokenize
				originalTokens := tokenizer.Tokenize("hello")
				// If there are merges, encoded tokens might be shorter
				if len(tokenizer.Merges) > 0 && len(tokens) > len(originalTokens) {
					t.Error("Encoded tokens should not be longer than original tokens")
				}
			},
		},
		{
			name: "empty text",
			text: "",
			validate: func(t *testing.T, tokenizer *BPETokenizer, tokens []int) {
				if len(tokens) != 0 {
					t.Errorf("Expected empty tokens for empty text, got %v", tokens)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer := NewBPETokenizer()
			// Add some dummy merges to test the encoding process
			tokenizer.vocab["h"] = 0
			tokenizer.vocab["e"] = 1
			tokenizer.idToToken[0] = "h"
			tokenizer.idToToken[1] = "e"
			tokenizer.vocabSize = 2
			tokenizer.Merges = append(tokenizer.Merges, Merge{
				Pair:  Pair{First: 0, Second: 1},
				Index: 256,
			})

			tokens := tokenizer.Encode(tt.text)
			tt.validate(t, tokenizer, tokens)
		})
	}
}

func TestTrain(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		validate func(*testing.T, *BPETokenizer)
	}{
		{
			name: "simple training",
			text: "hello hello world world",
			validate: func(t *testing.T, tokenizer *BPETokenizer) {
				// After training, should have merges
				expectedMerges := VOCAB_SIZE - 256
				if len(tokenizer.Merges) != expectedMerges {
					t.Errorf("Expected %d merges, got %d", expectedMerges, len(tokenizer.Merges))
				}

				// Merges should have valid indices starting from 256
				for i, merge := range tokenizer.Merges {
					expectedIndex := 256 + i
					if merge.Index != expectedIndex {
						t.Errorf("Merge %d has index %d, expected %d", i, merge.Index, expectedIndex)
					}
				}

				// Vocab should be populated
				if len(tokenizer.vocab) == 0 {
					t.Error("Expected vocab to be populated after training")
				}
			},
		},
		{
			name: "empty text training",
			text: "",
			validate: func(t *testing.T, tokenizer *BPETokenizer) {
				// Training on empty text should still create merges (though they might be empty)
				expectedMerges := VOCAB_SIZE - 256
				if len(tokenizer.Merges) != expectedMerges {
					t.Errorf("Expected %d merges even for empty text, got %d", expectedMerges, len(tokenizer.Merges))
				}
			},
		},
		{
			name: "single character training",
			text: "a",
			validate: func(t *testing.T, tokenizer *BPETokenizer) {
				expectedMerges := VOCAB_SIZE - 256
				if len(tokenizer.Merges) != expectedMerges {
					t.Errorf("Expected %d merges, got %d", expectedMerges, len(tokenizer.Merges))
				}

				// Should have at least one token in vocab
				if len(tokenizer.vocab) == 0 {
					t.Error("Expected vocab to be populated")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer := NewBPETokenizer()
			tokenizer.Train(tt.text)
			tt.validate(t, tokenizer)
		})
	}
}

func TestIntegration(t *testing.T) {
	// Test the full pipeline: Train -> Encode -> Decode
	tokenizer := NewBPETokenizer()
	text := "hello world hello world"

	// Train the tokenizer
	tokenizer.Train(text)

	// Encode the text
	encoded := tokenizer.Encode(text)

	// Decode back
	decoded := tokenizer.Decode(encoded)

	// The decoded text should match the original
	if decoded != text {
		t.Errorf("Integration test failed: original=%q, decoded=%q", text, decoded)
	}

	// Test with different text
	newText := "hello world"
	encoded2 := tokenizer.Encode(newText)
	decoded2 := tokenizer.Decode(encoded2)

	if decoded2 != newText {
		t.Errorf("Integration test with new text failed: original=%q, decoded=%q", newText, decoded2)
	}
}
