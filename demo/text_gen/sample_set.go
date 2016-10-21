package main

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"strings"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

type SampleSet struct {
	sentences []string
}

func ReadSampleSet(path string) *SampleSet {
	return &SampleSet{sentences: readSentences(path)}
}

func (s *SampleSet) Len() int {
	return len(s.sentences)
}

func (s *SampleSet) Swap(i, j int) {
	s.sentences[i], s.sentences[j] = s.sentences[j], s.sentences[i]
}

func (s *SampleSet) GetSample(i int) interface{} {
	sample := seqtoseq.Sample{}
	sentence := s.sentences[i]
	sample.Inputs = make([]linalg.Vector, len(sentence))
	for i, b := range []byte(sentence) {
		vec := make(linalg.Vector, CharCount)
		vec[int(b)] = 5
		for j := range vec {
			if j != int(b) {
				vec[j] = rand.NormFloat64() - 5
			}
		}
		softened := QuadSquash{}.Apply(&autofunc.Variable{Vector: vec}).Output()
		sample.Inputs[i] = softened
	}
	return sample
}

func (s *SampleSet) Copy() sgd.SampleSet {
	res := &SampleSet{
		sentences: make([]string, len(s.sentences)),
	}
	copy(res.sentences, s.sentences)
	return res
}

func (s *SampleSet) Subset(start, end int) sgd.SampleSet {
	return &SampleSet{
		sentences: s.sentences[start:end],
	}
}

func readSentences(path string) []string {
	contents, err := ioutil.ReadFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Read sentences failed:", err)
		os.Exit(1)
	}
	strContent := string(contents)
	var sentence string
	var res []string
	for _, chr := range strContent {
		endOfSentence := chr == '.' || chr == '!' || chr == '?'
		if chr == '.' {
			if strings.HasSuffix(sentence, "Mr") ||
				strings.HasSuffix(sentence, "Dr") ||
				strings.HasSuffix(sentence, "Mrs") ||
				strings.HasSuffix(sentence, "Ms") {
				endOfSentence = false
			}
		}
		if endOfSentence {
			sentence = strings.TrimSpace(sentence)
			if isUsableSentence(sentence) {
				sentence += string(chr)
				res = append(res, sentence)
			}
			sentence = ""
		} else {
			sentence += string(chr)
		}
	}
	return res
}

func isUsableSentence(s string) bool {
	if len(s) == 0 {
		return false
	}
	for _, chr := range s {
		if !(chr >= 'a' && chr <= 'z') && !(chr >= 'A' && chr <= 'Z') &&
			chr != ' ' {
			return false
		}
	}
	return true
}
