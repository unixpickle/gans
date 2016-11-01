package gans

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/rnntest"
)

func TestFeedbackBlock(t *testing.T) {
	inner := rnn.NewLSTM(6, 2)
	block := NewFeedbackBlock(inner, 2)
	for i := range block.InitFeedback.Vector {
		block.InitFeedback.Vector[i] = rand.NormFloat64()
	}
	checker := rnntest.NewChecker4In(block, block)
	checker.FullCheck(t)
}
