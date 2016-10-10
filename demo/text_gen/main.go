package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/gans"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	CharCount = 256
	RandCount = 10
	MaxLen    = 150

	GenAtEnd = 10

	BatchSize = 16
	StepSize  = 0.001
)

func main() {
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "corpus.txt model_file")
		os.Exit(1)
	}
	samples := ReadSampleSet(os.Args[1])
	model := readOrCreateModel(os.Args[2])

	g := &sgd.Adam{
		Gradienter: model,
	}

	log.Println("Training model...")
	var iteration int
	sgd.SGDMini(g, samples, StepSize, BatchSize, func(s sgd.SampleSet) bool {
		log.Printf("iteration %d: real=%f gen=%f", iteration,
			model.SampleRealCost(s), model.SampleGenCost())
		iteration++
		return true
	})

	log.Println("Saving model...")
	data, err := model.Serialize()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Serialize failed:", err)
		os.Exit(1)
	}
	if err := ioutil.WriteFile(os.Args[2], data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, "Save failed:", err)
		os.Exit(1)
	}

	log.Println("Generating sentences...")
	for i := 0; i < GenAtEnd; i++ {
		fmt.Println(generateSentence(model))
	}
}

func readOrCreateModel(path string) *gans.Recurrent {
	contents, err := ioutil.ReadFile(path)
	if err == nil {
		rec, err := gans.DeserializeRecurrent(contents)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Deserialize failed:", err)
			os.Exit(1)
		}
		log.Println("Loaded model.")
		return rec
	}

	log.Println("Creating new model...")

	discOutputBlock := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  100,
			OutputCount: 1,
		},
	}
	discOutputBlock.Randomize()
	genOutputBlock := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  100,
			OutputCount: CharCount + 1,
		},
	}
	genOutputBlock.Randomize()
	rec := &gans.Recurrent{
		Discriminator: &rnn.BlockSeqFunc{
			Block: rnn.StackedBlock{
				rnn.NewLSTM(CharCount, 100),
				rnn.NewNetworkBlock(discOutputBlock, 0),
			},
		},
		Generator: rnn.StackedBlock{
			rnn.NewLSTM(RandCount, 100),
			rnn.NewNetworkBlock(genOutputBlock, 0),
		},
		GenActivation: &neuralnet.SoftmaxLayer{},
		RandomSize:    RandCount,
		MaxLen:        MaxLen,
	}
	return rec
}

func generateSentence(model *gans.Recurrent) string {
	var res string
	runner := &rnn.Runner{Block: model.Generator}
	for i := 0; i < MaxLen; i++ {
		input := make(linalg.Vector, model.RandomSize)
		for j := range input {
			input[j] = rand.NormFloat64()
		}

		outVec := runner.StepTime(input)
		endProbVar := &autofunc.Variable{Vector: outVec[CharCount:]}
		endProb := autofunc.Sigmoid{}.Apply(endProbVar).Output()[0]

		out := &autofunc.Variable{Vector: outVec[:CharCount]}
		softmax := &autofunc.Softmax{}
		softmaxed := softmax.Apply(out).Output()
		var idx int
		num := rand.Float64()
		for i, x := range softmaxed {
			if x > num {
				idx = i
				break
			}
			num -= x
		}
		res += string(byte(idx))

		if rand.Float64() < endProb {
			break
		}
	}
	return res
}
