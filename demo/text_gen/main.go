package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/gans"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	CharCount = 256
	RandCount = 50
	MaxLen    = 150

	GenAtEnd = 10

	BatchSize      = 16
	GenTemperature = 2
)

var StepSize = 1e-3

func main() {
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "corpus.txt model_file")
		os.Exit(1)
	}
	samples := ReadSampleSet(os.Args[1])
	model := readOrCreateModel(os.Args[2])

	model.DiscIterations = 1
	model.GenIterations = 1
	model.GenTrans = &sgd.RMSProp{Resiliency: 0.9}
	model.DiscTrans = &sgd.RMSProp{Resiliency: 0.9}

	log.Println("Training model...")
	var iteration int
	var lastBatch sgd.SampleSet
	var lastGenIn seqfunc.Result
	sgd.SGDMini(model, samples, StepSize, BatchSize, func(s sgd.SampleSet) bool {
		var lastReal, lastGen float64
		if lastBatch != nil {
			lastReal = model.SampleRealCost(lastBatch)
			lastGen = model.SampleGenCost(lastGenIn)
		}
		lastBatch = s.Copy()
		lastGenIn = model.RandomGenInputs(s)
		log.Printf("iteration %d: real=%f gen=%f last_real=%f last_gen=%f", iteration,
			model.SampleRealCost(s), model.SampleGenCost(lastGenIn), lastReal, lastGen)
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
			InputCount:  200,
			OutputCount: 1,
		},
	}
	discOutputBlock.Randomize()
	discOutputBlock[0].(*neuralnet.DenseLayer).Biases.Var.Vector.Scale(0)
	genOutputBlock := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  100,
			OutputCount: CharCount,
		},
		&neuralnet.RescaleLayer{Scale: 1.0 / GenTemperature},
		&neuralnet.SoftmaxLayer{},
	}
	genOutputBlock.Randomize()

	rec := &gans.Recurrent{
		DiscrimFeatures: &rnn.BlockSeqFunc{
			B: NewExpBlock(CharCount, 200),
		},
		DiscrimClassify: &rnn.BlockSeqFunc{
			B: rnn.NewNetworkBlock(discOutputBlock, 0),
		},
		Generator: &rnn.BlockSeqFunc{
			B: rnn.StackedBlock{
				rnn.NewLSTM(RandCount, 200),
				rnn.NewLSTM(200, 100),
				rnn.NewNetworkBlock(genOutputBlock, 0),
			},
		},
		RandomSize: RandCount,
	}
	return rec
}

func generateSentence(model *gans.Recurrent) string {
	var res string
	runner := &rnn.Runner{Block: model.Generator.(*rnn.BlockSeqFunc).B}

	for i := 0; i < MaxLen; i++ {
		input := make(linalg.Vector, model.RandomSize)
		for j := range input {
			input[j] = rand.NormFloat64()
		}

		outVec := runner.StepTime(input)
		res += string(byte(randomSample(outVec[:CharCount])))
	}

	return res
}

func randomSample(probs linalg.Vector) int {
	num := rand.Float64()
	for i, x := range probs {
		if x > num {
			return i
		}
		num -= x
	}
	return len(probs) - 1
}
