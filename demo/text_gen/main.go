package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

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

	GenAtEnd       = 10
	GenTemperature = 1

	BatchSize = 64
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
	sgd.SGDMini(model, samples, StepSize, BatchSize, func(s sgd.SampleSet) bool {
		var lastReal, lastGen float64
		if lastBatch != nil {
			lastReal = model.DiscCost(lastBatch).Output()[0]
			lastGen = model.GenReward(lastBatch)
		}
		lastBatch = s.Copy()
		log.Printf("iteration %d: disc=%f gen=%f last_disc=%f last_gen=%f", iteration,
			model.DiscCost(s).Output()[0], model.GenReward(s), lastReal, lastGen)
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

	rec := &gans.Recurrent{
		Discriminator: &rnn.BlockSeqFunc{
			B: rnn.StackedBlock{
				rnn.NewLSTM(CharCount, 200),
				rnn.NewNetworkBlock(neuralnet.Network{
					neuralnet.NewDenseLayer(200, 1),
				}, 0),
			},
		},
		Generator: &rnn.BlockSeqFunc{
			B: rnn.StackedBlock{
				rnn.NewLSTM(RandCount, 200),
				rnn.NewLSTM(200, 100),
				rnn.NewNetworkBlock(neuralnet.Network{
					neuralnet.NewDenseLayer(100, CharCount),
					&neuralnet.RescaleLayer{Scale: 1.0 / GenTemperature},
					&neuralnet.LogSoftmaxLayer{},
				}, 0),
			},
		},
		RandomSize:     RandCount,
		DiscountFactor: 0.8,
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
		num -= math.Exp(x)
		if num < 0 {
			return i
		}
	}
	return len(probs) - 1
}
