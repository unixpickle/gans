package main

import (
	"fmt"
	"image/png"
	"io/ioutil"
	"log"
	"math/rand"
	"os"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/gans"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	StepSize  = 0.001
	BatchSize = 128
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage: mnist_gen <model_out> <output.png>")
		os.Exit(1)
	}

	fm := createModel()
	dataSet := mnist.LoadTrainingDataSet()
	samples := dataSet.SGDSampleSet()
	var iteration int
	sgd.SGDMini(fm, samples, StepSize, BatchSize, func(s sgd.SampleSet) bool {
		posCost := testPositiveCost(fm, samples)
		genCost := testGeneratedCost(fm)
		log.Printf("iteration %d: pos_cost=%f  gen_cost=%f", iteration,
			posCost, genCost)
		iteration++
		// TODO: figure out some kind of cost to output here.
		return true
	})

	log.Println("Saving model...")
	data, err := fm.Serialize()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := ioutil.WriteFile(os.Args[1], data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	log.Println("Creating generation grid...")

	renderings := mnist.ReconstructionGrid(func(ignore []float64) []float64 {
		randVec := make([]float64, fm.RandomSize)
		for i := range randVec {
			randVec[i] = rand.NormFloat64()
		}
		return fm.Generator.Apply(&autofunc.Variable{Vector: randVec}).Output()
	}, dataSet, 3, 5)
	outFile, err := os.Create(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer outFile.Close()
	png.Encode(outFile, renderings)
}

func testPositiveCost(fm *gans.FeatureMatching, samples sgd.SampleSet) float64 {
	sample := samples.GetSample(rand.Intn(samples.Len()))
	inVec := sample.(neuralnet.VectorSample).Input
	output := fm.Discriminator.Apply(&autofunc.Variable{Vector: inVec})
	return neuralnet.SigmoidCECost{}.Cost(linalg.Vector{1}, output).Output()[0]
}

func testGeneratedCost(fm *gans.FeatureMatching) float64 {
	genIn := make(linalg.Vector, fm.RandomSize)
	for i := range genIn {
		genIn[i] = rand.NormFloat64()
	}
	genOut := fm.Generator.Apply(&autofunc.Variable{Vector: genIn})
	output := fm.Discriminator.Apply(genOut)
	return neuralnet.SigmoidCECost{}.Cost(linalg.Vector{0}, output).Output()[0]
}

func createModel() *gans.FeatureMatching {
	existing, err := ioutil.ReadFile(os.Args[1])
	if err == nil {
		model, err := gans.DeserializeFeatureMatching(existing)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Failed to deserialize model:", err)
			os.Exit(1)
		}
		log.Println("Loaded existing model.")
		return model
	}

	log.Println("Created new model.")

	genNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  100,
			OutputCount: 300,
		},
		neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  300,
			OutputCount: 400,
		},
		neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  400,
			OutputCount: 784,
		},
	}
	genNet.Randomize()

	discrimNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  784,
			OutputCount: 400,
		},
		neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  400,
			OutputCount: 300,
		},
		neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  300,
			OutputCount: 100,
		},
		neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  100,
			OutputCount: 1,
		},
	}
	discrimNet.Randomize()

	return &gans.FeatureMatching{
		Discriminator: discrimNet,
		FeatureLayers: 6,
		Generator:     genNet,
		RandomSize:    100,
	}
}
