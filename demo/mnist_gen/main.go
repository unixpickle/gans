package main

import (
	"fmt"
	"image/png"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/gans"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	StepSize  = 0.001
	BatchSize = 64
)

func main() {
	rand.Seed(time.Now().UnixNano())

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
	}, dataSet, 5, 8)
	outFile, err := os.Create(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer outFile.Close()
	png.Encode(outFile, renderings)
}

func testPositiveCost(fm *gans.FM, samples sgd.SampleSet) float64 {
	sample := samples.GetSample(rand.Intn(samples.Len()))
	inVec := sample.(neuralnet.VectorSample).Input
	output := fm.Discriminator.Apply(&autofunc.Variable{Vector: inVec})
	return neuralnet.SigmoidCECost{}.Cost(linalg.Vector{1}, output).Output()[0]
}

func testGeneratedCost(fm *gans.FM) float64 {
	genIn := make(linalg.Vector, fm.RandomSize)
	for i := range genIn {
		genIn[i] = rand.NormFloat64()
	}
	genOut := fm.Generator.Apply(&autofunc.Variable{Vector: genIn})
	output := fm.Discriminator.Apply(genOut)
	return neuralnet.SigmoidCECost{}.Cost(linalg.Vector{0}, output).Output()[0]
}

func createModel() *gans.FM {
	existing, err := ioutil.ReadFile(os.Args[1])
	if err == nil {
		model, err := gans.DeserializeFM(existing)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Failed to deserialize model:", err)
			os.Exit(1)
		}
		log.Println("Loaded existing model.")
		return model
	}

	log.Println("Created new model.")

	discrim := createDiscriminator()
	return &gans.FM{
		Discriminator: discrim,
		FeatureLayers: len(discrim) - 2,
		Generator:     createGenerator(),
		RandomSize:    14 * 14,
	}
}

func createGenerator() neuralnet.Network {
	var res neuralnet.Network

	res = append(res, &neuralnet.DenseLayer{
		InputCount:  14 * 14,
		OutputCount: 14 * 14,
	}, neuralnet.HyperbolicTangent{})

	lastDepth := 1
	for i, outDepth := range []int{6, 15, 20, 20, 4} {
		if i > 0 {
			res = append(res, neuralnet.ReLU{})
		}
		res = append(res, &neuralnet.BorderLayer{
			InputWidth:   14,
			InputHeight:  14,
			InputDepth:   lastDepth,
			LeftBorder:   1,
			RightBorder:  1,
			TopBorder:    1,
			BottomBorder: 1,
		}, &neuralnet.ConvLayer{
			InputWidth:   16,
			InputHeight:  16,
			InputDepth:   lastDepth,
			Stride:       1,
			FilterCount:  outDepth,
			FilterWidth:  3,
			FilterHeight: 3,
		})
		lastDepth = outDepth
	}
	res = append(res, &neuralnet.UnstackLayer{
		InputWidth:    14,
		InputHeight:   14,
		InputDepth:    4,
		InverseStride: 2,
	}, neuralnet.Sigmoid{})
	res.Randomize()
	for _, layer := range res {
		if conv, ok := layer.(*neuralnet.ConvLayer); ok {
			for i := range conv.Biases.Vector {
				conv.Biases.Vector[i] = 1
			}
		}
	}
	return res
}

func createDiscriminator() neuralnet.Network {
	var res neuralnet.Network
	width := 28
	height := 28
	depth := 1
	for i := 0; i < 3; i++ {
		conv := &neuralnet.ConvLayer{
			FilterCount:  10 + i*10,
			FilterWidth:  3,
			FilterHeight: 3,
			Stride:       1,
			InputWidth:   width,
			InputHeight:  height,
			InputDepth:   depth,
		}
		res = append(res, conv)
		res = append(res, neuralnet.ReLU{})
		max := &neuralnet.MaxPoolingLayer{
			InputWidth:  conv.OutputWidth(),
			InputHeight: conv.OutputHeight(),
			InputDepth:  conv.OutputDepth(),
			XSpan:       2,
			YSpan:       2,
		}
		res = append(res, max)
		width = max.OutputWidth()
		height = max.OutputHeight()
		depth = conv.OutputDepth()
	}
	res = append(res, &neuralnet.DenseLayer{
		InputCount:  width * height * depth,
		OutputCount: 100,
	})
	res = append(res, neuralnet.HyperbolicTangent{})
	res = append(res, &neuralnet.DenseLayer{
		InputCount:  100,
		OutputCount: 1,
	})
	res.Randomize()
	return res
}
