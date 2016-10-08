package gans

import (
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// FeatureMatching trains a generative adversarial network
// by making the generator learn to approximate expected
// feature vectors in a layer of the discriminator.
type FeatureMatching struct {
	// Discriminator is the full discriminator network
	// minus the output sigmoid layer.
	Discriminator neuralnet.Network

	// FeatureLayers is the number of layers from the
	// discriminator to use to generate feature vectors
	// for the generator to generate.
	FeatureLayers int

	// Generator is the generator network.
	Generator neuralnet.Network

	// RandomSize is the size of the generator's random
	// input vectors.
	RandomSize int
}

// Gradient computes the gradient to train both the
// generator and the discriminator on the mini-batch of
// actual samples.
// The samples' output vectors are ignored.
func (f *FeatureMatching) Gradient(samples sgd.SampleSet) autofunc.Gradient {
	n := samples.Len()

	var realBatch linalg.Vector
	for i := 0; i < samples.Len(); i++ {
		vecSamp := samples.GetSample(i).(neuralnet.VectorSample)
		realBatch = append(realBatch, vecSamp.Input...)
	}
	featureNet := f.Discriminator[:f.FeatureLayers].BatchLearner()
	discrimTail := f.Discriminator[f.FeatureLayers:].BatchLearner()

	realFeatures := featureNet.Batch(&autofunc.Variable{Vector: realBatch}, n)
	realOutput := discrimTail.Batch(realFeatures, n)
	realMean := meanFeatures(realFeatures, n)

	randomIn := make(linalg.Vector, n*f.RandomSize)
	for i := range randomIn {
		randomIn[i] = rand.NormFloat64()
	}
	genOut := f.Generator.BatchLearner().Batch(&autofunc.Variable{Vector: randomIn}, n)
	genMeanFeatures := meanFeatures(featureNet.Batch(genOut, n), n)
	genCost := neuralnet.MeanSquaredCost{}.Cost(realMean.Output(), genMeanFeatures)

	genGrad := autofunc.NewGradient(f.Generator.Parameters())
	genCost.PropagateGradient(linalg.Vector{1}, genGrad)

	genDiscrimOut := f.Discriminator.BatchLearner().Batch(genOut, samples.Len())
	discrimGrad := autofunc.NewGradient(f.Discriminator.Parameters())
	realDiscrimCost := neuralnet.SigmoidCECost{}.Cost(repeat(linalg.Vector{1}, n),
		realOutput)
	genDiscrimCost := neuralnet.SigmoidCECost{}.Cost(repeat(linalg.Vector{0}, n),
		genDiscrimOut)
	realDiscrimCost.PropagateGradient(linalg.Vector{0.1}, discrimGrad)
	genDiscrimCost.PropagateGradient(linalg.Vector{0.1}, discrimGrad)

	resGrad := autofunc.Gradient{}
	for _, subGrad := range []autofunc.Gradient{genGrad, discrimGrad} {
		for key, val := range subGrad {
			resGrad[key] = val
		}
	}
	return resGrad
}

func repeat(vec linalg.Vector, n int) linalg.Vector {
	var res linalg.Vector
	for i := 0; i < n; i++ {
		res = append(res, vec...)
	}
	return res
}

func meanFeatures(features autofunc.Result, n int) autofunc.Result {
	return autofunc.Pool(features, func(features autofunc.Result) autofunc.Result {
		featureLen := len(features.Output()) / n
		var res autofunc.Result
		for i := 0; i < n; i++ {
			if i == 0 {
				res = autofunc.Slice(features, 0, featureLen)
			} else {
				featureSubVec := autofunc.Slice(features, i*featureLen, (i+1)*featureLen)
				res = autofunc.Add(res, featureSubVec)
			}
		}
		res = autofunc.Scale(res, 1/float64(n))
		return res
	})
}
