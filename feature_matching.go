package gans

import (
	"errors"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	var f FM
	serializer.RegisterTypedDeserializer(f.SerializerType(), DeserializeFM)
}

// FM trains a generative adversarial network using the
// feature matching approach, making the generator aim
// to approximate a hidden layer in the discriminator.
type FM struct {
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

// DeserializeFM deserializes an instance
// of FM.
func DeserializeFM(d []byte) (*FM, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) != 4 {
		return nil, errors.New("invalid FM slice")
	}
	discrim, ok1 := slice[0].(neuralnet.Network)
	gen, ok2 := slice[1].(neuralnet.Network)
	layers, ok3 := slice[2].(serializer.Int)
	size, ok4 := slice[3].(serializer.Int)
	if !ok1 || !ok2 || !ok3 || !ok4 {
		return nil, errors.New("invalid FM slice")
	}
	return &FM{
		Discriminator: discrim,
		FeatureLayers: int(layers),
		Generator:     gen,
		RandomSize:    int(size),
	}, nil
}

// Gradient computes the gradient to train both the
// generator and the discriminator on the mini-batch of
// actual samples.
// The samples' output vectors are ignored.
func (f *FM) Gradient(samples sgd.SampleSet) autofunc.Gradient {
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

// SampleRealCost measures the cross-entropy cost of the
// discriminator on a randomly chosen sample.
func (f *FM) SampleRealCost(samples sgd.SampleSet) float64 {
	sample := samples.GetSample(rand.Intn(samples.Len()))
	inVec := sample.(neuralnet.VectorSample).Input
	output := f.Discriminator.Apply(&autofunc.Variable{Vector: inVec})
	return neuralnet.SigmoidCECost{}.Cost(linalg.Vector{1}, output).Output()[0]
}

// SampleGenCost measures the cross-entropy cost of the
// discriminator on a generated input.
func (f *FM) SampleGenCost() float64 {
	genIn := make(linalg.Vector, f.RandomSize)
	for i := range genIn {
		genIn[i] = rand.NormFloat64()
	}
	genOut := f.Generator.Apply(&autofunc.Variable{Vector: genIn})
	output := f.Discriminator.Apply(genOut)
	return neuralnet.SigmoidCECost{}.Cost(linalg.Vector{0}, output).Output()[0]
}

// SerializerType returns the unique ID used to serialize
// a FM instance with the serializer package.
func (f *FM) SerializerType() string {
	return "github.com/unixpickle/gans.FeatureMatching"
}

// Serialize serializes the instance as binary data.
func (f *FM) Serialize() ([]byte, error) {
	s := []serializer.Serializer{
		f.Discriminator,
		f.Generator,
		serializer.Int(f.FeatureLayers),
		serializer.Int(f.RandomSize),
	}
	return serializer.SerializeSlice(s)
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
