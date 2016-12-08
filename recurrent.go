package gans

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

func init() {
	var r Recurrent
	serializer.RegisterTypedDeserializer(r.SerializerType(), DeserializeRecurrent)
}

// Recurrent trains a GAN comprised of two RNNs.
type Recurrent struct {
	GenIterations  int
	DiscIterations int

	GenTrans  sgd.Transformer
	DiscTrans sgd.Transformer

	// DiscrimFeatures transforms a sample (either generated
	// or real) into some features for feature matching.
	DiscrimFeatures seqfunc.RFunc

	// DiscrimClassify produces classifications ("real" or
	// "generated") for every timestep in an input sequence
	// of features from DiscrimFeatures.
	//
	// The classifications should be raw linear values, since
	// they are automatically fed into a sigmoid during
	// training.
	DiscrimClassify seqfunc.RFunc

	// Generator takes sequences of random vectors and makes
	// synthetic sequences.
	Generator seqfunc.RFunc

	// RandomSize specifies the vector input size of the
	// generator.
	RandomSize int

	iterIdx int
}

// DeserializeRecurrent deserializes a Recurrent instance.
func DeserializeRecurrent(d []byte) (*Recurrent, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) != 4 {
		return nil, errors.New("invalid Recurrent slice")
	}
	features, ok1 := slice[0].(seqfunc.RFunc)
	classify, ok2 := slice[1].(seqfunc.RFunc)
	gen, ok3 := slice[2].(seqfunc.RFunc)
	size, ok4 := slice[3].(serializer.Int)
	if !ok1 || !ok2 || !ok3 || !ok4 {
		return nil, errors.New("invalid Recurrent slice")
	}
	return &Recurrent{
		DiscrimFeatures: features,
		DiscrimClassify: classify,
		Generator:       gen,
		RandomSize:      int(size),
	}, nil
}

// Gradient computes the collective gradient for the set
// of seqtoseq.Samples and an equal number of generated
// sequences.
func (r *Recurrent) Gradient(s sgd.SampleSet) autofunc.Gradient {
	subIdx := r.iterIdx % (r.GenIterations + r.DiscIterations)
	r.iterIdx++

	genGrad := autofunc.NewGradient(r.Generator.(sgd.Learner).Parameters())
	discGrad := autofunc.NewGradient(r.discriminatorParams())

	genInputs := r.generatorInputs(s)
	realInputs := r.sampleInputs(s)

	genOutput := r.Generator.ApplySeqs(genInputs)
	genFeatures := r.DiscrimFeatures.ApplySeqs(genOutput)
	genClassifications := r.DiscrimClassify.ApplySeqs(genFeatures)

	if subIdx < r.DiscIterations {
		realFeatures := r.DiscrimFeatures.ApplySeqs(realInputs)
		realClassifications := r.DiscrimClassify.ApplySeqs(realFeatures)

		posCostFunc := func(a autofunc.Result) autofunc.Result {
			return neuralnet.SigmoidCECost{}.Cost([]float64{1}, a)
		}
		negCostFunc := func(a autofunc.Result) autofunc.Result {
			return neuralnet.SigmoidCECost{}.Cost([]float64{0}, a)
		}
		discrimCost := autofunc.Add(
			seqfunc.AddAll(seqfunc.Map(realClassifications, posCostFunc)),
			seqfunc.AddAll(seqfunc.Map(genClassifications, negCostFunc)),
		)
		discrimCost.PropagateGradient([]float64{1}, discGrad)

		if r.DiscTrans != nil {
			discGrad = r.DiscTrans.Transform(discGrad)
		}
	} else {
		genCostFunc := func(a autofunc.Result) autofunc.Result {
			return neuralnet.SigmoidCECost{}.Cost([]float64{0}, a)
		}
		cost := seqfunc.AddAll(seqfunc.Map(genClassifications, genCostFunc))
		cost.PropagateGradient([]float64{-1}, genGrad)

		if r.GenTrans != nil {
			genGrad = r.GenTrans.Transform(genGrad)
		}
	}

	res := autofunc.Gradient{}
	for _, g := range []autofunc.Gradient{genGrad, discGrad} {
		for k, v := range g {
			res[k] = v
		}
	}

	return res
}

// SampleRealCost measures the cross-entropy cost of the
// discriminator on the first sample.
func (r *Recurrent) SampleRealCost(samples sgd.SampleSet) float64 {
	sample := samples.GetSample(0).(seqtoseq.Sample)
	inRes := seqfunc.ConstResult([][]linalg.Vector{sample.Inputs})
	features := r.DiscrimFeatures.ApplySeqs(inRes)
	res := r.DiscrimClassify.ApplySeqs(features).OutputSeqs()[0]
	var totalCost float64
	for _, x := range res {
		outVar := &autofunc.Variable{Vector: x}
		cost := neuralnet.SigmoidCECost{}.Cost(linalg.Vector{1}, outVar)
		totalCost += cost.Output()[0]
	}
	return totalCost
}

// RandomGenInputs generates random inputs for the
// generator which can be fed to SampleGenCost.
func (r *Recurrent) RandomGenInputs(samples sgd.SampleSet) seqfunc.Result {
	idx := rand.Intn(samples.Len())
	return r.generatorInputs(samples.Subset(idx, idx+1))
}

// SampleGenCost measures the cross-entropy cost of the
// discriminator on a generated input.
func (r *Recurrent) SampleGenCost(genIn seqfunc.Result) float64 {
	generated := r.Generator.ApplySeqs(genIn)
	features := r.DiscrimFeatures.ApplySeqs(generated)
	res := r.DiscrimClassify.ApplySeqs(features).OutputSeqs()[0]
	var totalCost float64
	for _, x := range res {
		outVar := &autofunc.Variable{Vector: x}
		cost := neuralnet.SigmoidCECost{}.Cost(linalg.Vector{0}, outVar)
		totalCost += cost.Output()[0]
	}
	return totalCost
}

// SerializerType returns the unique ID used to serialize
// Recurrent instances with the serializer package.
func (r *Recurrent) SerializerType() string {
	return "github.com/unixpickle/gans.Recurrent"
}

// Serialize serializes the instance.
func (r *Recurrent) Serialize() ([]byte, error) {
	featuresSerializer, ok := r.DiscrimFeatures.(serializer.Serializer)
	if !ok {
		return nil, fmt.Errorf("type %T is not a Serializer", r.DiscrimFeatures)
	}
	classifySerializer, ok := r.DiscrimClassify.(serializer.Serializer)
	if !ok {
		return nil, fmt.Errorf("type %T is not a Serializer", r.DiscrimClassify)
	}
	genSerializer, ok := r.Generator.(serializer.Serializer)
	if !ok {
		return nil, fmt.Errorf("type %T is not a Serializer", r.Generator)
	}
	serializers := []serializer.Serializer{
		featuresSerializer,
		classifySerializer,
		genSerializer,
		serializer.Int(r.RandomSize),
	}
	return serializer.SerializeSlice(serializers)
}

func (r *Recurrent) discriminatorParams() []*autofunc.Variable {
	var res []*autofunc.Variable
	for _, layer := range []seqfunc.RFunc{r.DiscrimClassify, r.DiscrimFeatures} {
		res = append(res, layer.(sgd.Learner).Parameters()...)
	}
	return res
}

func (r *Recurrent) generatorInputs(s sgd.SampleSet) seqfunc.Result {
	var res [][]linalg.Vector
	for i := 0; i < s.Len(); i++ {
		var inSeq []linalg.Vector
		for _ = range s.GetSample(i).(seqtoseq.Sample).Inputs {
			inVec := make(linalg.Vector, r.RandomSize)
			for j := range inVec {
				inVec[j] = rand.NormFloat64()
			}
			inSeq = append(inSeq, inVec)
		}
		res = append(res, inSeq)
	}
	return seqfunc.ConstResult(res)
}

func (r *Recurrent) sampleInputs(s sgd.SampleSet) seqfunc.Result {
	var res [][]linalg.Vector
	for i := 0; i < s.Len(); i++ {
		res = append(res, s.GetSample(i).(seqtoseq.Sample).Inputs)
	}
	return seqfunc.ConstResult(res)
}
