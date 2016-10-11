package gans

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

func init() {
	var r Recurrent
	serializer.RegisterTypedDeserializer(r.SerializerType(), DeserializeRecurrent)
}

// Recurrent trains a GAN comprised of two RNNs.
type Recurrent struct {
	DiscrimFeatures rnn.SeqFunc
	DiscrimClassify rnn.SeqFunc

	Generator  rnn.SeqFunc
	RandomSize int
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
	features, ok1 := slice[0].(rnn.SeqFunc)
	classify, ok2 := slice[1].(rnn.SeqFunc)
	gen, ok3 := slice[2].(rnn.SeqFunc)
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
	featureFunc := rnn.ComposedSeqFunc{r.Generator, r.DiscrimFeatures}
	discFunc := rnn.ComposedSeqFunc{r.DiscrimFeatures, r.DiscrimClassify}
	fullFunc := append(rnn.ComposedSeqFunc{r.Generator}, discFunc...)

	genInputs := r.generatorInputs(s)
	realInputs := r.sampleInputs(s)
	genFeatures := featureFunc.BatchSeqs(genInputs)
	realFeatures := r.DiscrimFeatures.BatchSeqs(realInputs)

	discRealResults := discFunc.BatchSeqs(realInputs)
	discGenResults := fullFunc.BatchSeqs(genInputs)

	genGrad := autofunc.NewGradient(r.Generator.(sgd.Learner).Parameters())
	discGrad := autofunc.NewGradient(discFunc.Parameters())

	r.discrimGradient(discRealResults, 1, discGrad)
	r.discrimGradient(discGenResults, 0, discGrad)
	r.genGradient(genFeatures, realFeatures, genGrad)

	res := autofunc.Gradient{}
	for _, g := range []autofunc.Gradient{genGrad, discGrad} {
		for k, v := range g {
			res[k] = v
		}
	}

	return res
}

// SampleRealCost measures the cross-entropy cost of the
// discriminator on a randomly chosen sample.
func (r *Recurrent) SampleRealCost(samples sgd.SampleSet) float64 {
	sample := samples.GetSample(rand.Intn(samples.Len())).(seqtoseq.Sample)
	var inSeq []autofunc.Result
	for _, x := range sample.Inputs {
		inSeq = append(inSeq, &autofunc.Variable{Vector: x})
	}
	composed := rnn.ComposedSeqFunc{r.DiscrimFeatures, r.DiscrimClassify}
	res := composed.BatchSeqs([][]autofunc.Result{inSeq}).OutputSeqs()[0]
	var totalCost float64
	for _, x := range res {
		outVar := &autofunc.Variable{Vector: x}
		cost := neuralnet.SigmoidCECost{}.Cost(linalg.Vector{1}, outVar)
		totalCost += cost.Output()[0]
	}
	return totalCost
}

// SampleGenCost measures the cross-entropy cost of the
// discriminator on a generated input with the same length
// as a randomly selected sample.
func (r *Recurrent) SampleGenCost(samples sgd.SampleSet) float64 {
	idx := rand.Intn(samples.Len())
	inSeqs := r.generatorInputs(samples.Subset(idx, idx+1))
	composed := rnn.ComposedSeqFunc{r.DiscrimFeatures, r.DiscrimClassify}
	res := composed.BatchSeqs(inSeqs).OutputSeqs()[0]
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

func (r *Recurrent) generatorInputs(s sgd.SampleSet) [][]autofunc.Result {
	var res [][]autofunc.Result
	for i := 0; i < s.Len(); i++ {
		var inSeq []autofunc.Result
		for _ = range s.GetSample(i).(seqtoseq.Sample).Inputs {
			inVec := make(linalg.Vector, r.RandomSize)
			for j := range inVec {
				inVec[j] = rand.NormFloat64()
			}
			inSeq = append(inSeq, &autofunc.Variable{Vector: inVec})
		}
		res = append(res, inSeq)
	}
	return res
}

func (r *Recurrent) sampleInputs(s sgd.SampleSet) [][]autofunc.Result {
	var res [][]autofunc.Result
	for i := 0; i < s.Len(); i++ {
		var inSeq []autofunc.Result
		for _, x := range s.GetSample(i).(seqtoseq.Sample).Inputs {
			inSeq = append(inSeq, &autofunc.Variable{Vector: x})
		}
		res = append(res, inSeq)
	}
	return res
}

func (r *Recurrent) discrimGradient(res rnn.ResultSeqs, desired float64, g autofunc.Gradient) {
	var upstream [][]linalg.Vector
	for _, outSeq := range res.OutputSeqs() {
		var resSeq []linalg.Vector
		for _, outVec := range outSeq {
			v := &autofunc.Variable{Vector: outVec}
			cost := neuralnet.SigmoidCECost{}.Cost(linalg.Vector{desired}, v)
			grad := autofunc.Gradient{v: linalg.Vector{0}}
			cost.PropagateGradient(linalg.Vector{1}, grad)
			resSeq = append(resSeq, grad[v])
		}
		upstream = append(upstream, resSeq)
	}
	res.Gradient(upstream, g)
}

func (r *Recurrent) genGradient(gen, real rnn.ResultSeqs, g autofunc.Gradient) {
	var realAvg linalg.Vector
	var count int
	for _, realSeq := range real.OutputSeqs() {
		for _, realVec := range realSeq {
			if realAvg == nil {
				realAvg = realVec.Copy()
			} else {
				realAvg.Add(realVec)
			}
			count++
		}
	}
	realAvg.Scale(1 / float64(count))

	var poolVars [][]*autofunc.Variable
	var poolSum autofunc.Result
	poolGrad := autofunc.Gradient{}
	count = 0
	for _, genSeq := range gen.OutputSeqs() {
		var poolSeq []*autofunc.Variable
		for _, genVec := range genSeq {
			v := &autofunc.Variable{Vector: genVec}
			poolSeq = append(poolSeq, v)
			if poolSum == nil {
				poolSum = v
			} else {
				poolSum = autofunc.Add(poolSum, v)
			}
			poolGrad[v] = make(linalg.Vector, len(v.Output()))
			count++
		}
		poolVars = append(poolVars, poolSeq)
	}
	poolSum = autofunc.Scale(poolSum, 1/float64(count))

	cost := neuralnet.MeanSquaredCost{}.Cost(realAvg, poolSum)
	cost.PropagateGradient(linalg.Vector{1}, poolGrad)

	var upstream [][]linalg.Vector
	for _, poolSeq := range poolVars {
		var upstreamVec []linalg.Vector
		for _, poolVar := range poolSeq {
			upstreamVec = append(upstreamVec, poolGrad[poolVar])
		}
		upstream = append(upstream, upstreamVec)
	}

	gen.Gradient(upstream, g)
}
