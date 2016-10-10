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
	// Discriminator learns to tell if the incoming sequence
	// is a training sample or a generated sample.
	// The inputs will have length x-1 where x is the
	// output size of the generator.
	Discriminator rnn.SeqFunc

	// Generator learns to turn random input sequences into
	// sequences that trick the discriminator.
	// The last output of the block is used as a scaled
	// termination probability.
	Generator rnn.Block

	// RandomSize is the input size of the generator.
	RandomSize int

	// GenActivation is the activation function through
	// which generator outputs are fed before being given
	// to the discriminator.
	// It may be nil for the identity map.
	GenActivation autofunc.Func

	// MaxLen is the maximum generated sequence length.
	MaxLen int
}

// DeserializeRecurrent deserializes a Recurrent instance.
func DeserializeRecurrent(d []byte) (*Recurrent, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) != 4 && len(slice) != 5 {
		return nil, errors.New("invalid Recurrent slice")
	}
	discrim, ok1 := slice[0].(rnn.SeqFunc)
	gen, ok2 := slice[1].(rnn.Block)
	size, ok3 := slice[2].(serializer.Int)
	maxLen, ok4 := slice[3].(serializer.Int)
	if !ok1 || !ok2 || !ok3 || !ok4 {
		return nil, errors.New("invalid Recurrent slice")
	}
	res := &Recurrent{
		Discriminator: discrim,
		Generator:     gen,
		RandomSize:    int(size),
		MaxLen:        int(maxLen),
	}
	if len(slice) == 5 {
		res.GenActivation, ok1 = slice[4].(autofunc.Func)
		if !ok1 {
			return nil, errors.New("invalid Recurrent slice")
		}
	}
	return res, nil
}

// Gradient computes the collective gradient for the set
// of seqtoseq.Samples and an equal number of generated
// sequences.
func (r *Recurrent) Gradient(s sgd.SampleSet) autofunc.Gradient {
	var discParams []*autofunc.Variable
	if p, ok := r.Discriminator.(sgd.Learner); ok {
		discParams = p.Parameters()
	}
	discrimGrad := autofunc.NewGradient(discParams)
	r.sampleGradients(s, discrimGrad)

	var genParams []*autofunc.Variable
	if p, ok := r.Generator.(sgd.Learner); ok {
		genParams = p.Parameters()
	}
	genGrad := autofunc.NewGradient(genParams)
	r.genGradients(genGrad, discrimGrad, s.Len())

	res := autofunc.Gradient{}
	for _, grad := range []autofunc.Gradient{discrimGrad, genGrad} {
		for key, val := range grad {
			res[key] = val
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
	res := r.Discriminator.BatchSeqs([][]autofunc.Result{inSeq})
	lastOut := &autofunc.Variable{
		Vector: res.OutputSeqs()[0][len(res.OutputSeqs()[0])-1],
	}
	cost := neuralnet.SigmoidCECost{}.Cost(linalg.Vector{1}, lastOut)
	return cost.Output()[0]
}

// SampleGenCost measures the cross-entropy cost of the
// discriminator on a generated input.
func (r *Recurrent) SampleGenCost() float64 {
	randomSeq := make([]autofunc.Result, r.MaxLen)
	for i := range randomSeq {
		normVec := make(linalg.Vector, r.RandomSize)
		for j := range normVec {
			normVec[j] = rand.NormFloat64()
		}
		randomSeq[i] = &autofunc.Variable{Vector: normVec}
	}

	genFunc := &rnn.BlockSeqFunc{Block: r.Generator}
	genOut := genFunc.BatchSeqs([][]autofunc.Result{randomSeq})

	discInput := make([]autofunc.Result, r.MaxLen)

	var rawMasks linalg.Vector
	for i, x := range genOut.OutputSeqs()[0] {
		outVec := &autofunc.Variable{Vector: x}
		prefix := autofunc.Slice(outVec, 0, len(x)-1)
		if r.GenActivation != nil {
			prefix = r.GenActivation.Apply(prefix)
		}
		discInput[i] = prefix
		rawMasks = append(rawMasks, x[len(x)-1])
	}
	softmax := autofunc.Softmax{}
	discMask := softmax.Apply(&autofunc.Variable{Vector: rawMasks}).Output()

	discOut := r.Discriminator.BatchSeqs([][]autofunc.Result{discInput})
	var totalCost float64
	for i, out := range discOut.OutputSeqs()[0] {
		outVar := &autofunc.Variable{Vector: out}
		cost := neuralnet.SigmoidCECost{}.Cost(linalg.Vector{0}, outVar)
		totalCost += cost.Output()[0] * discMask[i]
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
	discSerializer, ok := r.Discriminator.(serializer.Serializer)
	if !ok {
		return nil, fmt.Errorf("type %T is not a Serializer", r.Discriminator)
	}
	genSerializer, ok := r.Generator.(serializer.Serializer)
	if !ok {
		return nil, fmt.Errorf("type %T is not a Serializer", r.Generator)
	}
	serializers := []serializer.Serializer{
		discSerializer,
		genSerializer,
		serializer.Int(r.RandomSize),
		serializer.Int(r.MaxLen),
	}
	if r.GenActivation != nil {
		actSerializer, ok := r.GenActivation.(serializer.Serializer)
		if !ok {
			return nil, fmt.Errorf("type %T is not a Serializer", r.GenActivation)
		}
		serializers = append(serializers, actSerializer)
	}
	return serializer.SerializeSlice(serializers)
}

func (r *Recurrent) sampleGradients(s sgd.SampleSet, g autofunc.Gradient) {
	var inputSeqs [][]autofunc.Result
	for i := 0; i < s.Len(); i++ {
		sample := s.GetSample(i).(seqtoseq.Sample)
		seq := make([]autofunc.Result, len(sample.Inputs))
		for j, x := range sample.Inputs {
			seq[j] = &autofunc.Variable{Vector: x}
		}
		inputSeqs = append(inputSeqs, seq)
	}
	output := r.Discriminator.BatchSeqs(inputSeqs)
	upstream := make([][]linalg.Vector, len(output.OutputSeqs()))
	for i, outSeq := range output.OutputSeqs() {
		upstream[i] = make([]linalg.Vector, len(outSeq))
		for j := 0; j < len(outSeq)-1; j++ {
			upstream[i][j] = linalg.Vector{0}
		}
		outVec := outSeq[len(outSeq)-1]
		outVar := &autofunc.Variable{Vector: outVec}
		cost := neuralnet.SigmoidCECost{}.Cost(linalg.Vector{1}, outVar)
		tempGrad := autofunc.Gradient{outVar: linalg.Vector{0}}
		cost.PropagateGradient(linalg.Vector{1}, tempGrad)
		upstream[i][len(outSeq)-1] = tempGrad[outVar]
	}
	output.Gradient(upstream, g)
}

func (r *Recurrent) genGradients(genGrad, discGrad autofunc.Gradient, count int) {
	randomSeqs := make([][]autofunc.Result, count)
	for i := range randomSeqs {
		seq := make([]autofunc.Result, r.MaxLen)
		randomSeqs[i] = seq
		for j := range seq {
			normVec := make(linalg.Vector, r.RandomSize)
			for k := range normVec {
				normVec[k] = rand.NormFloat64()
			}
			seq[j] = &autofunc.Variable{Vector: normVec}
		}
	}

	genFunc := &rnn.BlockSeqFunc{Block: r.Generator}
	genOut := genFunc.BatchSeqs(randomSeqs)

	poolVariables := make([][]*autofunc.Variable, len(genOut.OutputSeqs()))
	discInput := make([][]autofunc.Result, len(genOut.OutputSeqs()))
	discMask := make([][]autofunc.Result, len(genOut.OutputSeqs()))
	for i, outSeq := range genOut.OutputSeqs() {
		poolSeq := make([]*autofunc.Variable, len(outSeq))
		discInputSeq := make([]autofunc.Result, len(outSeq))
		poolVariables[i] = poolSeq
		discInput[i] = discInputSeq

		var rawEndWeights []autofunc.Result
		for j, x := range outSeq {
			poolSeq[j] = &autofunc.Variable{Vector: x}
			prefix := autofunc.Slice(poolSeq[j], 0, len(x)-1)
			if r.GenActivation != nil {
				prefix = r.GenActivation.Apply(prefix)
			}
			discInputSeq[j] = prefix
			lastVal := autofunc.Slice(poolSeq[j], len(x)-1, len(x))
			rawEndWeights = append(rawEndWeights, lastVal)
		}
		softmax := autofunc.Softmax{}
		endProbs := softmax.Apply(autofunc.Concat(rawEndWeights...))
		discMask[i] = make([]autofunc.Result, len(endProbs.Output()))
		for j := range discMask[i] {
			discMask[i][j] = autofunc.Slice(endProbs, j, j+1)
		}
	}

	discOut := r.Discriminator.BatchSeqs(discInput)
	r.propGenThroughDisc(discOut, discMask, discGrad, 0)
	r.propGenThroughGen(genOut, discOut, poolVariables, discMask, genGrad)
}

func (r *Recurrent) propGenThroughDisc(out rnn.ResultSeqs, mask [][]autofunc.Result,
	g autofunc.Gradient, desiredVal float64) {
	upstream := make([][]linalg.Vector, len(out.OutputSeqs()))
	for i, outSeq := range out.OutputSeqs() {
		upstream[i] = make([]linalg.Vector, len(outSeq))
		for j, x := range outSeq {
			outVar := &autofunc.Variable{Vector: x}
			g[outVar] = linalg.Vector{0}
			maskedOut := autofunc.Mul(outVar, mask[i][j])
			rawCost := neuralnet.SigmoidCECost{}.Cost(linalg.Vector{desiredVal}, maskedOut)
			rawCost.PropagateGradient(linalg.Vector{1}, g)
			upstream[i][j] = g[outVar]
			delete(g, outVar)
		}
	}
	out.Gradient(upstream, g)
}

func (r *Recurrent) propGenThroughGen(genOut, discOut rnn.ResultSeqs,
	pool [][]*autofunc.Variable, mask [][]autofunc.Result,
	g autofunc.Gradient) {
	tempGrad := autofunc.Gradient{}
	for _, x := range pool {
		for _, y := range x {
			tempGrad[y] = make(linalg.Vector, len(y.Output()))
		}
	}
	r.propGenThroughDisc(discOut, mask, tempGrad, 1)

	upstream := make([][]linalg.Vector, len(pool))
	for i, x := range pool {
		upstream[i] = make([]linalg.Vector, len(x))
		for j, y := range x {
			upstream[i][j] = tempGrad[y]
		}
	}
	genOut.Gradient(upstream, g)
}
