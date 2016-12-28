package gans

import (
	"math"
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

	// Discriminator outputs values which are meant to be fed
	// into a softmax, with higher values indicating "real"
	// samples.
	Discriminator seqfunc.RFunc

	// Generator takes sequences of random vectors and makes
	// synthetic sequences.
	// Its output activation should be neuralnet.LogSoftmax.
	Generator seqfunc.RFunc

	// RandomSize specifies the vector input size of the
	// generator.
	RandomSize int

	// DiscountFactor is the factor by which rewards are
	// discounted after every timestep.
	// A value of 0 is treated as 1.
	DiscountFactor float64

	iterIdx int
}

// DeserializeRecurrent deserializes a Recurrent instance.
func DeserializeRecurrent(d []byte) (*Recurrent, error) {
	res := &Recurrent{}
	var randomSize serializer.Int
	var discount serializer.Float64
	err := serializer.DeserializeAny(d, &res.Discriminator, &res.Generator,
		&randomSize, &discount)
	if err != nil {
		return nil, err
	}
	res.RandomSize = int(randomSize)
	res.DiscountFactor = float64(discount)
	return res, nil
}

// SerializerType returns the unique ID used to serialize
// Recurrent instances with the serializer package.
func (r *Recurrent) SerializerType() string {
	return "github.com/unixpickle/gans.Recurrent"
}

// Serialize serializes the instance.
func (r *Recurrent) Serialize() ([]byte, error) {
	return serializer.SerializeAny(r.Discriminator, r.Generator,
		serializer.Int(r.RandomSize),
		serializer.Float64(r.DiscountFactor))
}

// Gradient computes the gradient to be descended for the
// next training step.
func (r *Recurrent) Gradient(s sgd.SampleSet) autofunc.Gradient {
	genGrad := autofunc.NewGradient(r.Generator.(sgd.Learner).Parameters())
	discGrad := autofunc.NewGradient(r.Discriminator.(sgd.Learner).Parameters())

	subIdx := r.iterIdx % (r.GenIterations + r.DiscIterations)
	r.iterIdx++

	if subIdx < r.DiscIterations {
		r.DiscCost(s).PropagateGradient([]float64{1}, discGrad)
		if r.DiscTrans != nil {
			discGrad = r.DiscTrans.Transform(discGrad)
		}
	} else {
		genIn := r.generatorSeed(s)
		genOut := r.Generator.ApplySeqs(genIn)
		upstream := r.sampleReward(genOut)
		genOut.PropagateGradient(upstream, genGrad)
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

// DiscCost samples the discriminator cost.
func (r *Recurrent) DiscCost(s sgd.SampleSet) autofunc.Result {
	genIn := r.generatorSeed(s)
	genOut := r.Generator.ApplySeqs(genIn)
	sv := r.sampleGenSeq(genOut)
	genClassifications := r.Discriminator.ApplySeqs(seqfunc.ConstResult(sv))

	realIn := r.inputSequences(s)
	realClassifications := r.Discriminator.ApplySeqs(realIn)

	genCostFunc := func(a autofunc.Result) autofunc.Result {
		return neuralnet.SigmoidCECost{}.Cost([]float64{0}, a)
	}
	realCostFunc := func(a autofunc.Result) autofunc.Result {
		return neuralnet.SigmoidCECost{}.Cost([]float64{1}, a)
	}

	return autofunc.Add(
		seqfunc.AddAll(seqfunc.Map(realClassifications, realCostFunc)),
		seqfunc.AddAll(seqfunc.Map(genClassifications, genCostFunc)),
	)
}

// GenReward samples the generator reward.
func (r *Recurrent) GenReward(s sgd.SampleSet) float64 {
	genIn := r.generatorSeed(s)
	genOut := r.Generator.ApplySeqs(genIn)
	var sum float64
	for _, x := range r.sampleReward(genOut) {
		for _, y := range x {
			for _, k := range y {
				sum += k
			}
		}
	}
	return sum
}

// sampleReward samples from the policy's log-probability
// outputs.
// The resulting sequences are the same "shape" as policyOut,
// but the only non-zero entries are in the positions where
// the sampled character was chosen, and those entries are
// equal to -1 times the cumulative reward from that point
// onward.
func (r *Recurrent) sampleReward(policyOut seqfunc.Result) [][]linalg.Vector {
	sv := r.sampleGenSeq(policyOut)
	out := seqfunc.Map(r.Discriminator.ApplySeqs(seqfunc.ConstResult(sv)),
		autofunc.Sigmoid{}.Apply).OutputSeqs()

	cumulativeRewards := make([][]linalg.Vector, len(sv))
	for i, outSeq := range out {
		var cumulative float64
		cr := make([]linalg.Vector, len(outSeq))
		for j := len(outSeq) - 1; j >= 0; j-- {
			cumulative += outSeq[j][0]
			cr[j] = sv[i][j].Scale(-cumulative)
			if r.DiscountFactor != 0 {
				cumulative *= r.DiscountFactor
			}
		}
		cumulativeRewards[i] = cr
	}

	return cumulativeRewards
}

func (r *Recurrent) sampleGenSeq(policyOut seqfunc.Result) [][]linalg.Vector {
	var sampledVecs [][]linalg.Vector
	for _, seq := range policyOut.OutputSeqs() {
		var sampledVec []linalg.Vector
		for _, vec := range seq {
			choice := sampleVector(vec)
			v := make(linalg.Vector, len(vec))
			v[choice] = 1
			sampledVec = append(sampledVec, v)
		}
		sampledVecs = append(sampledVecs, sampledVec)
	}
	return sampledVecs
}

func (r *Recurrent) generatorSeed(s sgd.SampleSet) seqfunc.Result {
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

func (r *Recurrent) inputSequences(s sgd.SampleSet) seqfunc.Result {
	var res [][]linalg.Vector
	for i := 0; i < s.Len(); i++ {
		res = append(res, s.GetSample(i).(seqtoseq.Sample).Inputs)
	}
	return seqfunc.ConstResult(res)
}

func sampleVector(v linalg.Vector) int {
	n := rand.Float64()
	for i, x := range v {
		n -= math.Exp(x)
		if n < 0 {
			return i
		}
	}
	return 0
}
