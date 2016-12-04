package main

import (
	"encoding/json"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	expBlockIdentityScale = 0.8
	expWeightInit         = 0.01
)

func init() {
	var e expBlockFunc
	serializer.RegisterTypedDeserializer(e.SerializerType(), deserializeExpBlockFunc)
}

func NewExpBlock(inSize, stateSize int) rnn.Block {
	bf := &expBlockFunc{
		StateTrans: &neuralnet.DenseLayer{
			InputCount:  stateSize,
			OutputCount: stateSize,
		},
		InputWeights: make([]*autofunc.Variable, inSize),
		InitState:    &autofunc.Variable{Vector: make(linalg.Vector, stateSize)},
	}

	// Only used to initialize vectors to the right sizes.
	bf.StateTrans.Randomize()

	bf.StateTrans.Weights.Data.Vector.Scale(0)
	bf.StateTrans.Biases.Var.Vector.Scale(0)
	for i := 0; i < stateSize; i++ {
		// Identity initializations, like an IRNN.
		bf.StateTrans.Weights.Data.Vector[i+stateSize*i] = expBlockIdentityScale
	}
	for i := range bf.InputWeights {
		bf.InputWeights[i] = &autofunc.Variable{Vector: make(linalg.Vector, stateSize)}
		for j := 0; j < stateSize; j++ {
			bf.InputWeights[i].Vector[j] = rand.NormFloat64() * expWeightInit
		}
	}
	return &rnn.StateOutBlock{
		Block: rnn.NewNetworkBlock(neuralnet.Network{bf}, stateSize),
	}
}

type expBlockFunc struct {
	StateTrans   *neuralnet.DenseLayer
	InputWeights []*autofunc.Variable
	InitState    *autofunc.Variable
}

func deserializeExpBlockFunc(d []byte) (*expBlockFunc, error) {
	var f expBlockFunc
	if err := json.Unmarshal(d, &f); err != nil {
		return nil, err
	}
	return &f, nil
}

func (e *expBlockFunc) Apply(in autofunc.Result) autofunc.Result {
	stateSize := len(e.StateTrans.Biases.Var.Vector)
	inSize := len(in.Output()) - stateSize
	return autofunc.Pool(in, func(in autofunc.Result) autofunc.Result {
		newState := e.StateTrans.Apply(autofunc.Slice(in, inSize, inSize+stateSize))
		return autofunc.Pool(newState, func(newState autofunc.Result) autofunc.Result {
			activation := neuralnet.HyperbolicTangent{}
			var expOut autofunc.Result
			for i := 0; i < inSize; i++ {
				prob := autofunc.Slice(in, i, i+1)
				totalState := autofunc.Add(newState, e.InputWeights[i])
				activated := activation.Apply(totalState)
				masked := autofunc.ScaleFirst(activated, prob)
				if expOut == nil {
					expOut = masked
				} else {
					expOut = autofunc.Add(expOut, masked)
				}
			}
			return expOut
		})
	})
}

func (e *expBlockFunc) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	stateSize := len(e.StateTrans.Biases.Var.Vector)
	inSize := len(in.Output()) - stateSize
	return autofunc.PoolR(in, func(in autofunc.RResult) autofunc.RResult {
		newState := e.StateTrans.ApplyR(rv, autofunc.SliceR(in, inSize, inSize+stateSize))
		return autofunc.PoolR(newState, func(newState autofunc.RResult) autofunc.RResult {
			activation := neuralnet.HyperbolicTangent{}
			var expOut autofunc.RResult
			for i := 0; i < inSize; i++ {
				prob := autofunc.SliceR(in, i, i+1)
				weight := autofunc.NewRVariable(e.InputWeights[i], rv)
				totalState := autofunc.AddR(newState, weight)
				activated := activation.ApplyR(rv, totalState)
				masked := autofunc.ScaleFirstR(activated, prob)
				if expOut == nil {
					expOut = masked
				} else {
					expOut = autofunc.AddR(expOut, masked)
				}
			}
			return expOut
		})
	})
}

func (e *expBlockFunc) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	res = append(res, e.InitState)
	res = append(res, e.InputWeights...)
	res = append(res, e.StateTrans.Parameters()...)
	return res
}

func (e *expBlockFunc) SerializerType() string {
	return "github.com/unixpickle/gans/demo/text_gen.expBlockFunc"
}

func (e *expBlockFunc) Serialize() ([]byte, error) {
	return json.Marshal(e)
}
