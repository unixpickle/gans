package main

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	var q QuadSquash
	serializer.RegisterTypedDeserializer(q.SerializerType(), DeserializeQuadSquash)
}

// QuadSquash cubes its arguments and then normalizes
// them to be probabilities.
type QuadSquash struct{}

func DeserializeQuadSquash(d []byte) (QuadSquash, error) {
	return QuadSquash{}, nil
}

func (_ QuadSquash) Apply(in autofunc.Result) autofunc.Result {
	tanh := neuralnet.HyperbolicTangent{}
	tan := tanh.Apply(in)
	softmax := autofunc.Softmax{}
	return softmax.Apply(autofunc.Scale(tan, 4))
}

func (_ QuadSquash) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	panic("not yet implemented")
}

func (_ QuadSquash) SerializerType() string {
	return "github.com/unixpickle/gans/text_gen.QuadSquash"
}

func (_ QuadSquash) Serialize() ([]byte, error) {
	return []byte{}, nil
}
