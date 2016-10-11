package main

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
)

func init() {
	var q QuadSquash
	serializer.RegisterTypedDeserializer(q.SerializerType(), DeserializeQuadSquash)
}

// QuadSquash squares its arguments and then normalizes
// them to be probabilities.
type QuadSquash struct{}

func DeserializeQuadSquash(d []byte) (QuadSquash, error) {
	return QuadSquash{}, nil
}

func (_ QuadSquash) Apply(in autofunc.Result) autofunc.Result {
	square := autofunc.Square(in)
	sum := autofunc.SumAll(square)
	return autofunc.ScaleFirst(square, autofunc.Inverse(sum))
}

func (_ QuadSquash) SerializerType() string {
	return "github.com/unixpickle/gans/text_gen.QuadSquash"
}

func (_ QuadSquash) Serialize() ([]byte, error) {
	return []byte{}, nil
}
