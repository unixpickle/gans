package gans

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn"
)

func init() {
	var o OneHotLayer
	serializer.RegisterTypedDeserializer(o.SerializerType(), DeserializeOneHotLayer)

	var f FeedbackBlock
	serializer.RegisterTypedDeserializer(f.SerializerType(), DeserializeFeedbackBlock)
}

// A OneHotLayer generates a one-hot-vector by applying
// softmax to the input and choosing an index in a random
// biased fashion based on the resulting probabilities.
type OneHotLayer struct{}

// DeserializeOneHotLayer deserializes a OneHotLayer.
func DeserializeOneHotLayer(d []byte) (OneHotLayer, error) {
	return OneHotLayer{}, nil
}

// Apply applies the layer to an input.
func (_ OneHotLayer) Apply(in autofunc.Result) autofunc.Result {
	if len(in.Output()) == 0 {
		return in
	}
	softmax := autofunc.Softmax{}
	sm := softmax.Apply(in)
	idx := chooseRandom(sm.Output())
	mask := make(linalg.Vector, len(sm.Output()))
	mask[idx] = 1 / sm.Output()[idx]
	return autofunc.Mul(&autofunc.Variable{Vector: mask}, sm)
}

// ApplyR is like Apply for RResults.
func (_ OneHotLayer) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	if len(in.Output()) == 0 {
		return in
	}
	softmax := autofunc.Softmax{}
	sm := softmax.ApplyR(rv, in)
	idx := chooseRandom(sm.Output())
	mask := make(linalg.Vector, len(sm.Output()))
	mask[idx] = 1 / sm.Output()[idx]
	return autofunc.MulR(autofunc.NewRVariable(&autofunc.Variable{Vector: mask}, rv), sm)
}

// SerializerType returns the unique ID used to serialize
// a OneHotLayer with the serializer package.
func (_ OneHotLayer) SerializerType() string {
	return "github.com/unixpickle/gans.OneHotLayer"
}

// Serialize serializes the layer.
func (_ OneHotLayer) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func chooseRandom(in linalg.Vector) int {
	n := rand.Float64()
	for i, x := range in {
		n -= x
		if n < 0 {
			return i
		}
	}
	return len(in) - 1
}

type feedbackBlockState struct {
	Internal rnn.State
	Feedback linalg.Vector
}

type feedbackBlockRState struct {
	Internal  rnn.RState
	Feedback  linalg.Vector
	FeedbackR linalg.Vector
}

type feedbackBlockGrad struct {
	Internal rnn.StateGrad
	Feedback linalg.Vector
}

type feedbackBlockRGrad struct {
	Internal  rnn.StateGrad
	Feedback  linalg.Vector
	FeedbackR linalg.Vector
}

// A FeedbackBlock feeds a blocks output back into its
// input in the next timestep.
//
// More specifically, suppose the wrapped block outputs
// a vector v at time t and let u be the external input
// vector at time t+1.
// The wrapped block would receive input (u, v).
// That is, the last output would be appended to the end
// of the actual input for the timestep.
type FeedbackBlock struct {
	// B is the wrapped block.
	B rnn.Block

	// InitFeedback is the vector fed into the block in the
	// first timestep.
	InitFeedback *autofunc.Variable
}

// NewFeedbackBlock creates a FeedbackBlock with a zero'd
// initial feedback variable.
func NewFeedbackBlock(b rnn.Block, outSize int) *FeedbackBlock {
	return &FeedbackBlock{
		B: b,
		InitFeedback: &autofunc.Variable{
			Vector: make(linalg.Vector, outSize),
		},
	}
}

// DeserializeFeedbackBlock deserializes a FeedbackBlock.
func DeserializeFeedbackBlock(d []byte) (*FeedbackBlock, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) != 2 {
		return nil, errors.New("bad FeedbackBlock slice")
	}
	blockObj, ok1 := slice[0].(rnn.Block)
	initObj, ok2 := slice[1].(serializer.Bytes)
	if !ok1 || !ok2 {
		return nil, errors.New("bad FeedbackBlock slice")
	}
	var feedback autofunc.Variable
	if err := json.Unmarshal([]byte(initObj), &feedback); err != nil {
		return nil, err
	}
	return &FeedbackBlock{
		B:            blockObj,
		InitFeedback: &feedback,
	}, nil
}

// StartState returns the start state of the block.
func (f *FeedbackBlock) StartState() rnn.State {
	return &feedbackBlockState{
		Internal: f.B.StartState(),
		Feedback: f.InitFeedback.Vector,
	}
}

// StartRState returns the start state of the block.
func (f *FeedbackBlock) StartRState(rv autofunc.RVector) rnn.RState {
	v := autofunc.NewRVariable(f.InitFeedback, rv)
	return &feedbackBlockRState{
		Internal:  f.B.StartRState(rv),
		Feedback:  v.Output(),
		FeedbackR: v.ROutput(),
	}
}

// PropagateStart propagates through the start state.
func (f *FeedbackBlock) PropagateStart(s []rnn.State, u []rnn.StateGrad, g autofunc.Gradient) {
	var internal []rnn.StateGrad
	var internalStart []rnn.State
	for i, uGradObj := range u {
		uGrad := uGradObj.(*feedbackBlockGrad)
		f.InitFeedback.PropagateGradient(uGrad.Feedback, g)

		internal = append(internal, uGrad.Internal)
		state := s[i].(*feedbackBlockState)
		internalStart = append(internalStart, state.Internal)
	}
	f.B.PropagateStart(internalStart, internal, g)
}

// PropagateStartR propagates through the start state.
func (f *FeedbackBlock) PropagateStartR(s []rnn.RState, u []rnn.RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) {
	var internal []rnn.RStateGrad
	var internalStart []rnn.RState
	for i, uGradObj := range u {
		uGrad := uGradObj.(*feedbackBlockRGrad)
		rv := &autofunc.RVariable{Variable: f.InitFeedback}
		rv.PropagateRGradient(uGrad.Feedback, uGrad.FeedbackR, rg, g)

		internal = append(internal, uGrad.Internal)
		state := s[i].(*feedbackBlockRState)
		internalStart = append(internalStart, state.Internal)
	}
	f.B.PropagateStartR(internalStart, internal, rg, g)
}

// ApplyBlock applies the block to a batch of inputs.
func (f *FeedbackBlock) ApplyBlock(s []rnn.State, in []autofunc.Result) rnn.BlockResult {
	var poolVars []*autofunc.Variable
	var joinedIn []autofunc.Result
	var internalState []rnn.State
	for i, stateObj := range s {
		state := stateObj.(*feedbackBlockState)
		v := &autofunc.Variable{Vector: state.Feedback}
		poolVars = append(poolVars, v)
		joinedIn = append(joinedIn, autofunc.Concat(in[i], v))
		internalState = append(internalState, state.Internal)
	}

	internalOut := f.B.ApplyBlock(internalState, joinedIn)

	outStates := make([]rnn.State, len(internalOut.States()))
	for i, internalOutState := range internalOut.States() {
		outStates[i] = &feedbackBlockState{
			Internal: internalOutState,
			Feedback: internalOut.Outputs()[i],
		}
	}

	return &feedbackBlockResult{
		FeedbackPool: poolVars,
		Internal:     internalOut,
		OutStates:    outStates,
	}
}

// ApplyBlockR applies the block to a batch of inputs.
func (f *FeedbackBlock) ApplyBlockR(rv autofunc.RVector, s []rnn.RState,
	in []autofunc.RResult) rnn.BlockRResult {
	var poolVars []*autofunc.Variable
	var joinedIn []autofunc.RResult
	var internalState []rnn.RState
	for i, stateObj := range s {
		state := stateObj.(*feedbackBlockRState)
		v := &autofunc.Variable{Vector: state.Feedback}
		rVar := &autofunc.RVariable{
			Variable:   v,
			ROutputVec: state.FeedbackR,
		}
		poolVars = append(poolVars, v)
		joinedIn = append(joinedIn, autofunc.ConcatR(in[i], rVar))
		internalState = append(internalState, state.Internal)
	}

	internalOut := f.B.ApplyBlockR(rv, internalState, joinedIn)

	outStates := make([]rnn.RState, len(internalOut.RStates()))
	for i, internalOutState := range internalOut.RStates() {
		outStates[i] = &feedbackBlockRState{
			Internal:  internalOutState,
			Feedback:  internalOut.Outputs()[i],
			FeedbackR: internalOut.ROutputs()[i],
		}
	}

	return &feedbackBlockRResult{
		FeedbackPool: poolVars,
		Internal:     internalOut,
		OutStates:    outStates,
	}
}

// Parameters returns all of the internal block's
// parameters (if it has any) plus the initial feedback
// variable.
func (f *FeedbackBlock) Parameters() []*autofunc.Variable {
	res := []*autofunc.Variable{f.InitFeedback}
	if l, ok := f.B.(sgd.Learner); ok {
		res = append(res, l.Parameters()...)
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a FeedbackBlock with the serializer package.
func (f *FeedbackBlock) SerializerType() string {
	return "github.com/unixpickle/gans.FeedbackBlock"
}

// Serialize serializes the block.
func (f *FeedbackBlock) Serialize() ([]byte, error) {
	blockSer, ok := f.B.(serializer.Serializer)
	if !ok {
		return nil, fmt.Errorf("type is not a Serializer: %T", f.B)
	}
	data, err := json.Marshal(f.InitFeedback)
	if err != nil {
		return nil, err
	}
	return serializer.SerializeSlice([]serializer.Serializer{
		blockSer,
		serializer.Bytes(data),
	})
}

type feedbackBlockResult struct {
	FeedbackPool []*autofunc.Variable
	Internal     rnn.BlockResult
	OutStates    []rnn.State
}

func (f *feedbackBlockResult) Outputs() []linalg.Vector {
	return f.Internal.Outputs()
}

func (f *feedbackBlockResult) States() []rnn.State {
	return f.OutStates
}

func (f *feedbackBlockResult) PropagateGradient(u []linalg.Vector, s []rnn.StateGrad,
	g autofunc.Gradient) []rnn.StateGrad {
	internalUpstream := make([]linalg.Vector, len(f.Outputs()))
	for i, x := range u {
		internalUpstream[i] = x.Copy()
	}

	internalStateGrad := make([]rnn.StateGrad, len(f.Outputs()))
	for i, stateGradObj := range s {
		if stateGradObj != nil {
			stateGrad := stateGradObj.(*feedbackBlockGrad)
			internalStateGrad[i] = stateGrad.Internal
			if internalUpstream[i] != nil {
				internalUpstream[i].Add(stateGrad.Feedback)
			} else {
				internalUpstream[i] = stateGrad.Feedback.Copy()
			}
		}
	}

	for i, x := range internalUpstream {
		if x == nil {
			internalUpstream[i] = make(linalg.Vector, len(f.Outputs()[i]))
		}
	}

	for _, v := range f.FeedbackPool {
		g[v] = make(linalg.Vector, len(v.Vector))
	}

	internalDownstream := f.Internal.PropagateGradient(internalUpstream,
		internalStateGrad, g)

	downstream := make([]rnn.StateGrad, len(f.Outputs()))
	for i, v := range f.FeedbackPool {
		feedbackDown := g[v]
		delete(g, v)
		downstream[i] = &feedbackBlockGrad{
			Internal: internalDownstream[i],
			Feedback: feedbackDown,
		}
	}

	return downstream
}

type feedbackBlockRResult struct {
	FeedbackPool []*autofunc.Variable
	Internal     rnn.BlockRResult
	OutStates    []rnn.RState
}

func (f *feedbackBlockRResult) Outputs() []linalg.Vector {
	return f.Internal.Outputs()
}

func (f *feedbackBlockRResult) ROutputs() []linalg.Vector {
	return f.Internal.ROutputs()
}

func (f *feedbackBlockRResult) RStates() []rnn.RState {
	return f.OutStates
}

func (f *feedbackBlockRResult) PropagateRGradient(u, uR []linalg.Vector, s []rnn.RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) []rnn.RStateGrad {
	internalUpstream := make([]linalg.Vector, len(f.Outputs()))
	internalUpstreamR := make([]linalg.Vector, len(f.Outputs()))
	for i, x := range u {
		internalUpstream[i] = x.Copy()
		internalUpstreamR[i] = uR[i].Copy()
	}

	internalStateGrad := make([]rnn.RStateGrad, len(f.Outputs()))
	for i, stateGradObj := range s {
		if stateGradObj != nil {
			stateGrad := stateGradObj.(*feedbackBlockRGrad)
			internalStateGrad[i] = stateGrad.Internal
			if internalUpstream[i] != nil {
				internalUpstream[i].Add(stateGrad.Feedback)
				internalUpstreamR[i].Add(stateGrad.FeedbackR)
			} else {
				internalUpstream[i] = stateGrad.Feedback.Copy()
				internalUpstreamR[i] = stateGrad.FeedbackR.Copy()
			}
		}
	}

	for i, x := range internalUpstream {
		if x == nil {
			internalUpstream[i] = make(linalg.Vector, len(f.Outputs()[i]))
			internalUpstreamR[i] = internalUpstream[i]
		}
	}

	if g == nil {
		g = autofunc.Gradient{}
	}
	for _, v := range f.FeedbackPool {
		g[v] = make(linalg.Vector, len(v.Vector))
		rg[v] = make(linalg.Vector, len(v.Vector))
	}

	internalDownstream := f.Internal.PropagateRGradient(internalUpstream, internalUpstreamR,
		internalStateGrad, rg, g)

	downstream := make([]rnn.RStateGrad, len(f.Outputs()))
	for i, v := range f.FeedbackPool {
		feedbackDown := g[v]
		feedbackDownR := rg[v]
		delete(g, v)
		delete(rg, v)
		downstream[i] = &feedbackBlockRGrad{
			Internal:  internalDownstream[i],
			Feedback:  feedbackDown,
			FeedbackR: feedbackDownR,
		}
	}

	return downstream
}
