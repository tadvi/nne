package nne

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// sigmoid helper function
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Node for backpropagation training based feed forward network.
type Node struct {
	Threshold float64 // threshold
	Weights   []float64

	activation float64 // activation value
	error      float64
}

// NewNode creates new backpropagation network node.
func NewNode(wCount int) *Node {
	return &Node{Weights: make([]float64, wCount, wCount)}
}

// Network is feed forward backpropagation network.
// Public members can be persisted to json or database.
type Network struct {
	Input  []*Node
	Hidden []*Node
	Output []*Node

	lhRate float64 // learning rate of the hidden layer
	loRate float64 // learning rate of the output layer

	netInput   []float64
	desiredOut []float64
}

// NewNetwork creates new backpropagation network with input, hidden and output layers.
func NewNetwork(inCount, hideCount, outCount int) *Network {
	n := &Network{
		lhRate: 0.15,
		loRate: 0.2,
		Input:  make([]*Node, inCount, inCount),
		Hidden: make([]*Node, hideCount, hideCount),
		Output: make([]*Node, outCount, outCount),
	}

	rand.Seed(time.Now().Unix())
	for i := 0; i < inCount; i++ {
		n.Input[i] = NewNode(hideCount)
		for j := 0; j < hideCount; j++ {
			n.Input[i].Weights[j] = rand.Float64() - 0.49999
		}
	}

	for i := 0; i < hideCount; i++ {
		n.Hidden[i] = NewNode(outCount)
		for j := 0; j < outCount; j++ {
			n.Hidden[i].Weights[j] = rand.Float64()
		}
	}
	for i := 0; i < outCount; i++ {
		n.Output[i] = NewNode(0)
	}

	// reset thresholds
	for i := 0; i < len(n.Hidden); i++ {
		n.Hidden[i].Threshold = rand.Float64()
	}
	for i := 0; i < len(n.Output); i++ {
		n.Output[i].Threshold = rand.Float64()
	}

	return n
}

// TrainingData holds single block of inputs and outputs for the training to run.
// It is public for easier persistence to disk or database.
type TrainingData struct {
	Input  []float64
	Output []float64
}

type TrainingSet []*TrainingData

func (tr *TrainingSet) Add(input []float64, output []float64) {
	*tr = append(*tr, &TrainingData{Input: input, Output: output})
}

// Train performs network training for number of iterations, usually over 2000 epochs.
func (n *Network) Train(epochs int, data TrainingSet) {
	inputLen := len(n.Input)
	outputLen := len(n.Output)

	for i := 0; i < epochs; i++ {
		for _, tr := range data {
			if inputLen != len(tr.Input) {
				panic(fmt.Sprintf("expected training data input length %d got %d", inputLen, len(tr.Input)))
			}
			if outputLen != len(tr.Output) {
				panic(fmt.Sprintf("expected traing data output length %d got %d", outputLen, len(tr.Output)))
			}
			n.netInput = tr.Input
			n.desiredOut = tr.Output
			n.trainOnePattern()
		}
	}

}

// TrainOnePattern train single pattern.
func (n *Network) trainOnePattern() {
	n.calcActivation()
	n.calcErrorOutput()
	n.calcErrorHidden()
	n.calcNewThresholds()
	n.calcNewWeightsHidden()
	n.calcNewWeightsInput()
}

// SetLearningRate sets learning rate for the backpropagation.
func (n *Network) SetLearningRates(lhRate, loRate float64) {
	n.lhRate = lhRate
	n.loRate = loRate
}

func (n *Network) calcActivation() {
	// a loop to set the activations of the hidden layer
	for h := 0; h < len(n.Hidden); h++ {
		for i := 0; i < len(n.Input); i++ {
			n.Hidden[h].activation += n.netInput[i] * n.Input[i].Weights[h]
		}
	}

	// calculate the output of the hidden
	for _, hid := range n.Hidden {
		hid.activation += hid.Threshold
		hid.activation = sigmoid(hid.activation)
	}

	// a loop to set the activations of the output layer
	for j, val := range n.Output {
		for _, hid := range n.Hidden {
			val.activation += hid.activation * hid.Weights[j]
		}
	}

	// calculate the output of the output layer
	for _, val := range n.Output {
		val.activation += val.Threshold
		val.activation = sigmoid(val.activation)
	}

}

// calcErrorOutput calculates error of each output neuron.
func (n *Network) calcErrorOutput() {
	for j := 0; j < len(n.Output); j++ {
		n.Output[j].error = n.Output[j].activation * (1 - n.Output[j].activation) *
			(n.desiredOut[j] - n.Output[j].activation)
	}
}

// calcErrorHidden calculate error of each hidden neuron.
func (n *Network) calcErrorHidden() {
	for h := 0; h < len(n.Hidden); h++ {
		for j := 0; j < len(n.Output); j++ {
			n.Hidden[h].error += n.Hidden[h].Weights[j] * n.Output[j].error
		}
		n.Hidden[h].error *= n.Hidden[h].activation * (1 - n.Hidden[h].activation)
	}
}

// calcNewThresholds calculate new thresholds for each neuron.
func (n *Network) calcNewThresholds() {
	// computing the thresholds for next iteration for hidden layer
	for h := 0; h < len(n.Hidden); h++ {
		n.Hidden[h].Threshold += n.Hidden[h].error * n.lhRate
	}
	// computing the thresholds for next iteration for output layer
	for j := 0; j < len(n.Output); j++ {
		n.Output[j].Threshold += n.Output[j].error * n.loRate
	}

}

// calcNewWeightsHidden calculate new weights between hidden and output.
func (n *Network) calcNewWeightsHidden() {
	for h := 0; h < len(n.Hidden); h++ {
		temp := n.Hidden[h].activation * n.loRate
		for j := 0; j < len(n.Output); j++ {
			n.Hidden[h].Weights[j] += temp * n.Output[j].error
		}
	}
}

// calcNewWeightsInput .
func (n *Network) calcNewWeightsInput() {
	for i := 0; i < len(n.netInput); i++ {
		temp := n.netInput[i] * n.lhRate
		for h := 0; h < len(n.Hidden); h++ {
			n.Input[i].Weights[h] += temp * n.Hidden[h].error
		}
	}
}

// calcTotalErrorPattern.
func (n *Network) calcTotalError() float64 {
	temp := 0.0
	for j := 0; j < len(n.Output); j++ {
		temp += n.Output[j].error
	}
	return temp
}

// Results calculates network outputs and returns raw float64 activation values.
func (n *Network) Results(input []float64) []float64 {
	n.netInput = input
	n.calcActivation()
	out := make([]float64, len(n.Output), len(n.Output))
	for i, node := range n.Output {
		out[i] = node.activation
	}
	return out
}

// Result calculates network output value for single output node networks.
func (n *Network) Result(input []float64) float64 {
	if len(n.Output) != 1 {
		panic("nne: network output must have only 1 output node.")
	}
	n.netInput = input
	n.calcActivation()
	node := n.Output[0]
	return node.activation
}
