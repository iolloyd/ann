package nn

import (
	"fmt"
	"math"
	"math/rand"
	"../node"
)

type NN struct {
	Inputs  []node.Node
	Hidden  []node.Node
	Outputs []node.Node
	Error float64
}

type Layer []node.Node

func makeLayer(amount int) Layer {
	nodes := []node.Node{}
	for x := 0; x < amount; x++ {
		nodes = append(nodes, *node.MakeNode())
	}
	return nodes
}

func makeInLayer(amount int) Layer {
	layer := makeLayer(amount)
	for i, _ := range layer {
		layer[i].InputNodes = []*node.Node{}
		layer[i].Value = 0
	}
	return layer
}

func makeHiddenLayer(inputLayer Layer, amount int) Layer {
	layer := makeLayer(amount)
	for n := range layer {
		for m := range inputLayer {
			layer[n].AddNode(&inputLayer[m])
		}
	}
	return layer
}

func makeOutLayer(hiddenLayer Layer, amount int) Layer {
	layer := makeLayer(amount)
	for n := range layer {
		for m := range hiddenLayer {
			layer[n].AddNode(&hiddenLayer[m])
		}
	}
	return layer
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -float64(x))) 
}

func dsigmoid(y float64) float64 {
	return y * (1.0 - y)
}

func MakeNN(inAmount, hidAmount, outAmount int, errorThreshold float64) NN {
	inLayer := makeInLayer(inAmount)
	hiddenLayer := makeHiddenLayer(inLayer, hidAmount)
	outLayer := makeOutLayer(hiddenLayer, outAmount)
	nn := NN{inLayer, hiddenLayer, outLayer, errorThreshold}
	return nn
}

func (self *NN) Train(td TrainingData, iterations int) {
	for x := 0; x < len(td.Inputs); x++ {
        self.setInputs(td.Inputs[x])
        self.setOutputs(td.Outputs[x])
        self.startIterations(iterations)
    }
}

func (self *NN) setInputs(values []float64) {
    for i, value := range values {
        self.Inputs[i].Value = value
    }
}

func (self *NN) setOutputs(values []float64) {
	for i, value := range values {
    	self.Outputs[i].Err = value
    	self.Outputs[i].Value = rand.Float64() * 2 
    }
}

func (self *NN) startIterations(iterations int) {
	for x := 0; x < iterations; x++ {
    	self.FeedForward()
    	self.FeedBackward()
    }
    
}

// Adjust the values by recursively summing the
// weighted values of the incoming connections to the 
// out nodes and their subsequent connections
func (self *NN) FeedForward() {
	for j := 0; j < len(self.Outputs); j++ {
    	currentNode := self.Outputs[j]
    	sum := 0.0
    	for k := 0; k < len(currentNode.InputNodes); k++ {
        	sum += currentNode.InputNodes[k].GetFeedForwardValue()
        }
        currentNode.Value = sum
    }
}

// Adjust the weights
//
// We need to reset the error and update 
// weights on outputs
func (self *NN) FeedBackward() {
}

func (self *NN) Show() {
	for x := range self.Inputs{
    	n := self.Inputs[x]
    	fmt.Println("Input")
    	n.Show()
    	fmt.Println("-----")
    }
	for x := 0; x < len(self.Outputs); x++ {
		n := self.Outputs[x]
    	fmt.Println("Output")
    	n.Show()
    	fmt.Println("-----")
    }

}


