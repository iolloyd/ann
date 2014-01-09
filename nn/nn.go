package nn

import (
	"fmt"
	"math"
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
		nodes = append(nodes, *node.MakeNode(x))
	}
	return nodes
}

func makeInLayer(amount int) Layer {
	layer := makeLayer(amount)
	for x := 0; x < len(layer); x++ {
		layer[x].InputNodes = []*node.Node{}
		layer[x].Value = 0
	}
	return layer
}

func makeJoinedLayer(inputLayer Layer, amount int) Layer {
	layer := makeLayer(amount)
	for x := 0; x < len(layer); x++ {
		nodes := []*node.Node{}
		for n := 0; n < len(inputLayer); n++ {
			nodes = append(nodes, &inputLayer[n])
		}
        layer[x].InputNodes = nodes
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
	hiddenLayer := makeJoinedLayer(inLayer, hidAmount)
	outLayer := makeJoinedLayer(hiddenLayer, outAmount)
	nn := NN{inLayer, hiddenLayer, outLayer, errorThreshold}
	return nn
}

func (self *NN) Train(td TrainingData, iterations int) {
	for x := 0; x < len(td.Inputs); x++ {
        self.setInputs(td.Inputs[x])
        self.FeedForward()
    }
}

func (self *NN) setInputs(values []float64) {
    for i, value := range values {
        self.Inputs[i].Value = value
    }
}

func (self *NN) FeedForward() {
	for j := 0; j < len(self.Outputs); j++ {
    	self.Outputs[j].GetValue()
    }
}

func (self *NN) FeedBackward() {
}

func (self *NN) Show() {

	fmt.Println("INPUTS")

	for x := range self.Inputs{
    	n := self.Inputs[x]
    	n.Show()
    }

	fmt.Println("OUTPUTS")

	for x := 0; x < len(self.Outputs); x++ {
		n := self.Outputs[x]
    	n.Show()
    }

}


