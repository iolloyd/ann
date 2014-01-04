package nn

import (
	"fmt"
	"math/rand"
	"../node"
)

type NN struct {
	Inputs  []node.Node
	Hidden  []node.Node
	Outputs []node.Node
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
		layer[i].Ins = []*node.Node{}
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

func MakeNeuralNetwork(inAmount, hidAmount, outAmount int) NN {
	inLayer := makeInLayer(inAmount)
	hiddenLayer := makeHiddenLayer(inLayer, hidAmount)
	outLayer := makeOutLayer(hiddenLayer, outAmount)
	nn := NN{inLayer, hiddenLayer, outLayer}
	return nn
}

func (self *NN) Train(td TrainingData) {
	for x := 0; x < len(td.Inputs); x++ {
        self.setInputs(td.Inputs[x])
        self.setOutputs(td.Outputs[x])
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


