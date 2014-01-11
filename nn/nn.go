package nn

import (
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

	// Add each member of the input layer as an input node 
	// to each member of the created layer
	for x := 0; x < len(layer); x++ {
		for n := 0; n < len(inputLayer); n++ {
			layer[x].InputNodes = append(layer[x].InputNodes, &inputLayer[n])
		}
	}

	// Add each member of the created layer as an output node 
	// to each member of the input layer
	for x := 0; x < len(inputLayer); x++ {
		for n := 0; n < len(layer); n++ {
			inputLayer[x].OutputNodes = append(inputLayer[x].OutputNodes, &layer[n])
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
	hiddenLayer := makeJoinedLayer(inLayer, hidAmount)
	outLayer := makeJoinedLayer(hiddenLayer, outAmount)
	nn := NN{inLayer, hiddenLayer, outLayer, errorThreshold}
	return nn
}

func (self *NN) Train(td TrainingData, iterations int, rate float64) {
	self.randomizeWeights()
	for n := 0; n < iterations; n++ {
        for x := 0; x < len(td.Inputs); x++ {
            self.setInputs(td.Inputs[x])
            self.FeedBackward(td.Outputs[0], rate)
        }
    }
}

func (self *NN) randomizeWeights() {
    self.randomizeLayer(self.Inputs)
    self.randomizeLayer(self.Hidden)
    self.randomizeLayer(self.Outputs)
}

func (self *NN) randomizeLayer(layer []node.Node) {
    for x := 0; x < len(layer); x++ {
        weights := []float64{}
        for y := 0; y < len(layer[x].InputNodes); y++ {
            randomWeight := rand.Float64() * 10
            weights = append(weights, randomWeight) 
        }
        layer[x].InputWeights = weights
    }


}

func (self *NN) setInputs(values []float64) {
    for i, value := range values {
        self.Inputs[i].Value = value
    }
}

func (self *NN) FeedBackward(targets []float64, rate float64) {
    self.updateWeights(targets, rate)
}

func (self *NN) updateWeights(targets []float64, rate float64) {
    for x := 0; x < len(self.Outputs); x++ {
        self.Outputs[x].UpdateWeights(targets[x], rate)
    }
}

func (self *NN) Show() {
	for x := range self.Inputs{self.Inputs[x].Show()}
	for x := range self.Outputs {self.Outputs[x].Show()}
}


