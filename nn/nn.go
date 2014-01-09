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
	self.randomizeWeights()
	for n := 0; n < iterations; n++ {
        for x := 0; x < len(td.Inputs); x++ {
            self.setInputs(td.Inputs[x])
            self.FeedBackward(td.Outputs[0])
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

func (self *NN) FeedBackward(targets []float64) {
    self.calculateErrors(targets)
    self.updateWeights()
}

func (self *NN) calculateErrors(targets []float64) {
    for x := 0; x < len(self.Outputs); x++ {
        cur := self.Outputs[x]
        cur.Err = sigmoid(cur.GetValue() - targets[x])
        for child := 0; child < len(cur.InputNodes); child++ {
            cur.InputNodes[child].CalculateError(cur.Err)
        }
    }
        
}

func (self *NN) updateWeights() {
    for x := 0; x < len(self.Outputs); x++ {
        self.Outputs[x].UpdateWeights()
    }
}

func (self *NN) Show() {


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


