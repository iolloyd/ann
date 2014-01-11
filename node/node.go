package node

import (
	"fmt"
)

type Node struct {
	Index int
	InputNodes, OutputNodes  []*Node
	Value, Err, Target  float64
	InputWeights    []float64
}

func MakeNode(index int) *Node {
	n := &Node{index, []*Node{}, []*Node{}, 0, 0, 0, []float64{}}
	return n
}

func (self *Node) GetValue() float64 {
    value := 0.0
    for x := 0; x < len(self.InputNodes); x++ {
        value += self.InputNodes[x].GetValue() * self.InputWeights[x]
    }
    return self.Value + value
}

func (self *Node) CalculateError(target float64) {
    err := 0.0
    for x := 0; x < len(self.OutputNodes); x++ {
        err += self.OutputNodes[x].Err
    }
    self.Err += err
}

func (self *Node) UpdateWeights(target float64, rate float64) {
    for x := 0; x < len(self.InputWeights); x++ {
        if self.GetValue() > target {
            self.InputWeights[x] -= rate
        } else {
            self.InputWeights[x] += rate
        }
    }
}

func (self *Node) Show() {
	fmt.Printf("Index: %d, Value: %f\n", self.Index, self.GetValue())
}

