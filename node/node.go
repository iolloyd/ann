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

func (self *Node) AddNode(node *Node) {
	node.OutputNodes = append(node.OutputNodes, self)
}

func (self *Node) GetValue() float64 {
    value := 0.0
    tot := len(self.InputNodes)
    for x := 0; x < tot; x++ {
        value += self.InputNodes[x].GetValue()
    }
    return self.Value + value
}

func (self *Node) UpdateError() {
    err := 0.0
    for x := 0; x < len(self.OutputNodes); x++ {
        err += self.OutputNodes[x].Err
    }
    self.Err += err
}

func (self *Node) GetTotalInput() float64 {
	total := 0.0
	for x := 0; x < len(self.InputNodes); x++ {
		total += self.InputNodes[x].Value * self.InputWeights[x]
    }
    return total
}

func (self *Node) Show() {
	fmt.Printf("Index: %d, Value: %f\n", self.Index, self.GetValue())
}

