package node

import (
	"fmt"
)

type Node struct {
	InputNodes, OutputNodes  []*Node
	Value, Err, Target  float64
	InputWeights    []float64
}

func MakeNode() *Node {
	n := &Node{[]*Node{}, []*Node{}, 0, 0, 0, []float64{}}
	return n
}

func (self *Node) AddNode(node *Node) {
	node.OutputNodes = append(node.OutputNodes, self)
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
	fmt.Printf("Input Node %f\n", self.Value)
	fmt.Printf("With Error %f\n", self.Err)
}

func (self *Node) GetFeedForwardValue() float64 {
	sum := 0.0
	for x := 0; x < len(self.InputNodes); x++ {
		sum += self.InputNodes[x].GetFeedForwardValue()
    }
    return self.Value + sum
}
