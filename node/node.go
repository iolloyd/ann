package node

import (
	"fmt"
)

type Node struct {
	Ins, Outs  []*Node
	Value, Err float64
	Weights    []float64
}

func MakeNode() *Node {
	n := &Node{[]*Node{}, []*Node{}, 0, 0, []float64{}}
	return n
}

func (self *Node) AddNode(node *Node) {
	self.Ins = append(self.Ins, node)
	node.Outs = append(node.Outs, self)
}

func (self *Node) Show() {
	fmt.Printf("Input Node %f\n", self.Value)
	fmt.Printf("With Error %f\n", self.Err)
}

func (self *Node) FeedForward() {
}
