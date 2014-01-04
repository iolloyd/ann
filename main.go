package main

import (
	"./nn"
)

var (
	inLayer     = 4
	hiddenLayer = 4
	outLayer    = 2
)

func main() {
	td := nn.TrainingData{}
	td.Inputs = [][]float64{
		{0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 1.0},
		{0.0, 0.0, 1.0, 0.0}}

	td.Outputs = [][]float64{
		{0.0, 0.0},
		{1.0, 0.0},
		{0.0, 1.0}}
	nn := nn.MakeNeuralNetwork(inLayer, hiddenLayer, outLayer)
	nn.Train(td)
	nn.Show()
}
