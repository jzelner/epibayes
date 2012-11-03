package epibayes

import (
	"fmt"
	"strings"
)

func ZeroVector(length int) []float64 {
	x := make([]float64, length)
	for i := 0; i < length; i++ {
		x[i] = 0.0
	}
	return x
}

type StateMatrix struct {
	Values        [][]float64
	orderedStates []string
	States        map[string]int
}

func (sm *StateMatrix) Set(x float64, state string, index int) {
	sm.Values[sm.States[state]][index] = x
}

func (sm *StateMatrix) Get(state string, index int) float64 {
	return sm.Values[sm.States[state]][index]
}

func (sm *StateMatrix) String() string {
	var rows string
	for i, s := range sm.orderedStates {
		row := make([]string, 0)
		row = append(row, s+":")
		for _, v := range sm.Values[i] {
			row = append(row, fmt.Sprintf("%0.2g", v))
		}
		rows += strings.Join(row, ",") + "\n"
	}
	return rows
}

func NewStateMatrix(length int, states ...string) *StateMatrix {
	sm := new(StateMatrix)
	sm.States = make(map[string]int)
	sm.orderedStates = states
	for i, s := range states {
		sm.States[s] = i
	}
	sm.Values = make([][]float64, len(sm.States))
	for i := 0; i < len(sm.States); i++ {
		sm.Values[i] = ZeroVector(length)
	}
	return sm
}

type StateProbability struct {
	State       string
	Probability float64
	Time        int
}

type PartialObserved struct {
	Observed        []StateProbability
	Prior           StateMatrix
	PosteriorSample StateMatrix
}
