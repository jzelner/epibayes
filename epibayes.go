package epibayes

import (
	"fmt"
	//	"log"
	"math"
	"math/rand"
	"strings"
)

const (
	S = iota
	I
	R
)

func ZeroVector(length int) []float64 {
	x := make([]float64, length)
	for i := 0; i < length; i++ {
		x[i] = 0.0
	}
	return x
}

type StateMatrix struct {
	length        int
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

func (sm *StateMatrix) Len() int {
	return sm.length
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
	sm.length = length
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

type TimeSlice struct {
	Time          int
	Probabilities []float64
}

type PartialObserved struct {
	StartingProbabilities    []float64
	InfectionObservations    []int
	NonInfectionObservations []int
	Prior                    StateMatrix
	PriorLP                  float64
	PosteriorSample          StateMatrix
	PosteriorLP              float64
}

func NewPartialObserved(length int, startingProbs []float64, infObs []int, nonInfObs []int) *PartialObserved {
	po := new(PartialObserved)
	po.StartingProbabilities = startingProbs
	po.InfectionObservations = infObs
	po.NonInfectionObservations = nonInfObs
	po.Prior = *NewStateMatrix(length, "S", "I", "R")
	po.PosteriorSample = *NewStateMatrix(length, "S", "I", "R")
	return po
}

func GeometricSample(l float64) int {
	return int(math.Ceil(rand.ExpFloat64() / l))
}

func GeometricInfectiousPeriodProposal(duration int, gamma float64) float64 {
	rSampleProb := 1.0 / float64(duration)
	geomProb := (1.0 - math.Exp(-gamma*float64(duration))) - (1.0 - math.Exp(-gamma*float64(duration-1)))
	return rSampleProb * geomProb
}

func SampleGeometricInfectiousPeriod(obsTime int, gamma float64) (int, int) {
	//Sample the duration of the infectious period
	duration := GeometricSample(gamma)

	//Now sample the offset for the observation, which could fall anywhere
	//from [0, duration-1]

	//If offset = 0, then start = 0, if offset = 3, then start = obsTime -3
	start := obsTime - rand.Intn(duration)
	end := start + duration

	if start < 0 {
		start = 0
	}
	return start, end
}

func ObservationsToPriorSample(po *PartialObserved, gamma float64) {
	po.PriorLP = 0.0

	nonObsTimes := make(map[int]bool)
	for _, nt := range po.NonInfectionObservations {
		nonObsTimes[nt] = true
	}
	for t := 0; t < po.Prior.Len(); t++ {
		_, ok := nonObsTimes[t]
		if ok {
			po.Prior.Values[S][t] = 1.0
			po.Prior.Values[I][t] = 0.0
			po.Prior.Values[R][t] = 1.0
		} else {
			po.Prior.Values[S][t] = 1.0
			po.Prior.Values[I][t] = 1.0
			po.Prior.Values[R][t] = 1.0
		}
	}

	for _, io := range po.InfectionObservations {
		s, e := SampleGeometricInfectiousPeriod(io, gamma)
		if e >= po.Prior.Len() {
			e = po.Prior.Len() - 1
		}
		for t := s; t <= e; t++ {
			po.Prior.Values[S][t] = 0.0
			po.Prior.Values[I][t] = 1.0
			po.Prior.Values[R][t] = 0.0
		}

		for t := 0; t < s; t++ {
			po.Prior.Values[S][t] = 1.0
			po.Prior.Values[I][t] = 0.0
			po.Prior.Values[R][t] = 0.0
		}

		for t := e + 1; t < po.Prior.Len()-1; t++ {
			po.Prior.Values[S][t] = 0.0
			po.Prior.Values[I][t] = 0.0
			po.Prior.Values[R][t] = 1.0
		}

		po.PriorLP += math.Log(1.0 / float64(e-s))
	}
}

func SampleStartingState(po *PartialObserved) {
	po.PosteriorLP = 0.0
	x := rand.Float64()
	total := 0.0
	var st int
	var pr float64
	for i := 0; i < 3; i++ {
		pr = po.StartingProbabilities[i]
		if pr == 0.0 {
			continue
		}
		total += po.StartingProbabilities[i]
		if x <= total {
			st = i
			break
		}
	}

	po.PosteriorSample.Values[st][0] = 1.0
	po.PosteriorLP += po.PriorLP + math.Log(pr)
}

func Step(po *PartialObserved, t int, infPr, recPr float64) {
	//Given the transition probabilities and priors, sample the 
	//next step and calculate the transition probability

	if po.PosteriorSample.Values[S][t] == 1 {
		StoI := po.PosteriorSample.Values[S][t] * po.Prior.Values[I][t+1] * infPr
		if StoI > 0.0 && rand.Float64() < StoI {
			po.PosteriorSample.Values[S][t+1] = 0.0
			po.PosteriorSample.Values[I][t+1] = 1.0
			po.PosteriorSample.Values[R][t+1] = 0.0
			po.PosteriorLP += math.Log(StoI)

		} else {
			po.PosteriorSample.Values[S][t+1] = 1.0
			po.PosteriorSample.Values[I][t+1] = 0.0
			po.PosteriorSample.Values[R][t+1] = 0.0
			po.PosteriorLP += math.Log(1.0 - StoI)

		}
	} else if po.PosteriorSample.Values[I][t] == 1 {
		ItoR := po.PosteriorSample.Values[I][t] * po.Prior.Values[R][t+1] * recPr
		if ItoR > 0.0 && rand.Float64() < ItoR {
			po.PosteriorSample.Values[S][t+1] = 0.0
			po.PosteriorSample.Values[I][t+1] = 0.0
			po.PosteriorSample.Values[R][t+1] = 1.0
			po.PosteriorLP += math.Log(ItoR)
		} else {
			po.PosteriorSample.Values[S][t+1] = 0.0
			po.PosteriorSample.Values[I][t+1] = 1.0
			po.PosteriorSample.Values[R][t+1] = 0.0
			po.PosteriorLP += math.Log(1.0 - ItoR)
		}
	} else if po.PosteriorSample.Values[R][t] == 1 {
		po.PosteriorSample.Values[S][t+1] = 0.0
		po.PosteriorSample.Values[I][t+1] = 0.0
		po.PosteriorSample.Values[R][t+1] = 1.0
	}

}
