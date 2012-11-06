package epibayes

import (
	"fmt"
	"github.com/jzelner/gsl-cgo/randist"
	tu "github.com/jzelner/timeutil"
	"math"
	"strings"
	"time"
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

type IndividualHistory struct {
	ID               string
	PositiveObsTimes []time.Time
	NegativeObsTimes []time.Time
}

func IntegerDiff(t1, t2 time.Time, unit tu.Unit) int {
	duration := int(tu.Diff(tu.FromDate(t1, unit), tu.FromDate(t2, unit)))
	return duration
}

func RandomIndividualHistory(id string, inf bool, start, end time.Time, unit tu.Unit, rng *randist.RNG) IndividualHistory {
	ih := new(IndividualHistory)
	ih.ID = id

	ih.PositiveObsTimes = make([]time.Time, 0)
	ih.NegativeObsTimes = make([]time.Time, 0)

	max := IntegerDiff(start, end, unit)

	t := randist.UniformRandomInt(rng, max)
	if inf {
		ih.PositiveObsTimes = append(ih.PositiveObsTimes, tu.ToDate(tu.Add(tu.FromDate(start, unit), int64(t))))
	} else {
		ih.NegativeObsTimes = append(ih.NegativeObsTimes, tu.ToDate(tu.Add(tu.FromDate(start, unit), int64(t))))

	}
	return *ih
}

func IntegerOffsets(startTime time.Time, times []time.Time, unit tu.Unit) []int {
	st := tu.FromDate(startTime, unit)
	rInts := make([]int, len(times))
	for i, t := range times {
		rInts[i] = int(tu.Diff(st, tu.FromDate(t, unit)))
	}
	return rInts
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

func GeometricSample(l float64, rd *randist.RNG) int {
	return int(math.Ceil(randist.ExponentialRandomFloat64(rd, 1.0/l)))
}

func GeometricInfectiousPeriodProposal(duration int, gamma float64) float64 {
	rSampleProb := 1.0 / float64(duration)
	geomProb := (1.0 - math.Exp(-gamma*float64(duration))) - (1.0 - math.Exp(-gamma*float64(duration-1)))
	return rSampleProb * geomProb
}

func SampleGeometricInfectiousPeriod(obsTime int, gamma float64, rd *randist.RNG) (int, int) {
	//Sample the duration of the infectious period
	duration := GeometricSample(gamma, rd)
	//Now sample the offset for the observation, which could fall anywhere
	//from [0, duration-1]

	//If offset = 0, then start = 0, if offset = 3, then start = obsTime -3
	start := obsTime - randist.UniformRandomInt(rd, duration)
	end := start + duration

	if start < 0 {
		start = 0
	}
	return start, end
}

func ObservationsToPriorSample(po *PartialObserved, gamma float64, rng *randist.RNG) {
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
		s, e := SampleGeometricInfectiousPeriod(io, gamma, rng)
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

func SampleStartingState(po *PartialObserved, rd *randist.RNG) float64 {
	po.PosteriorLP = 0.0
	x := randist.UniformRandomFloat64(rd)
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

	if st == I {
		return 1.0
	}
	return 0.0
}

func Step(po *PartialObserved, t int, StoI, ItoR float64, rng *randist.RNG) float64 {
	//Given the transition probabilities and priors, sample the 
	//next step and calculate the transition probability
	if po.PosteriorSample.Values[S][t] == 1 {
		ConditionalStoI := po.Prior.Values[I][t+1] * StoI
		ConditionalStoS := po.Prior.Values[S][t+1] * (1.0 - StoI)
		total := ConditionalStoI + ConditionalStoS
		ConditionalStoI = ConditionalStoI / total
		ConditionalStoS = ConditionalStoS / total
		if StoI > 0.0 && randist.UniformRandomFloat64(rng) < ConditionalStoI {
			po.PosteriorSample.Values[S][t+1] = 0.0
			po.PosteriorSample.Values[I][t+1] = 1.0
			po.PosteriorSample.Values[R][t+1] = 0.0
			po.PosteriorLP += math.Log(StoI)
			return 1.0
		} else {
			po.PosteriorSample.Values[S][t+1] = 1.0
			po.PosteriorSample.Values[I][t+1] = 0.0
			po.PosteriorSample.Values[R][t+1] = 0.0
			po.PosteriorLP += math.Log(1.0 - StoI)

		}
	} else if po.PosteriorSample.Values[I][t] == 1 {
		ConditionalItoR := po.Prior.Values[R][t+1] * ItoR
		ConditionalItoI := po.Prior.Values[I][t+1] * (1.0 - ItoR)
		total := ConditionalItoR + ConditionalItoI
		ConditionalItoR /= total
		ConditionalItoI /= total
		if ItoR > 0.0 && randist.UniformRandomFloat64(rng) < ConditionalItoR {
			po.PosteriorSample.Values[S][t+1] = 0.0
			po.PosteriorSample.Values[I][t+1] = 0.0
			po.PosteriorSample.Values[R][t+1] = 1.0
			po.PosteriorLP += math.Log(ItoR)
		} else {
			po.PosteriorSample.Values[S][t+1] = 0.0
			po.PosteriorSample.Values[I][t+1] = 1.0
			po.PosteriorSample.Values[R][t+1] = 0.0
			po.PosteriorLP += math.Log(1.0 - ItoR)
			return 1.0
		}
	} else if po.PosteriorSample.Values[R][t] == 1 {
		po.PosteriorSample.Values[S][t+1] = 0.0
		po.PosteriorSample.Values[I][t+1] = 0.0
		po.PosteriorSample.Values[R][t+1] = 1.0
	}
	return 0.0
}

type UnobservedMassAction struct {
	States                *StateMatrix
	StartingProbabilities []float64
	totalN                int
	N                     int
	SamplingLP            float64
}

func NewUnobservedMassAction(length, N, unobsN int, initProbs []float64) *UnobservedMassAction {
	uo := new(UnobservedMassAction)
	uo.totalN = N
	uo.N = unobsN
	uo.States = NewStateMatrix(length, "S", "I", "R")
	uo.StartingProbabilities = initProbs
	uo.SamplingLP = 0.0
	return uo
}

func SIRInitialize(uo *UnobservedMassAction, rd *randist.RNG) float64 {
	uo.States.Values[S][0] = 0.0
	uo.States.Values[I][0] = 0.0
	uo.States.Values[R][0] = 0.0
	uo.SamplingLP = 0.0

	for i := 0; i < uo.N; i++ {
		total := 0.0
		x := randist.UniformRandomFloat64(rd)
		for j := S; j < R; j++ {
			if uo.StartingProbabilities[j] == 0.0 {
				continue
			}
			total += uo.StartingProbabilities[j]
			if total >= x {
				uo.States.Values[j][0] += 1
				uo.SamplingLP += math.Log(uo.StartingProbabilities[j])
				break
			}
		}
	}
	return uo.States.Values[I][0]
}

func SIRStep(uo *UnobservedMassAction, t int, StoI, ItoR float64, rng *randist.RNG) float64 {
	sm := uo.States
	//Calculate rates at T
	nItoR := 0.0
	nStoI := 0.0

	if sm.Values[S][t] > 0.0 {
		if StoI > 0.0 {
			nStoI = float64(randist.BinomialRandomInt(rng, StoI, int(sm.Values[S][t])))
			uo.SamplingLP += math.Log(randist.BinomialPMF(int(nStoI), StoI, int(sm.Values[S][t])))
		}
	}

	if sm.Values[I][t] > 0.0 && ItoR > 0.0 {
		nItoR = float64(randist.BinomialRandomInt(rng, ItoR, int(sm.Values[I][t])))
		uo.SamplingLP += math.Log(randist.BinomialPMF(int(nItoR), ItoR, int(sm.Values[I][t])))
	}

	sm.Values[S][t+1] = sm.Values[S][t] - nStoI
	sm.Values[I][t+1] = sm.Values[I][t] + nStoI - nItoR
	sm.Values[R][t+1] = sm.Values[R][t] + nItoR

	return sm.Values[I][t+1]
}

type HybridSIR struct {
	N                   int
	PartialObservations []*PartialObserved
	Unobserved          *UnobservedMassAction
	rd                  *randist.RNG
	lastStepI           float64
}

func (h *HybridSIR) CurrentI() float64 {
	return h.lastStepI
}

func (h *HybridSIR) Size() float64 {
	return float64(h.N)
}

type PatchHistory struct {
	StartObs                  time.Time
	EndObs                    time.Time
	ID                        string
	N                         int
	InitialStateProbabilities []float64
	Observations              []IndividualHistory
}

func NewPatchHistory(ID string, N int, s, e time.Time, sp []float64) *PatchHistory {
	ph := new(PatchHistory)
	ph.ID = ID
	ph.N = N
	ph.StartObs = s
	ph.EndObs = e
	ph.InitialStateProbabilities = sp
	ph.Observations = make([]IndividualHistory, 0)
	return ph
}

func HybridSIRFromPatchHistory(ph PatchHistory, rng *randist.RNG) *HybridSIR {
	duration := int(tu.Diff(tu.FromDate(ph.StartObs, tu.DAY), tu.FromDate(ph.EndObs, tu.DAY)))
	//First make the individual observations
	indObs := make([]*PartialObserved, 0)
	for _, o := range ph.Observations {
		//If there are infection observations, these dominate and we ignore
		//the non-infection observations
		if len(o.PositiveObsTimes) > 0 {
			indObs = append(indObs, NewPartialObserved(duration, []float64{1.0, 0.0, 0.0}, IntegerOffsets(ph.StartObs, o.PositiveObsTimes, tu.DAY), nil))
		} else if len(o.NegativeObsTimes) > 0 {
			indObs = append(indObs, NewPartialObserved(duration, []float64{1.0, 0.0, 1.0}, nil, IntegerOffsets(ph.StartObs, o.NegativeObsTimes, tu.DAY)))
		}
	}
	//Now make the SIR model 
	hs := NewHybridSIR(duration, ph.N, indObs, ph.InitialStateProbabilities, rng)
	return hs
}

func NewHybridSIR(length, N int, po []*PartialObserved, initProbs []float64, rng *randist.RNG) *HybridSIR {
	hs := new(HybridSIR)
	hs.N = N
	hs.PartialObservations = po
	hs.rd = rng

	hs.Unobserved = NewUnobservedMassAction(length, hs.N, hs.N-len(po), initProbs)
	return hs
}

func (h *HybridSIR) AddPartialObserved(po *PartialObserved) {
	h.PartialObservations = append(h.PartialObservations, po)
}

func (h *HybridSIR) Initialize(gamma float64) float64 {
	h.lastStepI = 0.0
	for _, po := range h.PartialObservations {
		ObservationsToPriorSample(po, gamma, h.rd)
		h.lastStepI += SampleStartingState(po, h.rd)
	}

	h.lastStepI += SIRInitialize(h.Unobserved, h.rd)
	return h.lastStepI
}

func (h *HybridSIR) Step(t int, beta, externalInf, gamma float64) float64 {
	StoI := 1.0 - math.Exp((-beta*(h.lastStepI+externalInf))/float64(h.N))
	ItoR := 1.0 - math.Exp(-gamma)
	h.lastStepI = 0.0
	for _, po := range h.PartialObservations {
		x := Step(po, t, StoI, ItoR, h.rd)
		h.lastStepI += x
	}
	h.lastStepI += SIRStep(h.Unobserved, t, StoI, ItoR, h.rd)
	return h.lastStepI
}

func (h *HybridSIR) LogProbability() float64 {
	logLL := 0.0
	for _, po := range h.PartialObservations {
		logLL += po.PosteriorLP
	}
	logLL += h.Unobserved.SamplingLP
	return logLL
}
