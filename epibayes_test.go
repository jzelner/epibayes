package epibayes

import (
	"github.com/jzelner/gsl-cgo/randist"
	"log"
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestSMConstructor(t *testing.T) {
	sm := NewStateMatrix(200, "S", "I", "R")
	log.Println(sm.Get("S", 10))
	sm.Set(0.25, "S", 10)
	log.Println(sm.Get("S", 10))
}

func TestPOConstructor(t *testing.T) {
	NewPartialObserved(100, []float64{1.0, 0.0, 0.0}, []int{10}, nil)
}

func TestGeometricSample(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	rng := randist.NewRNG(randist.RAND48)
	rng.SetSeed(int(time.Now().UnixNano()))
	//log.Println(GeometricSample(0.6, rng))
	s, e := SampleGeometricInfectiousPeriod(10, 0.2, rng)
	log.Println(s, e)
	log.Println(GeometricInfectiousPeriodProposal(e-s, 0.2))
}

func TestPOAugment(t *testing.T) {
	log.Println("PO AUG")
	rand.Seed(time.Now().UnixNano())
	rng := randist.NewRNG(randist.RAND48)
	rng.SetSeed(int(time.Now().UnixNano()))

	//	po := NewPartialObserved(100, []float64{1.0, 0.0, 0.0}, []int{10}, []int{50, 80})

	//	ObservationsToPriorSample(po, 0.5)

	po2 := NewPartialObserved(1500, []float64{0.5, 0.0, 0.5}, nil, []int{50, 80})

	s := time.Now()
	ObservationsToPriorSample(po2, 0.5, rng)
	for i := 0; i < 30000; i++ {
		SampleStartingState(po2, rng)

		for i := 0; i < 99; i++ {
			Step(po2, i, 0.3, 0.5, rng)
		}
	}
	e := time.Now()
	log.Println(e.Sub(s))
	//log.Println(po.Prior, po.PriorLP, po.PosteriorSample, po.PosteriorLP)
	//	log.Println(po2.Prior, po2.PriorLP, po2.PosteriorSample, po2.PosteriorLP)

}

func TestTransModel(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	rng := randist.NewRNG(randist.RAND48)
	rng.SetSeed(int(time.Now().UnixNano()))

	T := 1600
	TotalPop := 30
	gamma := 0.01
	b := 0.1
	a := 0.001
	obs := []*PartialObserved{}
	st := time.Now()
	for i := 0; i < 10; i++ {

		po := NewPartialObserved(T, []float64{1.0, 0.0, 0.0}, []int{1 + rand.Intn(T-1)}, nil)
		ObservationsToPriorSample(po, gamma, rng)
		//	log.Println(po.Prior.Values)
		obs = append(obs, po)
	}

	for i := 0; i < TotalPop-10; i++ {
		po := NewPartialObserved(T, []float64{1.0, 0.0, 1.0}, nil, []int{rand.Intn(T)})
		ObservationsToPriorSample(po, gamma, rng)
		obs = append(obs, po)
	}

	N := 1
	//init step
	infSeries := []float64{}
	logProb := 0.0

	for i := 0; i < N; i++ {
		totalInf := 0.0
		nextInf := 0.0
		for _, o := range obs {
			totalInf += SampleStartingState(o, rng)
		}
		infSeries = append(infSeries, totalInf)
		lambda := 1.0 - math.Exp(-((b*totalInf)+a)/float64(TotalPop))
		for t := 0; t < T-1; t++ {
			for _, o := range obs {
				x := Step(o, t, lambda, gamma, rng)
				nextInf += x
				if t == T-2 {
					logProb += o.PosteriorLP
				}
			}

			infSeries = append(infSeries, nextInf)
			lambda = 1.0 - math.Exp(-((b*nextInf)+a)/float64(TotalPop))
			nextInf = 0.0

		}
	}
	en := time.Now()
	log.Println(en.Sub(st))
	log.Println(infSeries, logProb)
}
