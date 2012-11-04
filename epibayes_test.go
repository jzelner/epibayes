package epibayes

import (
	"log"
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
	log.Println(GeometricSample(0.6))
	s, e := SampleGeometricInfectiousPeriod(10, 0.2)
	log.Println(s, e)
	log.Println(GeometricInfectiousPeriodProposal(e-s, 0.2))
}

func TestPOAugment(t *testing.T) {
	log.Println("PO AUG")
	rand.Seed(time.Now().UnixNano())

	//	po := NewPartialObserved(100, []float64{1.0, 0.0, 0.0}, []int{10}, []int{50, 80})

	//	ObservationsToPriorSample(po, 0.5)

	po2 := NewPartialObserved(1500, []float64{0.5, 0.0, 0.5}, nil, []int{50, 80})

	s := time.Now()
	ObservationsToPriorSample(po2, 0.5)
	for i := 0; i < 3000; i++ {
		SampleStartingState(po2)

		for i := 0; i < 99; i++ {
			Step(po2, i, 0.3, 0.5)
		}
	}
	e := time.Now()
	log.Println(e.Sub(s))
	//log.Println(po.Prior, po.PriorLP, po.PosteriorSample, po.PosteriorLP)
	log.Println(po2.Prior, po2.PriorLP, po2.PosteriorSample, po2.PosteriorLP)

}
