package patch

import (
	"github.com/jzelner/epibayes"
	"github.com/jzelner/gsl-cgo/randist"
	"log"
	"testing"
	"time"
)

func SamplePatch(rng *randist.RNG) *epibayes.HybridSIR {
	T := 1500

	po := []*epibayes.PartialObserved{}
	for i := 0; i < 4; i++ {
		o := epibayes.NewPartialObserved(T, []float64{1.0, 0.0, 0.0}, []int{1 + randist.UniformRandomInt(rng, T-1)}, nil)
		po = append(po, o)
	}
	for i := 0; i < 35; i++ {
		o := epibayes.NewPartialObserved(T, []float64{1.0, 0.0, 1.0}, nil, []int{1 + randist.UniformRandomInt(rng, T-1)})
		po = append(po, o)
	}
	h := epibayes.NewHybridSIR(T, 100, po, []float64{1.0, 0.0, 0.0}, rng)
	return h
}

func SamplePars() Parameters {
	return Parameters{
		Beta:        0.6,
		Gamma:       0.2,
		SenderTau:   0.6,
		ReceiverTau: 0.2,
		DistanceTau: 2.0,
		Phi:         1.0}
}

func TestNetwork(t *testing.T) {
	//beta := 0.01
	rng := randist.NewRNG(randist.RAND48)
	rng.SetSeed(int(time.Now().UnixNano()))

	backgroundInfection := map[string]InfectionSource{
		"BORBON":     &ConstantRate{0.1},
		"COLON_ELOY": &ConstantRate{0.02}}

	n := NewSIRModel()
	n.AddPatch("BORBON", SamplePatch(rng))
	n.AddPatch("COLON_ELOY", SamplePatch(rng))
	n.AddEdge("BORBON", "COLON_ELOY", 10.0)
	log.Println(n.Initialize(SamplePars()))
	for i := 0; i < 98; i++ {
		n.Step(SamplePars(), backgroundInfection)
	}

	log.Println(n.LogProbability())

}
