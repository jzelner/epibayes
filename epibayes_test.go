package epibayes

import (
	"log"
	"testing"
)

func TestSMConstructor(t *testing.T) {
	sm := NewStateMatrix(200, "S", "I", "R")
	log.Println(sm.Get("S", 10))
	sm.Set(0.25, "S", 10)
	log.Println(sm.Get("S", 10))

}
