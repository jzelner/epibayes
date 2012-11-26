package patch

import (
	"encoding/json"
	"fmt"
	"math"
)

type Parameters struct {
	Beta        float64
	Gamma       float64
	SenderTau   float64
	ReceiverTau float64
	DistanceTau float64
	Phi         float64
}

type ConstantRate struct {
	Alpha float64
}

func (bi *ConstantRate) At(t int) float64 {
	return bi.Alpha
}

type DecayingInfectionSource struct {
	//This is the overall rate of transmission,
	//i.e. when there is no distance from the source
	alpha float64

	//This is the decay rate per unit of distance
	//or remoteness
	phi float64

	distance float64
	popSize  float64

	//Because this is constant, we'll cache the value when 
	//it's set instead of calling over and over
	cachedRate float64
}

func (d *DecayingInfectionSource) PopSize() float64 { return d.popSize }
func (d *DecayingInfectionSource) SetPopSize(px float64) {
	d.popSize = px
	d.calculateRate()
}

func (d *DecayingInfectionSource) Distance() float64 { return d.distance }
func (d *DecayingInfectionSource) SetDistance(dx float64) {
	d.distance = dx
	d.calculateRate()
}

func (d *DecayingInfectionSource) Alpha() float64 { return d.alpha }
func (d *DecayingInfectionSource) SetAlpha(ax float64) {
	d.alpha = ax
	d.calculateRate()
}

func (d *DecayingInfectionSource) Phi() float64 { return d.phi }
func (d *DecayingInfectionSource) SetPhi(px float64) {
	d.phi = px
	d.calculateRate()
}

func (d *DecayingInfectionSource) calculateRate() {
	d.cachedRate = d.alpha * d.popSize * math.Exp(-d.phi*d.distance)
}
func (d *DecayingInfectionSource) At(x int) float64 {
	return d.cachedRate
}

func (d *DecayingInfectionSource) GobEncode() ([]byte, error) {
	vals := map[string]float64{"alpha": d.alpha, "popSize": d.popSize, "distance": d.distance, "phi": d.phi}
	b, err := json.Marshal(vals)
	if err != nil {
		return nil, err
	}

	return b, nil
}

func (d *DecayingInfectionSource) GobDecode(b []byte) error {
	vals := map[string]float64{}
	err := json.Unmarshal(b, &vals)
	if err != nil {
		return err
	}

	d.alpha = vals["alpha"]
	d.phi = vals["phi"]
	d.popSize = vals["popSize"]
	d.distance = vals["distance"]

	d.calculateRate()
	return nil
}
func NewDecayingInfectionSource(alpha, phi, distance, popSize float64) *DecayingInfectionSource {
	d := &DecayingInfectionSource{alpha, phi, distance, popSize, 0.0}
	d.calculateRate()
	return d
}

type InfectionSource interface {
	At(int) float64
}

//SIRModel has an embedded network
//in it, as well as a counter, T, 
//for the current time
type SIRModel struct {
	*Network
	T int
}

func NewSIRModel() *SIRModel {
	m := new(SIRModel)
	m.Network = NewNetwork()
	m.T = 0
	return m
}

func (m *SIRModel) SetGravityWeights(p Parameters) {
	SetGravityWeights(m.Network, p.SenderTau, p.ReceiverTau, p.DistanceTau, p.Phi)
}

func (m *SIRModel) Initialize(p Parameters) float64 {
	totalI := 0.0
	m.T = 0
	m.SetGravityWeights(p)
	for _, n := range m.Nodes {
		totalI += n.P.Initialize(p.Gamma)
	}
	return totalI
}

func (m *SIRModel) Step(p Parameters, externalInf map[string]InfectionSource) float64 {
	totalI := 0.0
	//Loop through the nodes 
	//in the spatial network
	for k, n := range m.Nodes {
		var ei float64
		s, ok := externalInf[k]
		if ok {
			ei = s.At(m.T)
		} else {
			ei = 0.0
		}

		//Get total exposure to neighbors
		neighbors := m.IncidenceList[k]
		for _, ne := range neighbors.IncomingEdges {
			ei += ne.Weight * ne.From.P.CurrentI()
		}
		totalI += n.P.Step(m.T, p.Beta, ei, p.Gamma)
	}
	m.T++
	return totalI
}

func (m *SIRModel) LogProbability() float64 {
	logLL := 0.0
	for _, n := range m.Nodes {
		logLL += n.P.LogProbability()
	}
	return logLL
}

type Patch interface {
	Initialize(float64) float64
	Step(int, float64, float64, float64) float64
	Size() float64
	CurrentI() float64
	LogProbability() float64
}

type PatchNode struct {
	ID       string
	P        Patch
	Metadata interface{}
}

type IncidenceEntry struct {
	Node          *PatchNode
	IncomingEdges []*SpatialEdge
	OutgoingEdges []*SpatialEdge
}

func (ie *IncidenceEntry) AddIncomingEdge(e *SpatialEdge) {
	ie.IncomingEdges = append(ie.IncomingEdges, e)
}

func (ie *IncidenceEntry) AddOutgoingEdge(e *SpatialEdge) {
	ie.OutgoingEdges = append(ie.OutgoingEdges, e)
}

type SpatialEdge struct {
	Weight   float64
	Distance float64
	From     *PatchNode
	To       *PatchNode
	Metadata interface{}
}

func (se *SpatialEdge) String() string {
	return fmt.Sprintf("%s -> %s; Dist : %0.2f, Weight :%0.2g", se.From.ID, se.To.ID, se.Distance, se.Weight)
}

//Network implements a simple spatial network
//where interactions between nodes are governed
//by the weights of spatial edges. The weights
//indicate the impact of prevalence in the
//sending community on risk in the receiving
//community.
type Network struct {
	Edges         []*SpatialEdge
	Nodes         map[string]*PatchNode
	IncidenceList map[string]*IncidenceEntry
}

func (n *Network) AddPatch(id string, p Patch) *PatchNode {
	pn := new(PatchNode)
	pn.ID = id
	pn.P = p
	n.Nodes[id] = pn
	n.IncidenceList[id] = &IncidenceEntry{pn, make([]*SpatialEdge, 0), make([]*SpatialEdge, 0)}
	return pn
}

func (n *Network) AddEdge(from, to string, distance float64) *SpatialEdge {
	if from == to {
		return nil
	}
	e := new(SpatialEdge)
	e.Weight = 1.0
	e.Distance = distance
	e.From = n.Nodes[from]
	e.To = n.Nodes[to]
	n.Edges = append(n.Edges, e)
	n.IncidenceList[from].AddOutgoingEdge(e)
	n.IncidenceList[to].AddIncomingEdge(e)
	return e
}

func NewNetwork() *Network {
	n := new(Network)
	n.Edges = make([]*SpatialEdge, 0)
	n.Nodes = make(map[string]*PatchNode, 0)
	n.IncidenceList = make(map[string]*IncidenceEntry, 0)
	return n
}

func Gravity(st, rt, dt, phi float64) func(float64, float64, float64) float64 {
	return func(senderPop, receiverPop, distance float64) float64 {
		return phi * math.Pow(senderPop, st) * math.Pow(receiverPop, rt) / math.Pow(distance, dt)
	}
}

func SetGravityWeights(n *Network, senderTau, receiverTau, distanceTau, phi float64) {
	gFunc := Gravity(senderTau, receiverTau, distanceTau, phi)
	for _, e := range n.Edges {
		e.Weight = gFunc(e.From.P.Size(), e.To.P.Size(), e.Distance)
	}
}
