#=
quantum_bayesian:
- Julia version: 
- Author: marcin
- Date: 2018-06-02
=#

const DiscreteQCPD = DiscreteMCPD{HermitianMatrix}

const AcausalStructure = BayesNet{DiscreteQCPD}

AcausalStructure() = BayesNet(DiscreteQCPD)

