#=
acausal_structures:
- Julia version: 
- Author: marcin
- Date: 2018-05-10
=#

using BayesNets.CPDs.DiscreteCPD
using BayesNets.DAG
using BayesNets.DiscreteBayesNet

include("qpds.jl")

export
#     AcausalStructure
    xD,
    DiscreteQCPD,
    HermitianMatrix,
    isprobmat,
    event

function xD(i::Int64)
    i = i + 4;
    i+1
end



