#=
acausal_structures:
- Julia version: 
- Author: marcin
- Date: 2018-05-10
=#

using BayesNets.CPDs.DiscreteCPD
using BayesNets.DAG

export
    AcausalStructure


function xD(i::Int64)
    i = i + 4;
    i+1
end


mutable struct AcausalStructure{T<:DiscreteCPD} <: ProbabilisticGraphicalModel
	dag::DAG # nodes are in topological order

	static_cpds::Vector{T} # CPDs associated with nodes that are not entangled
	name_to_index::Dict{NodeName,Int} # NodeName → index in dag and cpds
end

AcausalStructure() = AcausalStructure(DAG(0), DiscreteCPD[], Dict{NodeName, Int}())

infer(as::AcausalStructure) = 1