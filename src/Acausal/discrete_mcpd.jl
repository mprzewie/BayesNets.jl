#=
qpds:
- Julia version: 
- Author: marcin
- Date: 2018-05-25
=#

using QI

star(A, B) = sqrtm(B)*A*sqrtm(B) # operator z (#2)


abstract type MCPD{D <: MatrixDistribution} <: CPD{D} end

struct DiscreteMCPD{D <: DiscreteMatrixDistribution} <: MCPD{D}
    # name of the node
    # TODO in the future, this should also be NodeNames,
    # as this class will ultimately represent a system of entangled variables
    # such system is now a single variable
    target::NodeName

    parents::NodeNames
    parental_ncategories::Vector{Int}

    # in the future, a vector of Ints for the same reason as in target
    ncategories::Int

    # conditional distribution on parents
    conditional_distribution::D

    #TODO inner constructor with argument checking
end

function (dmcpd::DiscreteMCPD)(a::Assignment=Assignment())
    if haskey(a, dmcpd.target)
        return a[dmcpd]
    else
        if isempty(dmcpd.parents)
            distribution = dmcpd.conditional_distribution.p
        else
            parents_assignments = [a[p] for p in dmcpd.parents]
            total_distribution = foldl(kron, 1, vcat(parents_assignments, [eye(dmcpd.ncategories)]))
            distribution = star(dmcpd.conditional_distribution.p, total_distribution)
        end

        all_ncategories = vcat(dmcpd.parental_ncategories, dmcpd.ncategories)
        trace_out_indices = [p for p in 1:length(dmcpd.parents)]
        return ptrace(distribution, all_ncategories, trace_out_indices)

        # when the target is a single variable, this is enough
        # in the future, this function will have to support cases when we want a distribution of some part of athe system
        # (of only some variables in the system)
        # i imagine it will return a list of HermitianMatrices then
        # especially an empty list, when none of the quried nodes are *in* the given system

        # also, in this case, should any parts of the system be in the assignment,
        # event() function must be applied on the distribution of the system before
        # the result has been traced out from it
    end
end

name(dmcpd::DiscreteMCPD) = dmcpd.target
parents(dmcpd::DiscreteMCPD) = dmcpd.parents

# TODO
# nparams(dmcpd::DiscreteMCPD)

