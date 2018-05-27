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
        result = a[dmcpd]
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
        result = ptrace(distribution, all_ncategories, trace_out_indices)

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

    HermitianMatrix(result)
end


struct HermitianMatrix{T<:Union{Complex{Float64}, Float64}} <: DiscreteMatrixDistribution
    K::Int
    # KxK matrix
    p::Matrix{T}

    # this constructor replaces the default constructor
    # this function must be called with explicit T specialization
    # so HermitianMatrix{Complex{Int64}}(3, y) will work
    # but HermitianMatrix(3, y) won't
    function HermitianMatrix{T}(K::Int, p::Matrix{T}) where T
        x, y = size(p)
        @check_args(HermitianMatrix, K==x==y)
#         @check_args(HermitianMatrix, is_probmat(p))
        new(K, p)
    end
end


function event(system::HermitianMatrix, e::Matrix)
    HermitianMatrix((e * system.p * e) / trace(e * system.p))
end

HermitianMatrix{T<:Union{Complex{Float64}, Float64}}(p::Matrix{T}) = HermitianMatrix{T}(length(diag(p)),p)
HermitianMatrix{T<:Union{Complex{Float64}, Float64}}(v::Vector{T}) = HermitianMatrix{T}(length(v), diagm(v))
# TODO
# construtor from integer (K => diagonal of 1/k)

function is_probmat(m::Matrix)
    isprobvec(real(diag(m)))
end

const DiscreteQCPD = DiscreteMCPD{HermitianMatrix}
