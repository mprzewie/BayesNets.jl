#=
hermitian_matrix_distribution:
- Julia version: 
- Author: marcin
- Date: 2018-06-02
=#

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

Base.convert(::Type{HermitianMatrix}, m::Matrix) = HermitianMatrix(m)
Base.convert(::Type{Matrix}, hm::HermitianMatrix) = hm.p