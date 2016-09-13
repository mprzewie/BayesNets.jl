"""
A linear Gaussian CPD, always returns a Normal{Float64}

	Assumes that target and all parents can be converted to Float64 (ie, are numeric)

	P(x|parents(x)) = Normal(μ=a×parents(x) + b, σ)
"""
type LinearGaussianCPD <: CPD{Normal{Float64}}
    target::NodeName
    parents::Vector{NodeName}

	a::Vector{Float64}
	b::Float64
    σ::Float64
end
LinearGaussianCPD(target::NodeName, μ::Float64, σ::Float64) = LinearGaussianCPD(target, NodeName[], Float64[], μ, σ)

name(cpd::LinearGaussianCPD) = cpd.target
parents(cpd::LinearGaussianCPD) = cpd.parents
nparams(cpd::LinearGaussianCPD) = length(cpd.a) + 2

function Base.call(cpd::LinearGaussianCPD, a::Assignment)
    # compute A⋅v + b
    μ = cpd.b
    for (i, p) in enumerate(cpd.parents)
        μ += a[p]*cpd.a[i]
    end

    Normal(μ, cpd.σ)
end

function Distributions.fit(::Type{LinearGaussianCPD},
    data::DataFrame,
    target::NodeName,
    prior::NormalInverseGamma = NormalInverseGamma(0.0, 1.0, 1.0, 1.0), # TODO: check whether this is reasonable
    )

    # no parents
    arr = data[target]
    eltype(arr) <: Real || error("fit LinearGaussianCPD requrires target to be numeric")

    μ, σ² = posterior_mode(prior, Normal, arr) # MLE
    σ = sqrt(σ²)

    LinearGaussianCPD(target, NodeName[], Float64[], μ, σ)
end
function Distributions.fit(::Type{LinearGaussianCPD},
    data::DataFrame,
    target::NodeName,
    parents::Vector{NodeName};
    prior::MvNormalInverseGamma = MvNormalInverseGamma(zeros(length(parents)+1), eye(length(parents)+1), 1.0, 1.0), # TODO: check whether this is reasonable
    )

    if isempty(parents)
        return fit(LinearGaussianCPD, data, target, prior=NormalInverseGamma(prior.μ[1], sqrt(1.0/prior.Λ[1]), prior.a, prior.b))
    end

    # ---------------------
    # pull parental dataset
    # 1st row is all of the data for the 1st parent
    # 2nd row is all of the data for the 2nd parent, etc.

    nparents = length(parents)
    X = Array(Float64, nrow(data), nparents+1)
    for (i,p) in enumerate(parents)
        arr = data[p]
    	for j in 1 : nrow(data)
            X[j,i] = convert(Float64, arr[j])
    	end
        X[j,end] = 1.0
    end

    y = convert(Vector{Float64}, data[target])

    # --------------------
    # solve the regression problem for β

    β, σ² = posterior_mode(prior, MvNormalInverseGamma, y, X)
    σ = sqrt(σ²)

    a = β[1:end-1]
    b = β[end]

    LinearGaussianCPD(target, parents, a, b, σ)
end