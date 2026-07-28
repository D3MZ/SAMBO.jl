function _projected_gradient!(destination, gradient, point, lower, upper)
    tolerance = sqrt(eps(eltype(point)))
    copyto!(destination, gradient)
    for index in eachindex(destination, point, lower, upper)
        at_lower = point[index] <= lower[index] + tolerance
        at_upper = point[index] >= upper[index] - tolerance
        (at_lower && destination[index] > 0) && (destination[index] = 0)
        (at_upper && destination[index] < 0) && (destination[index] = 0)
    end
    return destination
end

"""
Internal projected, bounded limited-memory BFGS kernel.

`evaluate(point, gradient_required)` returns `(value, gradient_or_nothing, cost)`.
The callback owns objective accounting and may signal termination through `stop`.
"""
function _bounded_bfgs(
    evaluate,
    initial,
    lower,
    upper;
    initial_value=nothing,
    max_iterations=40,
    max_cost=typemax(Int),
    gradient_cost=1,
    gradient_tolerance=1e-5,
    minimum_step=2.0^-16,
    value_tolerance=0.0,
    retry_identity=false,
    reset_on_bad_curvature=false,
    memory=10,
    displacement_tolerance=eps(eltype(initial)),
    stop=Returns(false),
)
    memory > 0 || throw(ArgumentError("L-BFGS memory must be positive"))
    T = eltype(initial)
    point = clamp.(copy(initial), lower, upper)
    value = initial_value
    cost = 0
    if isnothing(value)
        value, gradient, incurred = evaluate(point, true)
        cost += incurred
    else
        cost + gradient_cost <= max_cost || return point, value, cost
        _, gradient, incurred = evaluate(point, true)
        cost += incurred
    end
    isnothing(gradient) && return point, value, cost

    projected = similar(gradient)
    direction = similar(gradient)
    candidate = similar(point)
    parameter_step = similar(point)
    gradient_step = similar(gradient)
    steps = Vector{Vector{T}}()
    gradient_steps = Vector{Vector{T}}()
    inverse_curvatures = T[]

    for _ in 1:max_iterations
        stop() && break
        all(isfinite, gradient) || break
        _projected_gradient!(projected, gradient, point, lower, upper)
        norm(projected, Inf) <= T(gradient_tolerance) && break
        copyto!(direction, projected)
        coefficients = Vector{T}(undef, length(steps))
        for history in length(steps):-1:1
            coefficients[history] =
                inverse_curvatures[history] *
                dot(steps[history], direction)
            direction .-= coefficients[history] .* gradient_steps[history]
        end
        if !isempty(steps)
            last_step = steps[end]
            last_gradient_step = gradient_steps[end]
            direction .*= dot(last_step, last_gradient_step) /
                dot(last_gradient_step, last_gradient_step)
        end
        for history in eachindex(steps)
            coefficient =
                inverse_curvatures[history] *
                dot(gradient_steps[history], direction)
            direction .+=
                (coefficients[history] - coefficient) .* steps[history]
        end
        direction .*= -one(T)
        dot(projected, direction) < zero(T) || (direction .= -projected)

        accepted = false
        candidate_value = value
        step_length = one(T)
        while step_length >= T(minimum_step) && cost < max_cost && !stop()
            @. candidate = clamp(point + step_length * direction, lower, upper)
            parameter_step .= candidate .- point
            norm(parameter_step, Inf) > T(displacement_tolerance) || break
            candidate_value, _, incurred = evaluate(candidate, false)
            cost += incurred
            stop() && return point, value, cost
            if isfinite(candidate_value) &&
                    candidate_value <= value +
                        T(1e-4) * dot(gradient, parameter_step)
                accepted = true
                break
            end
            step_length /= T(2)
        end
        if !accepted && retry_identity && !isempty(steps)
            empty!(steps)
            empty!(gradient_steps)
            empty!(inverse_curvatures)
            continue
        end
        accepted || break

        improvement = value - candidate_value
        copyto!(point, candidate)
        value = candidate_value
        improvement >= zero(T) &&
            improvement < T(value_tolerance) && break
        cost + gradient_cost <= max_cost || break
        _, next_gradient, incurred = evaluate(point, true)
        cost += incurred
        isnothing(next_gradient) && break
        gradient_step .= next_gradient .- gradient
        curvature = dot(parameter_step, gradient_step)
        if isfinite(curvature) &&
                curvature >
                    sqrt(eps(T)) * norm(parameter_step) * norm(gradient_step)
            if length(steps) == memory
                popfirst!(steps)
                popfirst!(gradient_steps)
                popfirst!(inverse_curvatures)
            end
            push!(steps, copy(parameter_step))
            push!(gradient_steps, copy(gradient_step))
            push!(inverse_curvatures, inv(curvature))
        elseif reset_on_bad_curvature
            empty!(steps)
            empty!(gradient_steps)
            empty!(inverse_curvatures)
        end
        gradient = next_gradient
    end
    return point, value, cost
end
