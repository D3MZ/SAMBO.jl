module CorrectedInteropContracts

using MLJTuning
using OptimizationBase
using Random
using SAMBO
using Serialization
using Test

const CorrectedMLJBase = MLJTuning.MLJBase

mutable struct CorrectedLinearModel <: CorrectedMLJBase.Deterministic
    slope::Float64
end

CorrectedMLJBase.fit(
    model::CorrectedLinearModel,
    verbosity::Int,
    X,
    y,
) = model.slope, nothing, nothing
CorrectedMLJBase.predict(
    ::CorrectedLinearModel,
    slope,
    Xnew,
) = slope .* Xnew.x

@testset "corrected interoperability contracts" begin
    @testset "OptimizationBase common contracts" begin
        bounded = OptimizationBase.OptimizationProblem(
            (point, parameters) -> sum(abs2, point),
            nothing;
            lb=fill(-1.0, 2),
            ub=fill(1.0, 2),
        )
        without_u0 = OptimizationBase.solve(
            bounded,
            SMBO(candidate_pool=32);
            maxiters=6,
            rng=Xoshiro(201),
        )
        @test without_u0.stats.fevals == 6
        @test without_u0.original.sense isa Minimize

        maximization = OptimizationBase.OptimizationProblem(
            (point, parameters) -> -(point[1] - 0.7)^2,
            nothing;
            lb=[0.0],
            ub=[1.0],
            sense=OptimizationBase.MaxSense,
        )
        maximized = OptimizationBase.solve(
            maximization,
            SMBO(candidate_pool=32);
            maxiters=12,
            rng=Xoshiro(202),
        )
        @test maximized.original.sense isa Maximize
        @test maximized.objective > -0.05

        in_place_function = OptimizationBase.OptimizationFunction(
            (point, parameters) -> sum(abs2, point);
            cons=(residual, point, parameters) -> begin
                residual[1] = point[1] + point[2]
                residual
            end,
        )
        in_place_problem = OptimizationBase.OptimizationProblem(
            in_place_function,
            nothing;
            lb=fill(-1.0, 2),
            ub=fill(1.0, 2),
            lcons=[0.5],
            ucons=[Inf],
        )
        in_place = OptimizationBase.solve(
            in_place_problem,
            SCEUA();
            maxiters=30,
            rng=Xoshiro(203),
        )
        @test sum(in_place.u) >= 0.5

        out_of_place_function = OptimizationBase.OptimizationFunction{false}(
            (point, parameters) -> sum(abs2, point);
            cons=(point, parameters) -> [point[1] + point[2]],
        )
        out_of_place_problem = OptimizationBase.OptimizationProblem(
            out_of_place_function,
            nothing;
            lb=fill(-1.0, 2),
            ub=fill(1.0, 2),
            lcons=[0.5],
            ucons=[Inf],
        )
        out_of_place = OptimizationBase.solve(
            out_of_place_problem,
            SCEUA();
            maxiters=30,
            rng=Xoshiro(204),
        )
        @test sum(out_of_place.u) >= 0.5

        callback_calls = Ref(0)
        callback_stopped = OptimizationBase.solve(
            bounded,
            SMBO(candidate_pool=16);
            maxiters=10,
            maxtime=1.0,
            abstol=1e-8,
            reltol=1e-6,
            callback=(state, loss) -> begin
                callback_calls[] += 1
                @test state isa OptimizationBase.OptimizationState
                @test state.objective == loss
                true
            end,
            rng=Xoshiro(205),
        )
        @test callback_calls[] == 1
        @test callback_stopped.retcode ==
            OptimizationBase.ReturnCode.Terminated

        iteration_limited = OptimizationBase.solve(
            bounded,
            SMBO(initial_points=2, batch_size=1, candidate_pool=16);
            maxiters=1,
            maximum_evaluations=10,
            rng=Xoshiro(206),
        )
        @test iteration_limited.original.retcode == :iteration_limit
        @test iteration_limited.retcode ==
            OptimizationBase.ReturnCode.MaxIters

        extension =
            Base.get_extension(SAMBO, :SAMBOOptimizationExt)
        in_place_constraint = extension._constraint(in_place_problem)
        constraint_point = [0.3, 0.3]
        in_place_constraint(constraint_point)
        @test @allocated(in_place_constraint(constraint_point)) == 0
        expected_codes = (
            success=OptimizationBase.ReturnCode.Success,
            evaluation_limit=OptimizationBase.ReturnCode.MaxIters,
            iteration_limit=OptimizationBase.ReturnCode.MaxIters,
            time_limit=OptimizationBase.ReturnCode.MaxTime,
            stalled=OptimizationBase.ReturnCode.Stalled,
            callback_stop=OptimizationBase.ReturnCode.Terminated,
            infeasible_space=OptimizationBase.ReturnCode.Infeasible,
            numerical_failure=OptimizationBase.ReturnCode.Failure,
            space_exhausted=OptimizationBase.ReturnCode.Success,
        )
        for (native, expected) in pairs(expected_codes)
            @test extension._returncode(native) == expected
        end
        @test extension._returncode(:unknown) ==
            OptimizationBase.ReturnCode.Failure

        invalid_constraint_function = OptimizationBase.OptimizationFunction(
            (point, parameters) -> sum(abs2, point);
            cons=() -> nothing,
        )
        invalid_constraint_problem = OptimizationBase.OptimizationProblem(
            invalid_constraint_function,
            nothing;
            lb=fill(-1.0, 2),
            ub=fill(1.0, 2),
            lcons=[0.0],
            ucons=[1.0],
        )
        @test_throws ArgumentError extension._constraint(
            invalid_constraint_problem,
        )
    end

    @testset "MLJTuning pending and orientation" begin
        model = CorrectedLinearModel(0.5)
        parameter_range = range(
            model,
            :slope;
            lower=0.0,
            upper=2.0,
        )
        tuning = SAMBOTuning(
            algorithm=SMBO(
                initial_points=2,
                candidate_pool=32,
                batch_size=2,
            ),
            rng=Xoshiro(210),
        )
        state = MLJTuning.setup(
            tuning,
            model,
            parameter_range,
            6,
            0,
        )
        models, state = MLJTuning.models(
            tuning,
            model,
            NamedTuple[],
            state,
            6,
            0,
        )
        @test length(models) == 2
        pending = state.pending
        repeated, state = MLJTuning.models(
            tuning,
            model,
            NamedTuple[],
            state,
            4,
            0,
        )
        @test isempty(repeated)
        @test state.pending === pending

        loss_values = [0.8, 0.2]
        loss_history = [
            (
                measurement=[loss_values[index]],
                measure=[CorrectedMLJBase.rms],
            )
            for index in eachindex(models)
        ]
        _, state = MLJTuning.models(
            tuning,
            model,
            loss_history,
            state,
            4,
            0,
        )
        @test collect(objectivevalues(trace(state.solver))) ==
            loss_values

        zero_state = MLJTuning.setup(
            tuning,
            model,
            parameter_range,
            1,
            0,
        )
        zero_models, zero_state = MLJTuning.models(
            tuning,
            model,
            NamedTuple[],
            zero_state,
            0,
            0,
        )
        @test isempty(zero_models)
        @test isnothing(zero_state.pending)

        score_tuning = SAMBOTuning(
            algorithm=SMBO(
                initial_points=1,
                candidate_pool=16,
                batch_size=1,
            ),
            rng=Xoshiro(211),
        )
        score_state = MLJTuning.setup(
            score_tuning,
            model,
            parameter_range,
            3,
            0,
        )
        _, score_state = MLJTuning.models(
            score_tuning,
            model,
            NamedTuple[],
            score_state,
            3,
            0,
        )
        score_history = [(
            measurement=[0.9],
            measure=[CorrectedMLJBase.rsquared],
        )]
        _, score_state = MLJTuning.models(
            score_tuning,
            model,
            score_history,
            score_state,
            2,
            0,
        )
        @test only(objectivevalues(trace(score_state.solver))) == -0.9

        X = (x=collect(range(-1.0, 1.0; length=24)),)
        y = copy(X.x)
        for measure in (
            CorrectedMLJBase.rms,
            CorrectedMLJBase.rsquared,
        )
            tuned = MLJTuning.TunedModel(
                model=CorrectedLinearModel(0.5),
                tuning=SAMBOTuning(
                    algorithm=SMBO(
                        initial_points=2,
                        candidate_pool=24,
                    ),
                    rng=Xoshiro(212),
                ),
                range=parameter_range,
                measure=measure,
                n=4,
                resampling=CorrectedMLJBase.Holdout(
                    fraction_train=0.7,
                ),
            )
            machine = CorrectedMLJBase.machine(tuned, X, y)
            CorrectedMLJBase.fit!(machine; verbosity=0)
            report = CorrectedMLJBase.report(machine)
            @test length(report.history) == 4
            @test report.best_history_entry.measure[1] == measure
        end
    end

    @testset "checkpoint memory and serialization contract" begin
        problem = Problem(
            x -> (x.x - 0.3)^2,
            SearchSpace(x=Continuous(0.0, 1.0)),
        )
        state = init(
            problem,
            SMBO(initial_points=2, candidate_pool=16);
            maximum_evaluations=5,
            rng=Xoshiro(220),
        )
        pending = ask!(state, 2)
        saved = checkpoint(state)
        saved.pending[pending.identifier].points[1, 1] = 0.0
        @test state.pending[pending.identifier].points[1, 1] != 0.0

        pristine = checkpoint(state)
        io = IOBuffer()
        serialize(io, pristine)
        seekstart(io)
        roundtrip = deserialize(io)
        memory_restored = restore(problem, pristine)
        serialized_restored = restore(problem, roundtrip)
        values = [(point.x - 0.3)^2 for point in pending]
        tell!(state, pending, values)
        tell!(memory_restored, pending, values)
        tell!(serialized_restored, pending, values)
        original_batch = ask!(state, 1)
        memory_batch = ask!(memory_restored, 1)
        serialized_batch = ask!(serialized_restored, 1)
        @test latentpoints(original_batch) == latentpoints(memory_batch)
        @test latentpoints(memory_batch) ==
            latentpoints(serialized_batch)
        @test typeof(memory_restored.core.callback) ==
            typeof(serialized_restored.core.callback)
        @test typeof(memory_restored.core.executor) ==
            typeof(serialized_restored.core.executor)
    end
end

end
