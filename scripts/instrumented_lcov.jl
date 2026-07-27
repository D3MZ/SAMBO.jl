function coverage_files(source::AbstractString)
    directory = dirname(source)
    prefix = basename(source) * "."
    return filter(readdir(directory; join=true)) do path
        name = basename(path)
        startswith(name, prefix) && endswith(name, ".cov")
    end
end

function instrumented_counts(source::AbstractString)
    counts = Dict{Int,Int}()
    for coverage in coverage_files(source)
        for (line, text) in enumerate(eachline(coverage))
            fields = split(strip(text); limit=2)
            isempty(fields) && continue
            count = tryparse(Int, fields[1])
            isnothing(count) && continue
            counts[line] = get(counts, line, 0) + count
        end
    end
    return counts
end

function write_lcov(io::IO, roots)
    for root in roots
        for (directory, _, files) in walkdir(root)
            for file in sort(files)
                endswith(file, ".jl") || continue
                source = joinpath(directory, file)
                counts = instrumented_counts(source)
                isempty(counts) && continue
                println(io, "SF:", source)
                for line in sort!(collect(keys(counts)))
                    println(io, "DA:", line, ',', counts[line])
                end
                println(io, "LF:", length(counts))
                println(io, "LH:", count(>(0), values(counts)))
                println(io, "end_of_record")
            end
        end
    end
end

write_lcov(stdout, isempty(ARGS) ? ("src", "ext") : ARGS)
