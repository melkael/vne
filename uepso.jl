import JSON
using Graphs, MetaGraphs, DataStructures, StatsBase, Random, Statistics, JLD2

include("utils.jl")
include("shortest.jl")
#include("refine.jl")
#include("path_split.jl")
include("checks.jl")
include("play.jl")

function nr_sn(sn, node)
    cpu = get_prop(sn, node, :cpu_max) - get_prop(sn, node, :cpu_used) 
    bw = 0
    for n in neighbors(sn, node)
        bw += get_prop(sn, node, n, :BW_max) - get_prop(sn, node, n, :BW_used)
    end 
    return bw * cpu
end 

function nr_vnr(vnr, node)
    cpu = get_prop(vnr, node, :cpu) 
    bw = 0
    for n in neighbors(vnr, node)
        bw += get_prop(vnr, node, n, :BW)
    end 
    return bw * cpu
end


function random_particle(legals, sn, vnr, scores_sn, idx=0)
    legals = deepcopy(legals)
    
    particle = []
    for i in 1:nv(vnr)
        weights = scores_sn[legals[i]]
        if isempty(legals[i])
            return []
        end

        probs = weights ./ sum(weights)
        node = sample(legals[i], Weights(probs))
        
        push!(particle, node)
        for j in legals
            remove!(j, node)
        end
    end
    return particle
end

function get_values(population, sn, vnr, solver, order_links)
    values = []
    for particle in population
        sn_cpy = copy_graph(sn)
        vnr_cpy = copy_graph(vnr)

        done = false
        reward = 0.0
        curr_node = 1
        # place the vnr first
        for action in particle
            sn_cpy, vnr_cpy, curr_node, reward, done = play(sn_cpy, vnr_cpy, curr_node, Int64(action), solver, order_links)
            if done
                break
            end
        end
        push!(values, reward)
    end
    return values
end

function minus(X1, X2)
    result = []
    for i in 1:length(X1)
        if X1[i] == X2[i]
            push!(result, 1)
        else
            push!(result, 0)
        end
    end
    return result
end

function plus(P1, V1, P2, V2, P3, V3)
    result = []
    for i in 1:length(V1)
        if V1[i] == V2[i] == V3[i]
            push!(result, V1[i])
        else
            if rand() < P1
                push!(result, V1[i])
            elseif rand() < P2 + P1
                push!(result, V2[i])
            else
                push!(result, V3[i])
            end
        end
    end
    return result
end

function times(X, V, legals, scores_sn, idx=0)
    legals = deepcopy(legals)
    result = []
    for i in 1:length(X)
        if V[i] == 1
            push!(result, X[i])
            for k in legals
                remove!(k, X[i])
            end
        else
            push!(result, -1)
        end
    end

    for i in 1:length(X)
        if result[i] == -1
            # choose random legal value
            weights = scores_sn[legals[i]]
            if isempty(legals[i])
                return []
            end
            probs = weights ./ sum(weights)
            node = sample(legals[i], Weights(probs))
            result[i] = node
            for k in legals
                remove!(k, node)
            end
        end
    end
    return result
end

function UEPSO(sn, vnr, num_particles, num_its, solver, order_links, idx, max_time, distances)
    println(idx)
    P1 = 0.1
    P2 = 0.2
    P3 = 0.7
    moves = []
    for i in 1:nv(vnr)
        push!(moves, get_legal_moves(sn, vnr, i))
    end

    scores_sn = []
    for j in 1:nv(sn)
        push!(scores_sn, nr_sn(sn, j))
    end

    population = []
    for i in 1:num_particles
        push!(population, random_particle(moves, sn, vnr, scores_sn))
    end
    velocities = []
    for i in 1:num_particles
        push!(velocities, Random.bitrand(nv(vnr)) .+ 0 )
    end

    values = get_values(population, sn, vnr, solver, order_links)
    gBest_score = findmax(values)[1]
    gBest = copy(population[findmax(values)[2]])

    pBest = []

    for i in 1:length(values)
        push!(pBest, [copy(population[i]), values[i]])
    end
    time = 0
    #for i in 1:num_its
    while time < max_time
        time += @elapsed begin
        # update position vector and velocity vector
            for k in 1:length(population)
                if values[k] > 0
                    velocities[k] = plus(P1, velocities[k], P2, minus(pBest[k][1], population[k]), P3, minus(gBest, population[k]))
                    population[k] = times(population[k], velocities[k], moves, scores_sn, idx)
                else
                    velocities[k] = Random.bitrand(nv(vnr)) .+ 0
                    population[k] = random_particle(moves, sn, vnr, scores_sn, idx)
                end
            end
            values = get_values(population, sn, vnr, solver, order_links)
            for k in 1:length(population)
                if values[k] > pBest[k][2]
                    pBest[k][2] = values[k]
                    pBest[k][1] = population[k]
                end
                if pBest[k][2] > gBest_score
                    gBest_score = pBest[k][2]
                    gBest = copy(population[k])
                end
            end
        end
    end
    return gBest_score, gBest
    
end

# This is where the magic happens
function run_UEPSO(instance_path,
    solver_sim, 
    solver_final, 
    num_particles, 
    num_its,
    order_links,
    log_file,
    max_time)

    events, instance = load_instance(instance_path)

    accepted::Int64 = 0
    refused::Int64 = 0
    scores = Dict{Int64, Vector{Any}}()

    # sn loaded once
    sn = instance[-1]
    future_leaves = Int64[]
    l_dep = length(events)
    
    while !isempty(events)
        check_bounds_are_respected(sn)
        type::String, slice::Int64 = popfirst!(events)
        if type == "arrival"
            vnr = instance[slice]
    
            # make a hard copy, since NRPA modifies the vnr with garbage (sn is untouched)
            vnr = reorder_vnr_uepso(vnr)
            instance[slice] = vnr
            vnr_s = copy_graph(vnr)

            sn_prec = copy_graph(sn)

            score, seq =  @time UEPSO(sn, vnr, num_particles, num_its, solver_sim, order_links, refused+accepted, max_time, precompute_distances(sn, 0))
            if !haskey(scores, nv(vnr))
                scores[nv(vnr)] = []
            end
            push!(scores[nv(vnr)], score)
            if score > 0
                curr_node = 1
                for action in seq
                    sn, vnr, curr_node, _, _ = play(sn, vnr, curr_node, action, solver_final, order_links)
                end
                push!(future_leaves, slice)
                # log results
                accepted += 1
                check_each_vn_uses_different_node(vnr)
                check_each_vn_uses_resource_amount(sn, sn_prec, vnr)  
                check_each_vl_uses_resource_amount(sn, sn_prec, vnr)  
            else
                refused += 1
            end
            clear_occupied!(sn)
            else
                if issubset([slice], future_leaves)
                vnr = instance[slice]
                remove!(future_leaves, slice)
                # free the cpu
                for v in vertices(vnr)
                    p = props(vnr, v)
                    set_prop!(sn, p[:host_node], :cpu_used, get_prop(sn, p[:host_node], :cpu_used) - p[:cpu])
                end
                # free the BW
                for e in edges(vnr)
                    p = props(vnr, e)
                    for (key, value) in p[:vlink]
                        set_prop!(sn, Edge(key), :BW_used, get_prop(sn, Edge(key), :BW_used) - value)
                    end
                end
            end
        end
    end
    stats, glob_r_c = get_stats(scores, accepted)
    
    
    open(log_file,"a") do io
        print(io,accepted, ",", glob_r_c, ",")
        for (key,value) in stats
            print(io, key,":",value,",")
        end
        print(io,num_particles,",",num_its,",")
    end
end


function main(instance_path, log_file, num_particles, num_its, seed, max_time)
    Random.seed!(seed)    
    
    solver = place_links_sp
    t = @elapsed run_UEPSO(instance_path, solver, solver, num_particles, num_its, true, log_file, max_time)
    open(log_file,"a") do io
        println(io, t)
    end
end

println(ARGS[6])
println(ARGS)

main(ARGS[1], ARGS[2], parse(Int64, ARGS[3]), parse(Int64, ARGS[4]), parse(Int64, ARGS[5]), parse(Float64, ARGS[6]))