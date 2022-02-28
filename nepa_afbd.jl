import JSON
using LightGraphs, MetaGraphs, DataStructures, StatsBase, Random, Statistics, JLD2

include("utils.jl")
include("shortest.jl")
include("refine.jl")
#include("path_split.jl")
include("play.jl")
include("checks.jl")
include("nrpa_common_afbd.jl")

function NRPA(sn::MetaGraph{Int64, Float64},
              vnr::MetaGraph{Int64, Float64}, 
              level::Int64, N::Int64, 
              policy::Union{DefaultDict{String, Float64, Float64}, Dict{String, Float64}}, 
              distances, 
              solver::Function,
              order_links::Bool,
              max_bw_sn,
              max_bw_vnr,
              sum_bw_sn,
              sum_bw_vnr,
              refined_solutions,
              level_refine,
              num_its,
              distances_refine,
              num_moves_refine)
    if level == 0
        # copying is slow
        # we don't use modifications to vnr for reward calculations
        # hence do not copy it as it consumes lots of time
        # as a result, after NRPA, vnr contains the last calculated embedding, which is not the best one
    
        global test += 1
        score, sequence = playout(copy_graph(sn), vnr, policy, distances, solver, order_links, max_bw_sn, max_bw_vnr, sum_bw_sn, sum_bw_vnr)
        
        return score, sequence
    else
        best_score::Float64 = -9999
        best_seq = Int64[]
        for i = 1:N
            reward, sequence = NRPA(sn, vnr, level - 1, N, policy, distances, solver, order_links, 
                                    max_bw_sn, max_bw_vnr, sum_bw_sn, sum_bw_vnr, refined_solutions, 
                                    level_refine, num_its, distances_refine, num_moves_refine)

            if level == level_refine && reward != 0
                s, v, sequence, reward, counter_solves = refine_solution(sn, vnr, sequence, reward, solver, order_links, num_its, num_moves_refine, distances_refine)
                global counter_refines += num_moves_refine * num_its
                refined_solutions[string(sequence)] = (copy_graph(v), copy_graph(s), reward)
            end
            
            if reward > best_score
                best_score = reward
                best_seq = sequence
            end
            if reward > 0
                policy = adapt(policy, best_seq, copy_graph(sn), vnr, distances, solver, order_links, max_bw_sn, max_bw_vnr, sum_bw_sn, sum_bw_vnr)
            end
        end
        return best_score, best_seq
    end
end

# This is where the magic happens
function run(instance_path,
             solver_sim, 
             solver_final, 
             level,
             N,
             dist_heuristic, 
             order_links, 
             log_file,
             level_refine,
             num_refines)

    events, instance = load_instance(instance_path)
    
    accepted::Int64 = 0
    refused::Int64 = 0
    # do not use DefaultDict as it will initialize all lists with different references to the same list
    scores = Dict{Int64, Vector{Any}}()
    
    # sn loaded once
    sn = instance[-1]
    future_leaves = Int64[]
    l_dep = length(events)
    arv = 0
    global counter_refines = 0
    
    while !isempty(events)
        println(arv)
        check_bounds_are_respected(sn)
        type::String, slice::Int64 = popfirst!(events)
        if type == "arrival"
            arv += 1
            vnr = instance[slice]

            # always precompute distances, they are used for refining also
            if dist_heuristic
                distances = precompute_distances(sn, 0)
            else
                distances = nothing
            end

            #policy = DefaultDict{String, Float64}(0.0)
            policy = Dict{String, Float64}()

            
            sn_prec = copy_graph(sn)
            global test = 0
            # reorder vnr so the first node treated is the one with the least legal moves
            vnr = reorder_vnr(vnr, sn, max_bw_sn(sn), max_bw_vnr(vnr), sum_bw_sn(sn), sum_bw_vnr(vnr))
            instance[slice] = vnr
            vnr_s = copy_graph(vnr)
            
            refined_solutions = Dict()

            score, seq =  @time NRPA(sn, vnr_s, level, N, policy, distances, solver_sim, order_links, 
                                     max_bw_sn(sn), max_bw_vnr(vnr), sum_bw_sn(sn), sum_bw_vnr(vnr), 
                                     refined_solutions, level_refine, nv(vnr), precompute_distances(sn, 0), num_refines)
            println(test)
            if !haskey(scores, nv(vnr))
                scores[nv(vnr)] = []
            end
            push!(scores[nv(vnr)], score)

            if score > 0
                curr_node = 1
                sn_2 = copy(sn)
                vnr_2 = copy(vnr)
                for action in seq
                    sn_2, vnr_2, curr_node, rw, done = play(sn_2, vnr_2, curr_node, action, solver_final, order_links)
                    # sometimes the standard link solver gives worst result than links found by refining
                    # if that is the case, replace the vnr by the vnr found when refining
                    if rw < refined_solutions[string(seq)][3] && done
                        score = refined_solutions[string(seq)][3]

                        vnr = refined_solutions[string(seq)][1]
                        sn = refined_solutions[string(seq)][2]
                    elseif done
                        sn = sn_2
                        vnr = vnr_2
                    end
                end
                instance[slice] = vnr
                push!(future_leaves, slice)

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
        print(io,N, ",", level,",", dist_heuristic,",", level_refine, ",", num_refines, ",", counter_refines,",")
    end
end

function main(instance_path, log_file, level, N, NRPAD, level_refine, num_refines, seed)
    Random.seed!(seed)
    solver = place_links_sp

    t = @elapsed run(instance_path, solver, solver, level, N, NRPAD, true, log_file, level_refine, num_refines) 
    open(log_file,"a") do io
        println(io, t)
    end
end



main(ARGS[1], ARGS[2], parse(Int64,ARGS[3]), parse(Int64, ARGS[4]), parse(Bool,ARGS[5]), parse(Int64,ARGS[6]), parse(Int64,ARGS[7]), parse(Int64,ARGS[8]))