import JSON
using Graphs, MetaGraphs, DataStructures, StatsBase, Random, Statistics, JLD2

include("utils.jl")
include("shortest.jl")
#include("refine.jl")
#include("path_split.jl")
include("play.jl")
include("checks.jl")
include("nrpa_common.jl")



####################################################################
####################### NRPA ALGORITHM #############################
####################################################################


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
              sum_bw_vnr)
    if level == 0
        # copying is slow
        # we don't use modifications to vnr for reward calculations
        # hence do not copy it as it consumes lots of time
        # as a result, after NRPA, vnr contains the last calculated embedding, which is not the best one
    

        score, sequence = playout(copy_graph(sn), vnr, policy, distances, solver, order_links, max_bw_sn, max_bw_vnr, sum_bw_sn, sum_bw_vnr)
        
        return score, sequence
    else
        best_score::Float64 = -9999
        best_seq = Int64[]
        for i = 1:N
            reward, sequence = NRPA(sn, vnr, level - 1, N, policy, distances, solver, order_links, max_bw_sn, max_bw_vnr, sum_bw_sn, sum_bw_vnr)

            if reward > best_score
                best_score = reward
                best_seq = sequence
            end
            if best_score > 0
                policy = adapt(policy, best_seq, copy_graph(sn), vnr, distances, solver, order_links, max_bw_sn, max_bw_vnr, sum_bw_sn, sum_bw_vnr)
            end
        end
        return best_score, best_seq
    end
end


####################################################################
################ HELPER FUNCTIONS USED IN NRPA #####################
####################################################################


function default_weight!(policy, key)
    policy[key] = 0
end

function default_weight!(policy, key, distances)
    if distances == nothing
        policy[key] = 0
        return
    end
    if key == ""
        policy[key] = 0
    else
        nodes = split(key, ",")
        k = 0
        # first "node" is always empty string, last one is the node we are trying to rate
         for i in nodes[2:end-1]
            k += distances[parse(Int64, nodes[end])][parse(Int64, i)]
        end
        policy[key] = -k / (length(nodes)-1)
    end
end


####################################################################
################ HELPER FUNCTIONS USED IN MAIN #####################
####################################################################


function precompute_distances(sn, threshold)
    sn2 = copy_graph(sn)
    for e in edges(sn)
        p = props(sn, e)
        if p[:BW_max] - p[:BW_used] < threshold
            rem_edge!(sn2, e)
        end
    end
    distances = []
     for node in 1:nv(sn2)
        push!(distances, gdistances(sn2, node))
    end
    return distances 
end

function median_bw(vnr)
    a = []
    for e in edges(vnr)
        push!(a, get_prop(vnr, e, :BW))
    end
    return median(a)
end



# This is where the magic happens
function run(instance_path,
             solver_sim, 
             solver_final, 
             level,
             N,
             dist_heuristic, 
             order_links::Bool,
             log_file)

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
    while !isempty(events)
        println(arv)
        
        check_bounds_are_respected(sn)

        type::String, slice::Int64 = popfirst!(events)
        if type == "arrival"
            arv += 1
            vnr = instance[slice]

            if dist_heuristic
                distances = precompute_distances(sn, median_bw(vnr))
            else
                distances = nothing
            end

            policy = Dict{String, Float64}()

            sn_prec = copy_graph(sn)
            # reorder vnr so the first node treated is the one with the least legal moves
            vnr = reorder_vnr(vnr, sn, max_bw_sn(sn), max_bw_vnr(vnr), sum_bw_sn(sn), sum_bw_vnr(vnr))
            instance[slice] = vnr
            vnr_s = copy_graph(vnr)

            score, seq =  @time NRPA(sn, vnr_s, level, N, policy, distances, solver_sim, order_links, max_bw_sn(sn), max_bw_vnr(vnr), sum_bw_sn(sn), sum_bw_vnr(vnr))
        
            if !haskey(scores, nv(vnr))
                scores[nv(vnr)] = []
            end
            push!(scores[nv(vnr)], score)
            if score > 0
                curr_node = 1
                # play the full sequence in order to build the vnr up
                for action in seq
                    sn, vnr, curr_node, rw, _ = play(sn, vnr, curr_node, action, solver_final, order_links)
                end
                # add slice index to future slices to be removed
                push!(future_leaves, slice)
                accepted += 1
                # safety checks
                check_each_vn_uses_different_node(vnr)
                check_each_vn_uses_resource_amount(sn, sn_prec, vnr)  
                check_each_vl_uses_resource_amount(sn, sn_prec, vnr)  
            else
                refused += 1
            end
            # occupied holds 1 if sn node is already in use by other node of same vnr. set all to 0 before each new placement
            clear_occupied!(sn)
        # if departure remove vnodes and vlinks
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
        print(io,N, ",", level,",", dist_heuristic,",")
    end
end

# this holds the known legal states, if a state is legal it will remain legal anyways, no need to recompute
# Do not forget to initialize it before each run !
#known_legals = Dict{String, Vector{Int64}}()

function main(instance_path, log_file, level, N, NRPAD, seed)
    Random.seed!(seed)
    solver = place_links_sp

    t = @elapsed run(instance_path, solver, solver, level, N, NRPAD, true, log_file) 
    open(log_file,"a") do io
        println(io, t)
    end
end

main(ARGS[1], ARGS[2], parse(Int64,ARGS[3]), parse(Int64, ARGS[4]), parse(Bool,ARGS[5]), parse(Int64,ARGS[6]))