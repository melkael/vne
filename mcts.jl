import JSON
using Graphs, MetaGraphs, DataStructures, StatsBase, Random, Statistics, JLD2

include("utils.jl")
include("shortest.jl")
#include("refine.jl")
#include("path_split.jl")
include("play.jl")
include("checks.jl")

struct State
    sn::MetaGraph{Int64, Float64}
    vnr::MetaGraph{Int64, Float64} 
    nf_to_map::Int64
end


mutable struct Node
    parent::Union{Node, Nothing}
    children::Dict{Int64, Union{Nothing, Node}}
    visit_times::Int64
    value::Float64
    state::State
    expansion_count::Int64
    is_expandable::Bool
end

function uct(node::Node)
    best_val = typemin(Int)
    best_node = nothing
    
    if length(node.children) == 0
        return node, true
    end

    for (idx, n) in node.children
        if n != nothing
            uct = n.value/n.visit_times + 2 * sqrt(log(node.visit_times)/n.visit_times)
            if uct >= best_val
                best_val = uct
                best_node = n
            end
        end
        # if we found a nothing, it means we should stop descending and expand the current node
        if n == nothing
            return node, true
        end
    end
    return best_node, false
end

function is_final(node::Node)
    return length(keys(node.children)) == 0
end

function expand(node::Node, solver)
    for i in keys(node.children)
        if node.children[i] == nothing
            found = true
            node.expansion_count += 1
            if node.expansion_count == length(keys(node.children))
                node.is_expandable = false
            end
            sn = copy_graph(node.state.sn)
            vnr = copy_graph(node.state.vnr)

            sn, vnr, nf_to_map, reward, done = play(sn, vnr, node.state.nf_to_map, i, solver, true)

            s = State(sn, vnr, nf_to_map)


            children = Dict{Int64, Union{Nothing, Node}}()
            is_expandable = false
            if node.state.nf_to_map < nv(node.state.vnr)
                for m in get_legal_moves(sn, vnr, node.state.nf_to_map+1)
                    is_expandable = true
                    children[m] = nothing
                end
            end

            n = Node(node, children, 0, 0, s, 0, is_expandable)
            node.children[i] = n
            #println(node.expansion_count, " ", length(keys(node.children)), " ", node.is_expandable)
            return n
        end
    end
end

function simulate(node, solver)
    sn = copy_graph(node.state.sn)
    vnr = copy_graph(node.state.vnr)

    nf_to_map = node.state.nf_to_map
    reward = 0.0

    for i in nf_to_map:nv(vnr)
        try
            sn, vnr, _, reward, done = play(sn, vnr, i, rand(get_legal_moves(sn, vnr, i)), solver, true)
            if done
                break
            end
        catch e
            # if LoadError no legal move exists so we return 0
            if isa(e, LoadError)
                return 0.0
            end
        end
    end
    return reward
end

function backpropagate(node::Node, reward::Float64)
    #println(reward)
    while node != nothing
        node.visit_times += 1
        node.value += reward
        node = node.parent
    end 
end

function best_child(node::Node)
    best_node = nothing
    best_score = typemin(Int)
    best_action = -1
    for (key, n) in node.children
        if n != nothing
            score = n.value / n.visit_times
            if score >= best_score
                best_action = key
                best_score = score
                best_node = n
            end
        end
    end
    return best_node, best_action
end

function search(node::Node, sn::MetaGraph{Int64, Float64}, vnr::MetaGraph{Int64, Float64}, beta::Int64, solver)
    should_stop = false
    # Selection
    while !node.is_expandable && !should_stop
        node, should_stop = uct(node)
    end

    if !is_final(node)
        node = expand(node, solver)
    end
    reward = simulate(node, solver)
    backpropagate(node, reward)
end

function MCTS(sn::MetaGraph{Int64, Float64}, vnr::MetaGraph{Int64, Float64}, beta::Int64, solver)
    s = State(copy_graph(sn), copy_graph(vnr), 1)
    children = Dict{Int64, Union{Nothing, Node}}()
    sn = copy_graph(sn)
    vnr = copy_graph(vnr)
    
    for m in get_legal_moves(sn, vnr, 1)
        children[m] = nothing
    end

    root = Node(nothing, children, 0, 0, s, 0, true)
    sequence = []
    done = false
    curr_node = 1
    reward = 0

    while !done
        for i in 1:beta
            search(root, copy_graph(sn), copy_graph(vnr), beta, solver)
        end
        root, best_action = best_child(root)
        if root == nothing
            return 0, []
        end
        root.parent = nothing
        push!(sequence, best_action)
        sn, vnr, curr_node, reward, done = play(sn, vnr, curr_node, best_action, solver, true)
    end
    return reward, sequence
end


function run_MCTS(instance_path,
    solver_sim, 
    solver_final,
    order_links::Bool,
    log_file,
    total_budget)

    events, instance = load_instance(instance_path)

    accepted::Int64 = 0
    refused::Int64 = 0
    scores = Dict{Int64, Vector{Any}}([])

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
            vnr_s = copy_graph(vnr)
            #policy = DefaultDict{String, Float64}(0.0)
            policy = Dict{String, Float64}()
   
            sn_prec = copy_graph(sn)
            
            beta = div(total_budget, nv(vnr))

            score, seq =  @time MCTS(sn, vnr_s, beta, solver_sim)
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
        print(io,total_budget,",")
    end
end


function main(instance_path, log_file, total_budget, seed)
    Random.seed!(seed)
    solver = place_links_sp

    t = @elapsed run_MCTS(instance_path, solver, solver, true, log_file, total_budget)

    open(log_file, "a") do io
        println(io, t)
    end
end

println(ARGS)
main(ARGS[1], ARGS[2], parse(Int64,ARGS[3]), parse(Int64, ARGS[4]))