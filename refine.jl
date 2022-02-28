###########################################################
############# Solution refinement methods #################
###########################################################

function most_expensive_node(vnr::MetaGraph{Int64, Float64}, prec_nodes)
    most_expensive = -1
    highest_cost = 0
    for i in 1:nv(vnr)
        price = 0
        for j in neighbors(vnr, i)
            price += sum(values(get_prop(vnr, i, j, :vlink)))
        end
        price /= length(neighbors(vnr, i))
        if price > highest_cost && !issubset([i], prec_nodes)
            highest_cost = price
            most_expensive = i
        end
    end
    return most_expensive
end

function remove_vnr!(sn, vnr)
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

function clear_vnr!(vnr::MetaGraph{Int64, Float64})
    for e in edges(vnr)
        set_prop!(vnr, e, :vlink, Dict())
    end
    for v in 1:nv(vnr)
        set_prop!(vnr, v, :host_node, nothing)
    end
end

function clear_sn_from_virtual_node!(sn::MetaGraph{Int64, Float64}, vnr::MetaGraph{Int64, Float64}, node::Int64)
    # free the cpu
    for v in node:nv(vnr)
        p = props(vnr, v)
        set_prop!(sn, p[:host_node], :cpu_used, get_prop(sn, p[:host_node], :cpu_used) - p[:cpu])
        set_prop!(sn, p[:host_node], :occupied, 0)
    end
    # free the BW
    for e in edges(vnr)
        p = props(vnr, e)
        for (key, value) in p[:vlink]
            set_prop!(sn, Edge(key), :BW_used, get_prop(sn, Edge(key), :BW_used) - value)
        end
    end
end

function get_partial_key(vnr, most_expensive)
    key = ""
    for i in 1:most_expensive-1
        key *= ","
        key *= string(get_prop(vnr, i, :host_node))
    end
    return key
end

function get_sequence(vnr)
    seq = []
    for i in 1:nv(vnr)
        push!(seq, get_prop(vnr, i, :host_node))
    end
    return seq
end

function get_legal_partial(sn::MetaGraph{Int64, Float64}, vnr::MetaGraph{Int64, Float64}, sequence, most_expensive::Int64, solver, order_links)
    stop = get_prop(vnr, most_expensive, :host_node)
    key = ""
    curr_node = 1
    for i in sequence
        if i == stop
            break
        end
        sn, vnr, curr_node, reward, done = play(sn, vnr, curr_node, i, solver, order_links)
        key *= ","
        key *= string(i)
    end
    return get_legal_moves(sn, vnr, curr_node)
end

function clear_bw!(vnr)
    for e in edges(vnr)
        set_prop!(vnr, e, :vlink, Dict{Tuple{Int64, Int64}, Float64}())
    end
end


function playtest(sn, vnr, sequence, solver, order_links)
    curr_node = 1
    reward = 0
    for action in sequence
        sn, vnr, curr_node, reward, done = play(sn, vnr, curr_node, action, solver, order_links)
        if done
            break
        end
    end
    return reward
end

function clear_bw_node!(vnr, node)
    for n in neighbors(vnr, node)
        set_prop!(vnr, node, n, :vlink, Dict{Tuple{Int64, Int64}, Float64}())
    end
end

function use_bw!(sn, vnr)
    for e in edges(vnr)
        vlink = get_prop(vnr, e, :vlink)
        for (link, bw) in vlink
            set_prop!(sn, link[1], link[2], :BW_used, get_prop(sn, link[1], link[2], :BW_used) + bw)
        end
    end
end

function place_adjacent_links!(sn, vnr, most_expensive)
    for n in neighbors(vnr, most_expensive)
        src_host = get_prop(vnr, n, :host_node)
        dst_host = get_prop(vnr, most_expensive, :host_node)

        p = shortest_path_bfs(sn, src_host, dst_host, get_prop(vnr, n, most_expensive, :BW))
        if(isempty(p))
            # return empty graphs just to make sure they are never used afterwards
            return false
        end

        for j = 1:length(p)-1
            u = p[j]
            v = p[j+1]
            set_prop!(sn, u, v, :BW_used, get_prop(sn, u, v, :BW_used) + get_prop(vnr, n, most_expensive, :BW))
        end
        d::Dict{Tuple{Int64, Int64}, Float64} = path_to_dict(p, get_prop(vnr, n, most_expensive, :BW))
        set_prop!(vnr, n, most_expensive, :vlink, d)
        return true
    end
end

function free_bw!(sn, u, v, to_remove)
    bw = get_prop(sn, u, v, :BW_used)
    set_prop!(sn, u, v, :BW_used, bw - to_remove)
end

function free_bw_used_by_node!(sn, vnr, node)
    for n in neighbors(vnr, node)
        for (l, bw) in get_prop(vnr, n, node, :vlink)
            free_bw!(sn, l[1], l[2], bw)
        end
    end
end

function clear_vlink_of_node!(vnr, node)
    for n in neighbors(vnr, node)
        set_prop!(vnr, n, node, :vlink, Dict())
    end
end

function clear_cpu_used_by_node!(sn, node, cpu)
    set_prop!(sn, node, :cpu_used, get_prop(sn, node, :cpu_used) - cpu)
    set_prop!(sn, node, :occupied, 0)
end

function get_legals(sn, cpu)
    legals = []
    for n in 1:nv(sn)
        if get_prop(sn, n, :occupied) == 0 && get_prop(sn, n, :cpu_max) - get_prop(sn, n, :cpu_used) >= cpu
            push!(legals, n)
        end
    end
    return legals
end

function place_node!(sn, vnr, move, vnode)
    prev_cpu = get_prop(sn, move, :cpu_used)
    cpu = get_prop(vnr, vnode, :cpu)
    set_prop!(sn, move, :cpu_used, prev_cpu + cpu)
    set_prop!(sn, move, :occupied, 1)

    set_prop!(vnr, vnode, :host_node, move)
end

function place_links!(sn, vnr, move, vnode)
    sn_cpy = copy_graph(sn)
    vnr_cpy = copy_graph(vnr)
    for n in neighbors(vnr, vnode)
        bw = get_prop(vnr, vnode, n, :BW)
        
        u = get_prop(vnr, n, :host_node)
        p = shortest_path_bfs(sn, move, u, bw)
        

        # if failure, revert to original graph
        if isempty(p)
            vnr = vnr_cpy
            sn = sn_cpy
            return false
        end

        for j = 1:length(p)-1
            u = p[j]
            v = p[j+1]
            set_prop!(sn, u, v, :BW_used, get_prop(sn, u, v, :BW_used) + bw)
        end
        d::Dict{Tuple{Int64, Int64}, Float64} = path_to_dict(p, bw)
        set_prop!(vnr, n, vnode, :vlink, d)
    end

    return true
end

function reorder_legals(legals, distances, vnr, most_expensive_vnode, K)
    dists = []
    for phys_node in legals
        d = 0
        for i in neighbors(vnr, most_expensive_vnode)
            host = get_prop(vnr, i, :host_node)
            d += distances[host][phys_node]
        end
        push!(dists, d)
    end
    new_legs = []
    for i in sortperm(dists)[1:min(K, length(dists))]
        push!(new_legs, legals[i])
    end
    return new_legs
end

function refine_solution(sn::MetaGraph{Int64, Float64}, vnr::MetaGraph{Int64, Float64}, sequence, reward, solver, order_links, number_its, num_moves_refine, distances)
    sn_virgin = copy_graph(sn)
    vnr_virgin = copy_graph(vnr)

    sn = copy_graph(sn)
    vnr = copy_graph(vnr)

    clear_bw!(vnr)

    best_reward = playtest(sn, vnr, sequence, solver, order_links)
    best_sn = copy_graph(sn)
    best_vnr = copy_graph(vnr)
    best_sequence = copy(sequence)
    
    l = ReentrantLock()
    l_counter = ReentrantLock()
    counter_solves = 0

    for x in 1:number_its
        previous_sn = copy_graph(best_sn)
        previous_vnr = copy_graph(best_vnr)
        previous_r = best_reward
        most_expensive_vnode = most_expensive_node(vnr, [])
        legals = get_legals(sn, get_prop(vnr, most_expensive_vnode, :cpu)) 
        if !isempty(legals)
            legals = reorder_legals(legals, distances, vnr, most_expensive_vnode, num_moves_refine)
        end
        Threads.@threads for move in legals
            sn_th = copy_graph(sn)
            vnr_th = copy_graph(vnr)
            free_bw_used_by_node!(sn_th, vnr_th, most_expensive_vnode)
            clear_vlink_of_node!(vnr_th, most_expensive_vnode)
            clear_cpu_used_by_node!(sn_th, get_prop(vnr_th, most_expensive_vnode, :host_node), get_prop(vnr_th, most_expensive_vnode, :cpu))
            
            place_node!(sn_th, vnr_th, move, most_expensive_vnode)
            success = place_links!(sn_th, vnr_th, move, most_expensive_vnode)

            lock(l_counter) do
                counter_solves += length(neighbors(vnr_th, most_expensive_vnode))
            end
            r = calculateReward(sn_th, vnr_th, success)
            lock(l) do
                if r > best_reward
                    best_sn = copy_graph(sn_th)
                    best_vnr = copy_graph(vnr_th)
                    best_sequence[most_expensive_vnode] = move
                    best_reward = r
                end
            end
            """
            if r != 0
                check_bounds_are_respected(sn_th)
                check_each_vn_uses_different_node(vnr_th)
                check_each_vn_uses_resource_amount(sn_th, sn_virgin, vnr_th)
                check_each_vl_uses_resource_amount(sn_th, sn_virgin, vnr_th) 
            end
            """
        end
        sn = copy_graph(best_sn)
        vnr = copy_graph(best_vnr)
        if previous_r == best_reward
            break
        end
    end
    return best_sn, best_vnr, best_sequence, best_reward, counter_solves
end
