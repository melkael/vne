####################################################################
############# FUNCTIONS RELATED TO PLACING LINKS ###################
####################################################################
################## USING SHORTEST PATH (BFS) #######################
####################################################################

include("utils.jl")

function parallel_shortest_path_bfs(g::MetaGraph{Int64, Float64}, start::Int64, finish::Int64, threshold::Int64)
    finish_save = finish
    Q = ThreadQueue(Int64, nv(g))
    push!(Q, start)
    visited = [Threads.Atomic{Int64}(0) for i = 1:nv(g)]
    Threads.atomic_xchg!(visited[start], 1)

    parents = [Threads.Atomic{Int64}(0) for i = 1:nv(g)]
    success = false 
    while !isempty(Q)
        node = popfirst!(Q)
        if node == finish
            success = true
            break
        end

        Threads.@threads for i in neighbors(g, node)
            p = props(g, node, i)
            if p[:BW_max] - p[:BW_used] >= threshold
                if Threads.atomic_cas!(visited[i], 0, 1) == 0
                    Threads.atomic_xchg!(parents[i], node)
                    push!(Q, i)
                end
            end
        end
    end
    if success
        sp = Int64[finish]
        while parents[finish][] != 0
            push!(sp, parents[finish][])
            finish = parents[finish][]
        end
        return sp 
    else
        return Int64[]
    end
end


# returns (new sn, new vnr, new curr_node, reward, state_is_final)
function shortest_path_bfs(g::MetaGraph{Int64, Float64}, start::Int64, finish::Int64, threshold::Int64)
    finish_save = finish
    Q = Int64[start]
    visited = fill(0, nv(g))
    visited[start] = 1
    parents = fill(0, nv(g))
    success = false
    # standard BFS (check wikipedia for the algo)
    while !isempty(Q)
        node = popfirst!(Q)
        if node == finish
            success = true
            break
        end
        
        for i in neighbors(g, node)
            p = props(g, node, i)
            # only difference: we check the bandwidths here.
            # faster than removing from the graph then performing full bfs
            if visited[i] == 0 && p[:BW_max] - p[:BW_used] >= threshold
                visited[i] = 1
                parents[i] = node
                push!(Q, i)
            end
        end
    end
    if success
        sp = Int64[finish]
        while parents[finish] != 0
            push!(sp, parents[finish])
            finish = parents[finish]
        end
        return sp 
    else
        return Int64[]
    end
end

function path_to_dict(p::Vector{Int64}, BW::Int64)
    d = Dict{Tuple{Int64, Int64}, Int64}()
     for i in 1:length(p)-1
        d[(p[i],p[i+1])] = BW
    end
    return d
end

function place_links_sp(sn::MetaGraph{Int64, Float64}, vnr::MetaGraph{Int64, Float64}, order_links::Bool)
    
    # no need to copy them, since this function is only called in play, and when link placement fails, we do not return the sn and vnr anyways
    #sn_save = copy_graph(sn)
    #vnr_save = copy_graph(vnr)
    
    ed = collect(edges(vnr))
    if order_links
        v = []
        for e in ed
            push!(v, get_prop(vnr, e, :BW))
        end
        v = sortperm(v, rev=true)
    else
        v = 1:length(ed)
    end

    for i in v
        e = ed[i]
        src_host = get_prop(vnr, e.src, :host_node)
        dst_host = get_prop(vnr, e.dst, :host_node)
        
        p = shortest_path_bfs(sn, src_host, dst_host, get_prop(vnr, e, :BW))

        if(isempty(p))
            # return empty graphs just to make sure they are never used afterwards
            return MetaGraph{Int64, Float64}(), MetaGraph{Int64, Float64}(), false
        end
        for j = 1:length(p)-1
            u = p[j]
            v = p[j+1]
            set_prop!(sn, u, v, :BW_used, get_prop(sn, u, v, :BW_used) + get_prop(vnr, e, :BW))
        end
        d::Dict{Tuple{Int64, Int64}, Float64} = path_to_dict(p, get_prop(vnr, e, :BW))
        set_prop!(vnr, e, :vlink, d)
    end

    return sn, vnr, true
end

function place_links_sp_for_node(sn::MetaGraph{Int64, Float64}, vnr::MetaGraph{Int64, Float64}, order_links::Bool, node)
    
    # no need to copy them, since this function is only called in play, and when link placement fails, we do not return the sn and vnr anyways
    #sn_save = copy_graph(sn)
    #vnr_save = copy_graph(vnr)
    
    ed = collect(edges(vnr))
    if order_links
        v = []
        for e in ed
            push!(v, get_prop(vnr, e, :BW))
        end
        v = sortperm(v, rev=true)
    else
        v = 1:length(ed)
    end

    for i in v
        e = ed[i]
        src_host = get_prop(vnr, e.src, :host_node)
        dst_host = get_prop(vnr, e.dst, :host_node)
        if e.src != node && e.dst != node
            continue
        end
        
        p = shortest_path_bfs(sn, src_host, dst_host, get_prop(vnr, e, :BW))

        if(isempty(p))
            # return empty graphs just to make sure they are never used afterwards
            return MetaGraph{Int64, Float64}(), MetaGraph{Int64, Float64}(), false
        end
        for j = 1:length(p)-1
            u = p[j]
            v = p[j+1]
            set_prop!(sn, u, v, :BW_used, get_prop(sn, u, v, :BW_used) + get_prop(vnr, e, :BW))
        end
        d::Dict{Tuple{Int64, Int64}, Float64} = path_to_dict(p, get_prop(vnr, e, :BW))
        set_prop!(vnr, e, :vlink, d)
    end

    return sn, vnr, true
end