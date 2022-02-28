function playout(sn, vnr, policy, distances, solver::Function, order_links::Bool, max_bw_sn, max_bw_vnr, sum_bw_sn, sum_bw_vnr)
    sequence = []
    done::Bool = false
    curr_node = 1
    # each key identifies a state uniquely in our dictionnary
    key = ""
    reward::Float64 = 0
    while true
        if done
            return reward, sequence
        end
        if !haskey(policy, key)
            default_weight!(policy, key, distances)
        end
        z::Float64 = 0.0
        weights_vect = Float64[]
        actions_vect = Int64[]
        plm = false
        """
        for m in get_legal_moves(sn, vnr, curr_node)
            for i in sequence
                @assert m != i
            end
        end
        """
        for m in get_legal_moves(sn, vnr, curr_node, max_bw_sn, max_bw_vnr, sum_bw_sn, sum_bw_vnr)
            key_child = key * "," * string(m)
            if !haskey(policy, key_child)
                default_weight!(policy, key_child, distances)
            end
            ex::Float64 = exp(policy[key_child])
            z += ex
            push!(actions_vect, m)
            push!(weights_vect, ex)
        end

        # println(weights_vect)
        # if there is no more move here, it means the placement is failed since it is unfinishable
        if  length(actions_vect) == 0
            return 0, []
        end
        # sample actions according to Gibbs sampling
        #println(weights_vect)
        #println(weights_vect ./ z)
        action = sample(actions_vect, Weights(weights_vect ./ z), 1)[1]
        sn, vnr, curr_node, reward, done = play_afbd(sn, vnr, curr_node, action, solver, order_links)   
        push!(sequence, action)
        key *= ","
        key *= string(action)
    end
end


function adapt(policy, sequence, sn, vnr, distances, solver::Function, order_links::Bool, max_bw_sn, max_bw_vnr, sum_bw_sn, sum_bw_vnr)
    polp = copy(policy)
    key = ""
    alpha::Float64 = 1
    curr_node = 1
     for action in sequence
        key *= ","
        key *= string(action)
        if !haskey(polp, key)
            default_weight!(polp, key, distances)
        end
        polp[key] += alpha
        z::Float64 = 0
        moves = get_legal_moves(sn, vnr, curr_node, max_bw_sn, max_bw_vnr, sum_bw_sn, sum_bw_vnr)
        for m in moves
            key_child = key * "," * string(m)
            if !haskey(policy, key_child)
                default_weight!(policy, key_child, distances)
            end
            z += exp(policy[key_child])
        end
        for m in moves
            key_child = key * "," * string(m)
            if !haskey(polp, key_child)
                default_weight!(polp, key_child, distances)
            end
            polp[key_child] -= alpha * (exp(policy[key_child]) / z)
        end
        # we avoid doing the last call to place links as it is expensive and we do not need the reward here
        # legal nodes are all that matters
        if curr_node < nv(vnr)
            sn, vnr, curr_node, reward::Float64, done::Bool = play_afbd(sn, vnr, curr_node, action, solver, order_links)
        end
    end
    return polp
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


function median_bw(vnr)
    a = []
    for e in edges(vnr)
        push!(a, get_prop(vnr, e, :BW))
    end
    return median(a)
end 
