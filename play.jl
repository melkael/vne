function play(sn::MetaGraph{Int64, Float64},
    vnr::MetaGraph{Int64, Float64},
    curr_node::Int64,
    action::Int64,
    solver::Function,
    order_links::Bool)

    cpu_used::Int64 = get_prop(sn, action, :cpu_used)
    cpu::Int64 = get_prop(vnr, curr_node, :cpu)

    # if the chosen node (action) has not enough cpu, failure
    if get_prop(sn, action, :cpu_max) - cpu_used < cpu || get_prop(sn, action, :occupied) == 1
        return sn, vnr, curr_node, 0, true
    else
        # else update sn & vnr
        set_prop!(sn, action, :cpu_used, cpu_used + cpu)
        set_prop!(sn, action, :occupied, 1)
        set_prop!(vnr, curr_node, :host_node, action)
    end
    done::Bool = false
    success = false

    if curr_node == nv(vnr)
        sn, vnr, success = solver(sn, vnr, order_links)
        # the simulation ends here, even if link placement fails
        done = true
    end

    reward::Float64 = calculateReward(sn, vnr, success)

    return sn, vnr, min(curr_node+1, nv(vnr)), reward, done
end

function play_mcts(sn::MetaGraph{Int64, Float64},
    vnr::MetaGraph{Int64, Float64},
    curr_node::Int64,
    action::Int64,
    solver::Function,
    order_links::Bool)

    cpu_used::Int64 = get_prop(sn, action, :cpu_used)
    cpu::Int64 = get_prop(vnr, curr_node, :cpu)

    # if the chosen node (action) has not enough cpu, failure
    if get_prop(sn, action, :cpu_max) - cpu_used < cpu || get_prop(sn, action, :occupied) == 1
        return sn, vnr, curr_node, 0, true
    else
        # else update sn & vnr
        set_prop!(sn, action, :cpu_used, cpu_used + cpu)
        set_prop!(sn, action, :occupied, 1)
        set_prop!(vnr, curr_node, :host_node, action)
    end
    done::Bool = false
    success = false

    if curr_node == nv(vnr)
        sn, vnr, success = solver(sn, vnr, order_links)
        # the simulation ends here, even if link placement fails
        done = true
    end

    reward::Float64 = calculateRewardMCTS(sn, vnr, success)

    return sn, vnr, min(curr_node+1, nv(vnr)), reward, done
end

function play_afbd(sn::MetaGraph{Int64, Float64},
    vnr::MetaGraph{Int64, Float64},
    curr_node::Int64,
    action::Int64,
    solver::Function,
    order_links::Bool)

    cpu_used::Int64 = get_prop(sn, action, :cpu_used)
    cpu::Int64 = get_prop(vnr, curr_node, :cpu)

    # if the chosen node (action) has not enough cpu, failure
    if get_prop(sn, action, :cpu_max) - cpu_used < cpu || get_prop(sn, action, :occupied) == 1
        return sn, vnr, curr_node, 0, true
    else
        # else update sn & vnr
        set_prop!(sn, action, :cpu_used, cpu_used + cpu)
        set_prop!(sn, action, :occupied, 1)
        set_prop!(vnr, curr_node, :host_node, action)
    end
    done::Bool = false
    success = false

    if curr_node == nv(vnr)
        sn, vnr, success = solver(sn, vnr, order_links)
        # the simulation ends here, even if link placement fails
        done = true
    end

    reward::Float64 = calculateRewardAFBD(sn, vnr, success)

    return sn, vnr, min(curr_node+1, nv(vnr)), reward, done
end