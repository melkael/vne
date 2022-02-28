####################################################################
################## USING PATH-SPLITTING (LP) #######################
####################################################################

const GRB_ENV = Gurobi.Env()

function add_inf_edge!(g, a, b)
    add_edge!(g, a, b)
    set_prop!(g, a, b, :BW_max, 9999999)
    set_prop!(g, a, b, :BW_used, 0)
    set_prop!(g, a, b, :weight, 0)
end


function oriented(sn::MetaGraph{Int64, Float64})
    # 2 nodes per edge are required in order to make the gadget from Ahuja et al.
    g = MetaDiGraph(nv(sn) + 2 * ne(sn))
    for i in 1:nv(sn)
        p = props(sn, i)
        set_props!(g, i, copy(p))
    end

    k = nv(sn)+1

    for e in edges(sn)
        p = props(sn, e)
        p[:weight] = 1
        i = e.src
        j = e.dst

        i_prime = k
        j_prime = k+1

        add_inf_edge!(g, i, i_prime)
        add_inf_edge!(g, j, i_prime)
        add_inf_edge!(g, j_prime, i)
        add_inf_edge!(g, j_prime, j)

        add_edge!(g, i_prime, j_prime)
        set_props!(g, i_prime, j_prime, copy(p))

        add_edge!(g, i, j)
        

        set_prop!(g, i, j, :BW_max, 0)
        set_prop!(g, i, j, :BW_used, 0)
        set_prop!(sn, i, j, :edge_good, (i_prime, j_prime))
        set_prop!(g, i, j, :weight, 0)
        k += 2
    end
    return g
end

function out(edges_var, n, com, sn)
    ret = VariableRef[]
    for e in edges(sn)
        if e.src == n
            push!(ret, edges_var[e.src, e.dst, com])
        end
    end
    return ret 
end

function in(edges_var, n, com, sn)
    ret = VariableRef[]
    for e in edges(sn)
        if e.dst == n
            push!(ret, edges_var[e.src, e.dst, com])
        end
    end
    return ret 
end

function solve(sn::MetaDiGraph{Int64, Float64}, vnr::MetaGraph{Int64, Float64})
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "LogToConsole", 0)
    edges_var = Dict()
    commodities = Dict()
    for (idx, e) in enumerate(edges(vnr))
        commodities[idx] = e
    end
    
    for e in edges(sn)
        for i in keys(commodities)
            edges_var[(e.src, e.dst, i)] = @variable(model, base_name=string((e.src, e.dst, i)), lower_bound=0)
            # bound constraint
            #@constraint(model, edges_var[(e.src, e.dst, i)] >= 0)
        end
    end

    # max capacity constraints
    for e in edges(sn)
        cons = []
        for i in keys(commodities)
            push!(cons, edges_var[(e.src, e.dst, i)])
        end
        @constraint(model, sum(cons) <= (get_prop(sn, e, :BW_max) - get_prop(sn, e, :BW_used)))
    end

    # flow conservation constraints
    for n in 1:nv(sn)
        for (com, edge) in commodities
            src_host = get_prop(vnr, edge.src, :host_node)
            dst_host = get_prop(vnr, edge.dst, :host_node)
            if src_host == n
                dem = get_prop(vnr, edge, :BW)
            elseif dst_host == n
                dem = -get_prop(vnr, edge, :BW)
            else 
                dem = 0
            end

            @constraint(model, sum(out(edges_var, n, com, sn)) - sum(in(edges_var, n, com, sn)) == dem)
        end
    end

    vals_obj = []
    for (edge, var) in edges_var
        push!(vals_obj, get_prop(sn, edge[1], edge[2], :weight) * var)
    end
    @objective(model, Min, sum(vals_obj))
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        return model, edges_var, commodities
    else
        return nothing, nothing, nothing
    end
end

function embed_solution!(edges_var, com, sn, vnr)
    for e in edges(vnr)
        set_prop!(vnr, e, :vlink, Dict{Tuple{Int64, Int64}, Float64}())
    end
    for e in edges(sn)
        for (c, e_vnr) in com
            e_sol = get_prop(sn, e, :edge_good)
            if value(edges_var[(e_sol[1], e_sol[2], c)]) > 0
                p = props(sn, e)
                p[:BW_used] += value(edges_var[(e_sol[1], e_sol[2], c)])
                p = props(vnr, e_vnr)
                p[:vlink][(e.src, e.dst)] = value(edges_var[(e_sol[1], e_sol[2], c)])
            end
        end
    end
    return sn, vnr, true
end

function place_links_split(sn::MetaGraph{Int64, Float64}, vnr::MetaGraph{Int64, Float64}, order_links::Bool)
    #sn_oriented::MetaDiGraph{Int64, Float64} = oriented(sn)
    sn_oriented::MetaDiGraph{Int64, Float64} = oriented(sn)
    m, edges_var, com = solve(sn_oriented, vnr)
    if m == nothing
        return MetaGraph{Int64, Float64}(), MetaGraph{Int64, Float64}(), false
    end

    embed_solution!(edges_var, com, sn, vnr)

    return sn, vnr, true
end
