# PostProcessing.jl
#
# Helper functions to post-process results, and save files
# ========================================================



# ===========================================================
# Likelihood Plot
# ===========================================================
function likelihood_plot(trace_history; filename="likelihood_trace")
    scores = [get_score(t) for t in trace_history]
    
    p = plot(scores, 
        label="Log Probability", 
        xlabel="Iteration", 
        ylabel="Log Likelihood",
        #title="Inference Convergence",
        color=:blue, alpha=0.7, lw=1.5,
        grid=true, #legend=:bottomright
        legend=false
    )

    save_path = "$(OUTPUT_DIR)/$(filename).png"
    savefig(p, save_path)
    println("Saved Likelihood Plot: $save_path")
    return p
end

# ===========================================================
# MAP (Maximum A Posteriori) Plot
# ===========================================================
function map_plot(trace_history, gt_bitmap; 
                  method::Symbol=:layer, 
                  spp=OUTPUT_SPP, 
                  img_size=OUTPUT_IMG_SIZE,
                  show_target::Bool=false, 
                  filename="map_comparison")

    # 1. Find best trace
    scores = get_score.(trace_history)
    best_idx = argmax(scores)
    best_trace = trace_history[best_idx]
    best_score = scores[best_idx]

    # 2. Render MAP Estimate
    if method == :structure
        (map_bitmap, _) = view_structure_trace(best_trace, spp=spp, img_size=img_size)
    else
        (map_bitmap, _) = view_layer_trace(best_trace, spp=spp, img_size=img_size)
    end

    # 3. Convert to Julia Images
    img_map = py_bitmap_to_img(map_bitmap)

    # 4. Plotting Logic
    if show_target
        img_gt = py_bitmap_to_img(gt_bitmap)
        
        p1 = plot(img_gt, title="Ground Truth", axis=nothing, border=:none, aspect_ratio=:equal)
        p2 = plot(img_map, title="MAP Estimate", axis=nothing, border=:none, aspect_ratio=:equal)
        
        # Side-by-side layout
        final_plot = plot(p1, p2, layout=(1, 2), size=(img_size*2, img_size + 50))
    else
        # Single plot
        final_plot = plot(img_map, title="MAP Estimate", axis=nothing, border=:none, aspect_ratio=:equal, size=(img_size, img_size + 50))
    end
    
    save_path = "$(OUTPUT_DIR)/$(filename).png"
    savefig(final_plot, save_path)
    println("Saved MAP Plot: $save_path")
    return final_plot
end


# ================================
# Save ground truth as image
# ================================

function save_ground_truth_img(bitmap, filename="targets/target")
    # 1. Convert Mitsuba Bitmap -> Julia Image
    img = py_bitmap_to_img(bitmap)
    
    # 2. Save
    base_name = splitext(filename)[1]
    save_name = "$(OUTPUT_DIR)/$(base_name).png"
    
    save(save_name, img)
    println("Saved Ground Truth Image: $save_name")
end


# ================================
# Create gif from traces
# ================================

function generate_gif(trace_history; 
                      method::Symbol=:layer, 
                      #two_views::Bool=false, 
                      show_plot::Bool=true,  
                      filename="lego_inference", 
                      step=TRACE_STEP, 
                      spp=OUTPUT_SPP, 
                      fps=15, 
                      inference_size=IMAGE_SIZE, 
                      output_size=OUTPUT_SIZE) 
    
    println("Generating GIF ($method, Plot: $show_plot)...") 
    
    base_name = splitext(filename)[1]
    gif_name = "$(OUTPUT_DIR)/$(base_name).gif"

    # compute likelihood score
    all_scores = Vector{Float64}()
    y_lims = (0.0, 1.0) # Default placeholder

    if show_plot
        all_scores = [get_score(t) for t in trace_history]
        min_s, max_s = minimum(all_scores), maximum(all_scores)
        padding = (max_s - min_s) * 0.05
        y_lims = (min_s - padding, max_s + padding) 
    end

    img_w = output_size
    
    # Plot width (only exists if show_plot is true)
    plot_w = show_plot ? output_size : 0
    
    # Total Canvas Size
    canvas_w = img_w + plot_w
    canvas_h = output_size # Removed the +50 to fix height alignment

    # Calculate Width Ratios
    ratio_img = img_w / canvas_w
    ratio_plot = plot_w / canvas_w

    frames = []
    total_traces = length(trace_history)

    for i in 1:step:total_traces
        trace = trace_history[i]
        
        # --- A. Render LEGO Image ---
        if method == :structure
            (bitmap, _) = view_structure_trace(trace, spp=spp, img_size=output_size)
        elseif method == :layer
            (bitmap, _) = view_layer_trace(trace, spp=spp, img_size=output_size)
        else
            error("Unknown method: $method")
        end
        
        img_np = @pycall np.array(bitmap)::Array{Float32, 3}
        img_clamped = clamp.(img_np, 0f0, 1f0)
        img_julia = colorview(RGB, permutedims(img_clamped, (3, 1, 2)))

        # --- Create The Plots ---
        
        # 1. The Image Plot 
        p_img = plot(img_julia, 
                     axis=nothing, border=:none, ticks=nothing, 
                     aspect_ratio=:equal,
                     title="Iter: $i", 
                     titlefontsize=10)
        
        final_plot = nothing

        if show_plot
            curr_score = get_score(trace)

            # 2. The Score Plot
            p_track = plot(all_scores, 
                           label="", 
                           color=:gray, 
                           alpha=0.5, 
                           lw=1.5,
                           ylim=y_lims, 
                           xlabel="Iteration", 
                           ylabel="Log-Likelihood",
                           grid=false,
                           background_color_inside=colorant"white",
                           title="Score: $(round(curr_score, digits=0))",
                           titlefontsize=10,
                           guidefontsize=8,
                           tickfontsize=7,
                           # Adjust margins to fit text within the strict output_size height
                           bottom_margin=5Plots.mm,
                           top_margin=5Plots.mm,
                           left_margin=5Plots.mm,
                           right_margin=5Plots.mm)
            
            # Red dot for current frame
            scatter!(p_track, [i], [curr_score], color=:red, markersize=6, label="")

            # Combine Side-by-Side
            final_plot = plot(p_img, p_track, 
                              layout=grid(1, 2, widths=[ratio_img, ratio_plot]),
                              size=(canvas_w, canvas_h))
        else
            # Image Only
            final_plot = plot(p_img, size=(canvas_w, canvas_h))
        end

        # --- C. Save Frame ---
        temp_path = tempname() * ".png"
        savefig(final_plot, temp_path)
        push!(frames, load(temp_path))
        rm(temp_path, force=true)

        if i % 50 == 0 || i == total_traces; print("."); end
    end
    println("")
    
    # save gif
    save(gif_name, cat(frames..., dims=3), fps=fps)
    println("Saved: $gif_name")
end



# ===========================================================
# Helper: Convert PyObject (Mitsuba Bitmap) to Julia Image
# ===========================================================
function py_bitmap_to_img(bitmap)
    img_np = @pycall np.array(bitmap)::Array{Float32, 3}
    img_clamped = clamp.(img_np, 0f0, 1f0)
    # Mitsuba is usually (H, W, C), Julia Images expects (C, W, H) for construction
    # but colorview handles standard arrays well if permuted correctly.
    return colorview(RGB, permutedims(img_clamped, (3, 1, 2)))
end

# ===========================================================
# Helper: Reconstruct BuildState from Trace (No Rendering)
# ===========================================================
# This recreates the logic from view_trace but skips the heavy rendering step
# so we can compute statistics on thousands of traces quickly.
function get_trace_state(trace; method::Symbol=:layer, 
                         width=WIDTH_VAL, depth=DEPTH_VAL, max_height=MAX_HEIGHT_VAL)
    
    choices = get_choices(trace)
    
    # Setup Empty State
    initial_conns = get_initial_connections(width, depth)
    state = BuildState(
        zeros(Int, width, depth, max_height),
        Vector{BrickInstance}(),
        initial_conns
    )
    brick_types = [:b1x1, :b2x1, :b4x4]

    if method == :structure
        # --- Logic from view_structure_trace ---
        num_bricks = has_value(choices, :num_bricks) ? choices[:num_bricks] : 0
        for i in 1:num_bricks
            if !has_value(choices, :brick => i => :conn_idx); continue; end
            
            conn_idx = choices[:brick => i => :conn_idx]
            if conn_idx > length(state.connections); continue; end

            chosen_conn = collect(state.connections)[conn_idx] 
            chosen_type = brick_types[choices[:brick => i => :type]]
            chosen_orient = choices[:brick => i => :orient] * 90
            
            brick = BrickInstance(chosen_type, chosen_conn, chosen_orient)
            if is_valid_placement(state, brick); place_brick!(state, brick); end
        end

    elseif method == :layer
        # --- Logic from view_layer_trace ---
        for z in 1:max_height
            conns_at_z = get_connections_at_z(state, z)
            if !has_value(choices, :bricks_at_z => z => :num_bricks); continue; end
            
            num_bricks_at_z = choices[:bricks_at_z => z => :num_bricks]
            if isempty(conns_at_z) && num_bricks_at_z > 0; continue; end

            for i in 1:num_bricks_at_z
                if !has_value(choices, :bricks_at_z => z => i => :conn_idx); continue; end
                
                conn_idx = choices[:bricks_at_z => z => i => :conn_idx]
                if conn_idx > length(conns_at_z); continue; end

                chosen_conn = conns_at_z[conn_idx]
                chosen_type = brick_types[choices[:bricks_at_z => z => i => :type]]
                chosen_orient = choices[:bricks_at_z => z => i => :orient] * 90
                
                brick = BrickInstance(chosen_type, chosen_conn, chosen_orient)
                if is_valid_placement(state, brick)
                    place_brick!(state, brick)
                    conns_at_z = get_connections_at_z(state, z) # Update available conns
                    if isempty(conns_at_z); break; end
                end
            end
        end
    end
    
    return state
end

# ===========================================================
# Run multichains (used for expectation and varaince function below)
# ===========================================================

function run_multichain_inference(obs_map; 
                                  method::Symbol=:layer_one_view, 
                                  n_chains::Int=5, 
                                  mh_steps::Int=MH_ITERS, 
                                  layer_steps::Int=LAYER_ITERS)
    
    all_chains = []

    println("Starting Multichain Inference: $method ($n_chains chains)...")

    for c in 1:n_chains
        println("  -> Chain $c / $n_chains")
        
        # --- A. INITIALIZATION ---
        trace = nothing
        
        if method == :mh
            # 1. MH Init: Empty trace + Observation
            constraints = Gen.choicemap()
            constraints[:num_bricks] = 0 
            combined = merge(obs_map, constraints)
            (trace, _) = Gen.generate(lego_structure_model, (lambda_val, WIDTH_VAL, DEPTH_VAL, MAX_HEIGHT_VAL, IMAGE_SIZE), combined)
            
        elseif method == :layer_one_view || method == :layer_two_view
            # 2. Layer Init: Empty trace + Observation
            is_two_view = (method == :layer_two_view)
            
            constraints = Gen.choicemap()
            constraints[:bricks_at_z => 1 => :num_bricks] = 0
            combined = merge(obs_map, constraints)
            (trace, _) = Gen.generate(lego_layer_model, (WIDTH_VAL, DEPTH_VAL, MAX_HEIGHT_VAL, IMAGE_SIZE, is_two_view), combined)
        end
        
        chain_history = [trace]

        # --- B. INFERENCE LOOP ---
        
        # === CASE 1: Standard MH ===
        if method == :mh
            for i in 1:mh_steps
                # Kernel 1: Add/Remove
                (trace, _) = mh(trace, select(:num_bricks))
                
                # Kernel 2: Mutate
                if trace[:num_bricks] > 0
                    idx = rand(1:trace[:num_bricks])
                    (trace, _) = mh(trace, select(:brick => idx => :conn_idx))
                    (trace, _) = mh(trace, select(:brick => idx => :type))
                    (trace, _) = mh(trace, select(:brick => idx => :orient))
                end
                push!(chain_history, trace)
            end

        # === CASE 2: Layered Inference (Both 1-View and 2-View) ===
        elseif method == :layer_one_view || method == :layer_two_view
            
            # Helper to check connections based on current view mode
            # (We use view_layer_trace which returns the state without full rendering)
            
            for z_stage in 1:MAX_HEIGHT_VAL
                # Get current state to find valid connections
                (_, current_state) = view_layer_trace(trace, spp=1)
                valid_conns = get_connections_at_z(current_state, z_stage)
                
                if isempty(valid_conns)
                    push!(chain_history, trace)
                    continue 
                end
                
                # Run MCMC for this layer
                for i in 1:layer_steps
                    # Smart Scheduling: 80% current layer, 20% backtrack
                    z = z_stage
                    if z_stage > 1 && rand() < 0.2
                        z = rand(1:z_stage) 
                    end
                    
                    # Kernel 1: Add/Remove at z
                    (trace, _) = mh(trace, select(:bricks_at_z => z => :num_bricks))
                    
                    # Kernel 2: Mutate at z
                    choices = get_choices(trace)
                    n_at_z = has_value(choices, :bricks_at_z => z => :num_bricks) ? choices[:bricks_at_z => z => :num_bricks] : 0
                    
                    if n_at_z > 0
                        b_idx = rand(1:n_at_z)
                        (trace, _) = mh(trace, select(:bricks_at_z => z => b_idx => :conn_idx))
                        (trace, _) = mh(trace, select(:bricks_at_z => z => b_idx => :type))
                        (trace, _) = mh(trace, select(:bricks_at_z => z => b_idx => :orient))
                    end
                    push!(chain_history, trace)
                end
            end
        end
        
        # Save this chain
        push!(all_chains, chain_history)
    end
    
    return all_chains
end

# ===========================================================
# Expectation & Variance (Pixel-wise)
# ===========================================================
function expectation_variance_plot(traces_input; 
                                   method::Symbol=:structure, 
                                   burn_in=0, 
                                   step=1,
                                   spp=16, 
                                   img_size=OUTPUT_IMG_SIZE,
                                   filename="visual_expectation")
    
    # --- 1. Handle Input (Single Chain vs Multi-Chain) ---
    # We standardize everything into a list of chains (Vector of Vectors)
    chains = []
    if traces_input[1] isa Vector
        chains = traces_input # It's already a list of chains
        println("Processing $(length(chains)) chains...")
    else
        chains = [traces_input] # Wrap single chain in a list
        println("Processing single chain...")
    end

    # --- 2. Flatten and Apply Burn-in/Thinning ---
    valid_traces = []
    for chain in chains
        # Apply burn-in and step to EACH chain separately
        if length(chain) > burn_in
            filtered = chain[(burn_in+1):step:end]
            append!(valid_traces, filtered)
        end
    end

    n_samples = length(valid_traces)
    if n_samples == 0; error("No traces left after filtering! Check burn_in/step."); end
    println("Total pooled samples: $n_samples")

    # --- 3. Initialize Accumulators ---
    # Render the first sample to get dimensions
    if method == :structure
        (b_init, _) = view_structure_trace(valid_traces[1], spp=spp, img_size=img_size)
    else
        (b_init, _) = view_layer_trace(valid_traces[1], spp=spp, img_size=img_size)
    end
    
    img_init = @pycall np.array(b_init)::Array{Float64, 3}
    H, W, C = size(img_init)
    sum_img = zeros(Float64, H, W, C)
    sum_sq_img = zeros(Float64, H, W, C)

    # --- 4. Accumulate (Render Loop) ---
    p = Progress(n_samples, 1, "Rendering pooled samples...")
    for t in valid_traces
        if method == :structure
            (bitmap, _) = view_structure_trace(t, spp=spp, img_size=img_size)
        else
            (bitmap, _) = view_layer_trace(t, spp=spp, img_size=img_size)
        end
        curr_img = @pycall np.array(bitmap)::Array{Float64, 3}
        
        # Accumulate
        sum_img .+= curr_img
        sum_sq_img .+= (curr_img .^ 2)
        next!(p)
    end

    # --- 5. Compute Statistics ---
    mean_img_arr = sum_img ./ n_samples
    # Variance = E[X^2] - (E[X])^2
    var_img_arr = (sum_sq_img ./ n_samples) .- (mean_img_arr .^ 2)
    var_img_arr = clamp.(var_img_arr, 0.0, Inf) # Avoid negative zero errors
    std_img_arr = sqrt.(var_img_arr)
    
    # Flatten variance to intensity for plotting (take max across RGB channels)
    std_intensity = dropdims(maximum(std_img_arr, dims=3), dims=3)

    # --- 6. Plotting ---
    
    # A. Expectation Image
    mean_clamped = clamp.(Float32.(mean_img_arr), 0f0, 1f0)
    img_mean_julia = colorview(RGB, permutedims(mean_clamped, (3, 1, 2)))
    p1 = plot(img_mean_julia, 
              title="Expectation", 
              axis=nothing, border=:none, aspect_ratio=:equal)

    # B. Variance Image
    min_v, max_v = extrema(std_intensity)
    denom = (max_v - min_v) > 1e-6 ? (max_v - min_v) : 1.0 
    
    # Gradient: White -> Gold -> Orange -> Red -> DarkRed
    grad = cgrad([RGB(0.97, 0.97, 0.97), :gold, :orange, :red, :darkred])
    var_colors = [RGB(grad[(v - min_v) / denom]) for v in std_intensity]
    img_var_julia = colorview(RGB, var_colors)

    p2 = plot(img_var_julia, 
              title="Std Deviation", 
              axis=nothing, border=:none, aspect_ratio=:equal)

    # Layout
    final_plot = plot(p1, p2, layout=(1, 2), size=(img_size*2, img_size + 50))

    save_path = "$(OUTPUT_DIR)/$(filename).png"
    savefig(final_plot, save_path)
    println("Saved Visual Expectation Plot: $save_path")
    
    return final_plot
end

# ===========================================================
# Function to compare likelihoods between traces 
#     (can input few structures in data_dict)
# ===========================================================

function compare_likelihoods(data_dict; filename="comparison_likelihood")
    # data_dict format: Dict("StructureName" => (traces_mh, traces_layer))
    
    p = plot(xlabel="Iteration", ylabel="Log Likelihood", 
             title="", legend=:bottomright, 
             grid=true, background_color_inside=colorant"white")

    # Define a color cycle for different structures
    colors = [:blue, :red, :green, :purple]
    c_idx = 1

    for (struct_name, (t_mh, t_lay)) in data_dict
        curr_color = colors[c_idx % length(colors)]
        
        # 1. Plot MH (Dashed)
        scores_mh = get_score.(t_mh)
        plot!(p, scores_mh, 
              label="$struct_name (MH)", 
              color=curr_color, 
              linestyle=:dash, 
              lw=1.5, alpha=0.8)

        # 2. Plot Layer (Solid)
        scores_lay = get_score.(t_lay)
        plot!(p, scores_lay, 
              label="$struct_name (Layer)", 
              color=curr_color, 
              linestyle=:solid, 
              lw=2.0, alpha=0.8)
        
        c_idx += 1
    end

    save_path = "$(OUTPUT_DIR)/$(filename).png"
    savefig(p, save_path)
    println("Saved Comparison Plot: $save_path")
    return p
end