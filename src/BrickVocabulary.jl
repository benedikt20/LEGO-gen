# BrickVocabulary.jl
#
#
# Visualization of the brick vocabulary to be used
# ===========================================================

function plot_vocabulary(spp=OUTPUT_SPP,
                         img_size=OUTPUT_IMG_SIZE,
                         zoom=0.045, # Adjust zoom
                         filename="brick_vocabulary")
    
    println("Rendering Vocabulary Showcase (Scale: $zoom)...")

    # Define the specific grid dimensions needed to perfectly center each type
    # (Width, Depth) of the grid
    vocab_configs = [
        (:b1x1, "1x1 Brick", (1, 1)), # 1x1 grid centers a 1x1 brick
        (:b2x1, "2x1 Brick", (2, 1)), # 2x1 grid centers a 2x1 brick
        (:b4x4, "4x4 Plate", (4, 4))  # 4x4 grid centers a 4x4 plate
    ]
    
    plts = []

    # Shift the world slightly down so the brick isn't sitting on the bottom edge
    # [x, y, z] -> Y is vertical in Mitsuba
    vocab_pos = [0.0, -1.5, -1.5]

    for (type, title_str, (w, d)) in vocab_configs
        
        # 1. Create a grid exactly the size of the brick
        state = BuildState(
            zeros(Int, w, d, 5),
            Vector{BrickInstance}(),
            get_initial_connections(w, d)
        )

        # 2. Place the brick at (1, 1, 1)
        brick = BrickInstance(type, (1, 1, 1), 0)
        push!(state.bricks, brick)

        # 3. Render
        scene_d = build_scene_dict(state, vocab_pos, FIXED_GLOBAL_ROT_Y, zoom, img_size)
        
        modelparams = ModelParams(scene_d=scene_d, spp=spp)
        bitmap = render_bitmap(modelparams)

        img_np = @pycall np.array(bitmap)::Array{Float32, 3}
        img_clamped = clamp.(img_np, 0f0, 1f0)
        img_julia = colorview(RGB, permutedims(img_clamped, (3, 1, 2)))

        p = plot(img_julia, 
                 title=title_str, 
                 axis=nothing, border=:none, aspect_ratio=:equal)
        push!(plts, p)
    end

    final_plot = plot(plts..., layout=(1, 3), size=(img_size*3, img_size))

    save_path = "$(OUTPUT_DIR)/$(filename).png"
    savefig(final_plot, save_path)
    println("Saved Vocabulary Plot: $save_path")
    
    return final_plot
end