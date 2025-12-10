# -----------------------------------------------------------
# LegoStructures.jl
#
# Simple functions to generate ground truth scenes.
# Supports optional multi-view generation via `two_views=true`.
# -----------------------------------------------------------

using Gen
using PyCall

# ===========================================================
# Scene 1: Tall Table (Four 1x1 legs + 4x4 top)
# ===========================================================

function generate_table_ground_truth(;
    width::Int=WIDTH_VAL, 
    depth::Int=DEPTH_VAL, 
    max_height::Int=MAX_HEIGHT_VAL,
    spp::Int = SPP_VAL,
    pos::Vector{Float64} = FIXED_GLOBAL_POS,
    rot_y::Float64 = FIXED_GLOBAL_ROT_Y,
    scale::Float64 = FIXED_GLOBAL_SCALE,
    img_size=IMAGE_SIZE,
    two_views::Bool=false
)
    println("Generating 'Tall Table' ground truth (Two Views: $two_views)...")
    structure_name = "table"
    
    # --- 1. Setup ---
    origin_x = (width ÷ 2) - 1 
    origin_y = (depth ÷ 2) - 1
    
    state = BuildState(
        zeros(Int, width, depth, max_height),
        Vector{BrickInstance}(),
        get_initial_connections(width, depth)
    )
    
    # --- 2. Build ---
    # Legs Layer 1
    place_brick!(state, BrickInstance(:b1x1, (origin_x, origin_y, 1), 0))
    place_brick!(state, BrickInstance(:b1x1, (origin_x + 3, origin_y, 1), 0))
    place_brick!(state, BrickInstance(:b1x1, (origin_x, origin_y + 3, 1), 0))
    place_brick!(state, BrickInstance(:b1x1, (origin_x + 3, origin_y + 3, 1), 0))
    
    # Legs Layer 2
    place_brick!(state, BrickInstance(:b1x1, (origin_x, origin_y, 4), 0))
    place_brick!(state, BrickInstance(:b1x1, (origin_x + 3, origin_y, 4), 0))
    place_brick!(state, BrickInstance(:b1x1, (origin_x, origin_y + 3, 4), 0))
    place_brick!(state, BrickInstance(:b1x1, (origin_x + 3, origin_y + 3, 4), 0))

    # Top Plate
    place_brick!(state, BrickInstance(:b4x4, (origin_x, origin_y, 7), 0))
    
    num_bricks = length(state.bricks)

    # --- 3. Render ---
    obs_image_map = Gen.choicemap()

    # A. Front
    scene_d = build_scene_dict(state, pos, rot_y, scale, img_size)
    scene = @pycall mi.load_dict(scene_d)::PyObject
    image = @pycall mi.render(scene, spp=spp)::PyObject
    obs_bitmap = @pycall mi.Bitmap(image).convert(srgb_gamma=true)::PyObject
    p = @pycall np.array(obs_bitmap)::Array{Float64, 3}
    data_front = p[:, :, 1]
    
    if two_views
        # B. Side
        scene_d_side = build_scene_dict(state, pos, rot_y + 90.0, scale, img_size)
        scene_side = @pycall mi.load_dict(scene_d_side)::PyObject
        image_side = @pycall mi.render(scene_side, spp=spp)::PyObject
        
        obs_bitmap_side = @pycall mi.Bitmap(image_side).convert(srgb_gamma=true)::PyObject
        p_side = @pycall np.array(obs_bitmap_side)::Array{Float64, 3}
        data_side = p_side[:, :, 1]

        obs_image_map[:pred_front] = data_front
        obs_image_map[:pred_side] = data_side
        
        # RETURN 4 items
        return obs_image_map, obs_bitmap, obs_bitmap_side, num_bricks, structure_name
    else
        obs_image_map[:pred] = data_front
        # RETURN 3 items
        return obs_image_map, obs_bitmap, num_bricks, structure_name
    end
end


# ===========================================================
# Scene 2: The Pyramid (3 Layers of solid bricks)
# ===========================================================

function generate_pyramid_ground_truth(;
    width::Int=WIDTH_VAL, 
    depth::Int=DEPTH_VAL, 
    max_height::Int=MAX_HEIGHT_VAL,
    spp::Int = SPP_VAL,
    pos::Vector{Float64} = FIXED_GLOBAL_POS,
    rot_y::Float64 = FIXED_GLOBAL_ROT_Y,
    scale::Float64 = FIXED_GLOBAL_SCALE,
    img_size=IMAGE_SIZE,
    two_views::Bool=false
)
    println("Generating 'Pyramid' ground truth (Two Views: $two_views)...")
    structure_name = "pyramid"
    
    # --- 1. Setup ---
    state = BuildState(
        zeros(Int, width, depth, max_height),
        Vector{BrickInstance}(),
        get_initial_connections(width, depth)
    )
    
    # Offsets to center the pyramid in an 8x8 grid
    # Grid is 1..8. 
    
    # --- 2. Build ---
    
    # Layer 1: 6x6 Base (z=1)
    # Covering x=[2..7], y=[2..7] using 2x1 bricks
    # We place them horizontally (orient=0)
    for y in 2:7
        for x in 2:2:7 # 2, 4, 6
            place_brick!(state, BrickInstance(:b2x1, (x, y, 1), 0))
        end
    end

    # Layer 2: 4x4 Mid (z=4)
    # Covering x=[3..6], y=[3..6]
    # We place them vertically (orient=90) to interlock
    for x in 3:6
        for y in 3:2:6 # 3, 5
            place_brick!(state, BrickInstance(:b2x1, (x, y, 4), 90))
        end
    end

    # Layer 3: 2x2 Top (z=7)
    # Covering x=[4..5], y=[4..5]
    # Place horizontally (orient=0)
    place_brick!(state, BrickInstance(:b2x1, (4, 4, 7), 0))
    place_brick!(state, BrickInstance(:b2x1, (4, 5, 7), 0))
    
    num_bricks = length(state.bricks) # Should be ~28

    # --- 3. Render ---
    obs_image_map = Gen.choicemap()

    # A. Front
    scene_d = build_scene_dict(state, pos, rot_y, scale, img_size)
    scene = @pycall mi.load_dict(scene_d)::PyObject
    image = @pycall mi.render(scene, spp=spp)::PyObject
    obs_bitmap = @pycall mi.Bitmap(image).convert(srgb_gamma=true)::PyObject
    p = @pycall np.array(obs_bitmap)::Array{Float64, 3}
    data_front = p[:, :, 1]
    
    if two_views
        # B. Side
        scene_d_side = build_scene_dict(state, pos, rot_y + 90.0, scale, img_size)
        scene_side = @pycall mi.load_dict(scene_d_side)::PyObject
        image_side = @pycall mi.render(scene_side, spp=spp)::PyObject
        
        obs_bitmap_side = @pycall mi.Bitmap(image_side).convert(srgb_gamma=true)::PyObject
        p_side = @pycall np.array(obs_bitmap_side)::Array{Float64, 3}
        data_side = p_side[:, :, 1]

        obs_image_map[:pred_front] = data_front
        obs_image_map[:pred_side] = data_side
        
        return obs_image_map, obs_bitmap, obs_bitmap_side, num_bricks, structure_name
    else
        obs_image_map[:pred] = data_front
        return obs_image_map, obs_bitmap, num_bricks, structure_name
    end
end


# ===========================================================
# Scene 3: Bus Stop Shelter (Two 1x1x3 stacks + 4x4 top)
# ===========================================================

function generate_shelter_ground_truth(;
    width::Int=WIDTH_VAL, 
    depth::Int=DEPTH_VAL, 
    max_height::Int=MAX_HEIGHT_VAL,
    spp::Int = SPP_VAL,
    pos::Vector{Float64} = FIXED_GLOBAL_POS,
    rot_y::Float64 = FIXED_GLOBAL_ROT_Y,
    scale::Float64 = FIXED_GLOBAL_SCALE,
    img_size=IMAGE_SIZE,
    two_views::Bool=false
)
    println("Generating 'Bus Stop Shelter' ground truth (Two Views: $two_views)...")
    structure_name = "shelter"
    
    # --- 1. Setup ---
    # Center the 4x4 footprint
    origin_x = (width ÷ 2) - 1 
    origin_y = (depth ÷ 2) - 1
    
    state = BuildState(
        zeros(Int, width, depth, max_height),
        Vector{BrickInstance}(),
        get_initial_connections(width, depth)
    )
    
    # --- 2. Build ---
    
    # -- Left Pillar (Height 3) --
    place_brick!(state, BrickInstance(:b1x1, (origin_x, origin_y, 1), 0))
    place_brick!(state, BrickInstance(:b1x1, (origin_x, origin_y, 4), 0))
    place_brick!(state, BrickInstance(:b1x1, (origin_x, origin_y, 7), 0))

    # -- Right Pillar (Height 3) --
    # Placed at x+3 to align with the opposite corner of the 4x4 plate
    place_brick!(state, BrickInstance(:b1x1, (origin_x + 3, origin_y, 1), 0))
    place_brick!(state, BrickInstance(:b1x1, (origin_x + 3, origin_y, 4), 0))
    place_brick!(state, BrickInstance(:b1x1, (origin_x + 3, origin_y, 7), 0))

    # -- Roof (Top Plate) --
    # Placed at z=10 (sitting on top of the z=7 bricks)
    place_brick!(state, BrickInstance(:b4x4, (origin_x, origin_y, 10), 0))
    
    num_bricks = length(state.bricks)

    # --- 3. Render ---
    obs_image_map = Gen.choicemap()

    # A. Front
    scene_d = build_scene_dict(state, pos, rot_y, scale, img_size)
    scene = @pycall mi.load_dict(scene_d)::PyObject
    image = @pycall mi.render(scene, spp=spp)::PyObject
    obs_bitmap = @pycall mi.Bitmap(image).convert(srgb_gamma=true)::PyObject
    p = @pycall np.array(obs_bitmap)::Array{Float64, 3}
    data_front = p[:, :, 1]
    
    if two_views
        # B. Side
        scene_d_side = build_scene_dict(state, pos, rot_y + 90.0, scale, img_size)
        scene_side = @pycall mi.load_dict(scene_d_side)::PyObject
        image_side = @pycall mi.render(scene_side, spp=spp)::PyObject
        
        obs_bitmap_side = @pycall mi.Bitmap(image_side).convert(srgb_gamma=true)::PyObject
        p_side = @pycall np.array(obs_bitmap_side)::Array{Float64, 3}
        data_side = p_side[:, :, 1]

        obs_image_map[:pred_front] = data_front
        obs_image_map[:pred_side] = data_side
        
        return obs_image_map, obs_bitmap, obs_bitmap_side, num_bricks, structure_name
    else
        obs_image_map[:pred] = data_front
        return obs_image_map, obs_bitmap, num_bricks, structure_name
    end
end