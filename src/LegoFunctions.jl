# ╔══════════════════════════════════════════════════════════╗
# ║  LEGO-gen Helper Functions                               ║
# ║  Utilities supporting the LEGO generative model pipeline ║
# ╚══════════════════════════════════════════════════════════╝
# 
# -----------------------------------------------------------
# Written by: Benedikt Farag, benedikt.farag@yale.edu
# November 17th 2025
# -----------------------------------------------------------


# ===========================================================
# Scales and sizes:
# ===========================================================
PLATE_HEIGHT_MM = 3.2
STUD_WIDTH_MM = 8.0

# FIXED_GLOBAL_SCALE set in ipynb

# plate and stud in world scale
STUD_TO_WORLD_SCALE = STUD_WIDTH_MM * FIXED_GLOBAL_SCALE 
PLATE_TO_WORLD_SCALE = PLATE_HEIGHT_MM * FIXED_GLOBAL_SCALE


# ===========================================================
# Struct definitions:
# ===========================================================

# ModelParams
# Stores all parameters required to render a scene in Mitsuba.
# - scene_d: Python dictionary describing the Mitsuba scene.
# - scene:  The loaded Mitsuba scene object (constructed from scene_d).
# - spp:    Samples-per-pixel for rendering.
@with_kw struct ModelParams
    scene_d::PyDict
    scene::PyObject = @pycall mi.load_dict(scene_d)::PyObject
    spp::Int32 = 16
end;

# BrickInstance
# Stores the local state of a single LEGO brick placed in the scene.
# - type:   Brick type identifier (e.g., :b1x1, :b2x1, :b4x4).
# - pos:    Integer grid coordinates (x,y,z) where the brick originates.
# - orient: Orientation in degrees (0 or 90).
struct BrickInstance
    type::Symbol       # :b1x1, :b2x1, :b4x4
    pos::Tuple{Int,Int,Int} # Grid position (x, y, z)
    orient::Int        # 0 or 90
end

# BuildState
# Keeps track of the generated LEGO structure and stores the scene state
# - grid:        3D occupancy grid (0 = empty, 1 = filled).
# - bricks:      List of all BrickInstance objects placed.
# - connections: Set of (x,y,z) studs where new bricks can attach (just top studs).
mutable struct BuildState
    grid::Array{Int, 3}
    bricks::Vector{BrickInstance}
    connections::Set{Tuple{Int,Int,Int}}
end


# ===========================================================
# Rendering Functions
# ===========================================================
    
# Renders the Mitsuba scene and returns a Float32 image array
function render(modelparams)
    bitmap = render_bitmap(modelparams)
    mu = @pycall np.array(bitmap)::Array{Float32, 3}
    return mu[:, :, 1]
end

# Renders the Mitsuba scene and returns a Python Bitmap object
function render_bitmap(modelparams)
    image = @pycall mi.render(modelparams.scene, spp=modelparams.spp)::PyObject
    bitmap = @pycall mi.Bitmap(image).convert(srgb_gamma=true)::PyObject
    return bitmap
end


# Converts a BuildState into a Mitsuba scene dictionary.
# Loads OBJ models, applies local brick transforms, and assembles them
# Returns a Python dict ready to be passed to ModelParams.
function build_scene_dict(state::BuildState, global_pos, global_rot_y, global_scale, img_size=128)
    scene_d = py"initialize_empty_scene($img_size)"o::PyObject
    
    brick_files = Dict(
        :b1x1 => "bricks/1x1x1.obj",
        :b2x1 => "bricks/2x1x1.obj",
        :b4x4 => "bricks/4x4.obj"
    )
    
    global_transform = @pycall mi.ScalarTransform4f.translate(global_pos).rotate([0, 1, 0], global_rot_y).scale(global_scale)::PyObject

    w = size(state.grid, 1)
    d = size(state.grid, 2)
    grid_center_offset = [w / 2.0 + 0.5, d / 2.0 + 0.5]
    
    # Get the (mm) height for each type
    h_b1x1 = get_height_mm(:b1x1) # 9.6
    h_b2x1 = get_height_mm(:b2x1) # 9.6
    h_b4x4 = get_height_mm(:b4x4) # 3.2

    # Fix transforms (by local brick coordinates)
    fix_transforms = Dict(
        :b1x1 => (@pycall mi.ScalarTransform4f.translate([0, h_b1x1, 0]).rotate([1, 0, 0], -180)::PyObject),
        :b2x1 => (@pycall mi.ScalarTransform4f.translate([0, h_b2x1, 0]).rotate([1, 0, 0], -180)::PyObject),
        :b4x4 => (@pycall mi.ScalarTransform4f.translate([0, h_b4x4, 0]).rotate([1, 0, 0], -180)::PyObject)
    )

    for (i, brick) in enumerate(state.bricks)
        brick_name = "brick_$(i)"
        
        # Get the (x,y) offset for this brick's footprint center
        (offset_x, offset_y) = get_footprint_center_offset(brick.type, brick.orient)
        
        # Add this offset to the brick's origin position
        pos_x_centered = brick.pos[1] + offset_x
        pos_y_centered = brick.pos[2] + offset_y
        
        # local_pos is now the (x, y, z) of the BRICK'S FOOTPRINT CENTER in (mm)
        local_pos = [
            # Grid X -> Mitsuba X (horizontal)
            (pos_x_centered - grid_center_offset[1]) * STUD_WIDTH_MM,
            
            # Grid Z -> Mitsuba Y (vertical/height)
            (brick.pos[3] - 1.0) * PLATE_HEIGHT_MM, 
            
            # Grid Y -> Mitsuba Z (depth)
            (pos_y_centered - grid_center_offset[2]) * STUD_WIDTH_MM
        ]
        
        local_rot_y = Float64(brick.orient)
        fix_transform = fix_transforms[brick.type]
            
        # make build transform
        build_transform = @pycall mi.ScalarTransform4f.translate(local_pos).rotate([0, 1, 0], local_rot_y)::PyObject
        local_transform = build_transform."__matmul__"(fix_transform)
        final_transform = global_transform."__matmul__"(local_transform)
        
        brick_dict = PyDict(Dict(
            "type" => "obj",
            "filename" => brick_files[brick.type],
            "face_normals" => true, 
            "to_world" => final_transform,
            "bsdf" => PyDict(Dict( "type" => "ref", "id" => "lego-blue" ))
        ))
        
        set!(scene_d, brick_name, brick_dict)
    end
    
    return scene_d
end


# ===========================================================
# Physics Functions
# ===========================================================

# Returns the height of a brick in plate units (1 plate = 3.2 mm)
function get_height(type::Symbol)
    if type == :b4x4
        return 1 # 1 plate height (plate)
    elseif type == :b1x1 || type == :b2x1
        return 3 # 3 plates height (bricks)
    end
    return 3 # Default to brick height
end

# Returns mm units of brick height
function get_height_mm(type::Symbol)
    if type == :b4x4
        return 3.2 # 1 plate * 3.2mm/plate
    elseif type == :b1x1 || type == :b2x1
        return 9.6 # 3 plates * 3.2mm/plate
    end
    return 9.6 # Default to brick height
end

# Returns the (dx, dy) footprint of a brick relative to its origin
# Used to test collisions and update the occupancy grid.
function get_footprint(type::Symbol, orient::Int)
    if type == :b1x1
        return [(0, 0)]
    elseif type == :b2x1
        return orient == 0 ? [(0, 0), (1, 0)] : [(0, 0), (0, 1)]
    elseif type == :b4x4
        # 4x4 plate, orientation doesn't change footprint
        return [(x, y) for x in 0:3 for y in 0:3]
    end
    return []
end

# Helper function to get the (x,y) offset of a brick's footprint center
# Used to align OBJ models correctly inside Mitsuba.
function get_footprint_center_offset(type::Symbol, orient::Int)
    if type == :b1x1
        return (0.0, 0.0) # 1x1 brick's center *is* its origin
    elseif type == :b2x1
        # Center is halfway along its length
        return orient == 0 ? (0.5, 0.0) : (0.0, 0.5)
    elseif type == :b4x4
        # Center of a 4x4 (from 0 to 3) is at 1.5
        return (1.5, 1.5)
    end
    return (0.0, 0.0)
end

# Returns the set of all (x,y,1) studs on the base layer where the first layer can be placed
function get_initial_connections(width::Int, depth::Int)
    connections = Set{Tuple{Int,Int,Int}}()
    for x in 1:width, y in 1:depth
        push!(connections, (x, y, 1))
    end
    return connections
end

# Checks if a new brick placement is valid
# Ensures: 
#   - brick attaches to an existing connection
#   - footprint stays in bounds
#   - brick volume does not collide with existing bricks
#   - brick height does not exceed grid
function is_valid_placement(state::BuildState, brick::BrickInstance)
    base_x, base_y, base_z = brick.pos
    footprint = get_footprint(brick.type, brick.orient)
    height = get_height(brick.type)

    # 1. Check support for the ORIGIN stud: must be valid
    if !((base_x, base_y, base_z) in state.connections)
        return false # Origin stud is not supported
    end

    # 2. Check bounds and overlap for ALL cells in the brick's volume
    for (dx, dy) in footprint
        x, y = base_x + dx, base_y + dy
        
        # Check bounds
        if !(1 <= x <= size(state.grid, 1) && 
             1 <= y <= size(state.grid, 2))
            return false # Footprint goes out of bounds
        end
        
        # Check volume overlap (from base_z to top)
        for z_offset in 0:(height - 1)
            z = base_z + z_offset
            
            if (z > size(state.grid, 3))
                return false # Brick is too tall for this spot
            end
            
            # This is the cell we are checking
            is_origin_base = (dx == 0 && dy == 0 && z_offset == 0)
            
            # If this cell is the origin's base, we skip the overlap check
            # (because we *know* it's a connection, not empty grid)
            if !is_origin_base && state.grid[x, y, z] != 0
                return false # Overlap with an existing brick
            end
        end
    end
    
    return true
end


# Deterministically updates the state after a valid brick is placed
#   - adds the brick to the list
#   - marks its occupied volume in the grid
#   - removes used connection studs
#   - adds new connection studs at the top surface
function place_brick!(state::BuildState, brick::BrickInstance)
    # 1. Add brick to list
    push!(state.bricks, brick)
    
    base_x, base_y, base_z = brick.pos
    footprint = get_footprint(brick.type, brick.orient)
    height = get_height(brick.type)
    
    # The new connections are at the TOP of this brick
    new_connection_z = base_z + height
    
    for (dx, dy) in footprint
        x, y = base_x + dx, base_y + dy
        
        # Check bounds just in case (though is_valid should catch it)
        if !(1 <= x <= size(state.grid, 1) && 1 <= y <= size(state.grid, 2))
            continue
        end

        # 1. Mark grid as occupied (for the full height)
        for z_offset in 0:(height - 1)
            z = base_z + z_offset
            if (z <= size(state.grid, 3))
                state.grid[x, y, z] = 1
            end
        end
        
        # 2. Remove the connection point we just built on
        # (and any other connections this brick is now covering).
        # pop! with a default `nothing` safely does nothing if the key isn't there.
        pop!(state.connections, (x, y, base_z), nothing)
        
        # 3. Add new connection points on top
        if new_connection_z <= size(state.grid, 3)
            push!(state.connections, (x, y, new_connection_z))
        end
    end
end

# ===========================================================
# Generative Model 1: MH sampling
# ===========================================================

# Generative Model 1: Sequential brick placement for MH inference.
# Samples number of bricks, then builds top-down by sampling
# connection, brick type, and orientation at each step.
@gen function lego_structure_model(poi_lambda, width=WIDTH_VAL, depth=DEPTH_VAL, max_height=MAX_HEIGHT_VAL, img_size=IMAGE_SIZE)
    
    # deterministic location
    global_pos = FIXED_GLOBAL_POS
    global_rot_y = FIXED_GLOBAL_ROT_Y
    global_scale = FIXED_GLOBAL_SCALE
    
    
    # --- 2. Structural Priors (Discrete) ---
    #num_bricks ~ poisson(15) # Prior on number of bricks
    num_bricks ~ poisson(poi_lambda)
    
    # --- 3. (Non-Gen) Deterministic Setup ---
    # Initialize the state for the build process
    initial_conns = get_initial_connections(width, depth)
    state = BuildState(
        zeros(Int, width, depth, max_height),
        Vector{BrickInstance}(),
        initial_conns
    )
    
    brick_types = [:b1x1, :b2x1, :b4x4]

    # --- 4. The Build Process (Sequential Discrete Choices) ---
    for i in 1:num_bricks
        
        # Gracefully handle running out of connections
        if isempty(state.connections)
            # If we have no place to build, we stop. Sample dummy values (will be ignored)
            @trace(uniform_discrete(1, 1), :brick => i => :conn_idx)
            @trace(uniform_discrete(1, 3), :brick => i => :type)
            @trace(uniform_discrete(0, 1), :brick => i => :orient)
            continue # Skip to next brick
        end
    

        # A. Sample a connection point to build on
        # Use a nested address like `:brick => i => :conn_idx`
        conn_idx = @trace(uniform_discrete(1, length(state.connections)), :brick => i => :conn_idx)
        
        # B. Sample a brick type
        type_idx = @trace(uniform_discrete(1, 3), :brick => i => :type)
        
        # C. Sample an orientation
        orient = @trace(uniform_discrete(0, 1), :brick => i => :orient) # 0 or 90 degrees

        # Check if the sampled index is still valid after the proposal.
        # If not, this proposal "broke" the state, so we skip.
        if conn_idx > length(state.connections)
            continue # drop this brick proposal and move to the next
        end
        
        # --- 5. (Non-Gen) Deterministic State Update ---
        
        # Get the actual values from our samples
        chosen_conn = collect(state.connections)[conn_idx] 
        chosen_type = brick_types[type_idx]
        chosen_orient = orient * 90
        
        brick = BrickInstance(chosen_type, chosen_conn, chosen_orient)
        
        # Deterministically check if this is a valid move
        if is_valid_placement(state, brick)
            # If valid, update the state
            place_brick!(state, brick)
        end
    end

    # 6. Render the scene
    scene_d = build_scene_dict(state, global_pos, global_rot_y, global_scale, img_size)
    modelparams = ModelParams(scene_d=scene_d, spp=16) # Low SPP for inference
    
    # This render function is from your example
    mu = render(modelparams) 

    # --- 7. Likelihood ---
    # Compare rendered image (mu) to the observation: tight variance for a strong likelihood signal
    pred ~ broadcasted_normal(mu, 0.1)
end

# ===========================================================
# Generative Model 2: Layer building
# ===========================================================

# Return all valid connection points at height z
function get_connections_at_z(state::BuildState, z::Int)
    # Filter the Set into a Vector
    conns_at_z = filter(
        c -> c[3] == z, # c is a (x, y, z) tuple
        state.connections
    )
    return collect(conns_at_z) # Return a Vector for stable indexing
end

# Generative Model 2: Layer-by-layer LEGO construction.
# Builds each z-level by sampling brick count, connection index,
# brick type, and orientation within that layer.
@gen function lego_layer_model(width=WIDTH_VAL, depth=DEPTH_VAL, max_height=MAX_HEIGHT_VAL, img_size=IMAGE_SIZE, two_views::Bool=false)
    
    # --- 1. Global Pose (unchanged) ---
    global_pos = FIXED_GLOBAL_POS
    global_rot_y = FIXED_GLOBAL_ROT_Y
    global_scale = FIXED_GLOBAL_SCALE
    
    # --- 2. (Non-Gen) Deterministic Setup (unchanged) ---
    initial_conns = get_initial_connections(width, depth)
    state = BuildState(
        zeros(Int, width, depth, max_height),
        Vector{BrickInstance}(),
        initial_conns
    )
    brick_types = [:b1x1, :b2x1, :b4x4]

    # --- 3. The *NEW* Build Process ---
    # We loop *by height* (z-level)
    for z in 1:max_height
        
        # Get all valid connection points AT THIS Z-LEVEL
        # This is a *crucial* new helper function you'll write
        conns_at_z = get_connections_at_z(state, z)
        
        # Gracefully handle no connections at this level
        if isempty(conns_at_z)
            # We must still sample to keep the trace valid
            @trace(poisson(0.1), :bricks_at_z => z => :num_bricks)
            continue # Move to the next z-level
        end

        # A. Sample how many bricks to place at this z-level
        # The prior (e.g., poisson(2)) is a good tuning parameter
        num_bricks_at_z = @trace(poisson(2), :bricks_at_z => z => :num_bricks)

        # B. Loop and place those bricks
        for i in 1:num_bricks_at_z
            
            # C. Sample a connection *from this z-level's list*
            conn_idx = @trace(uniform_discrete(1, length(conns_at_z)), :bricks_at_z => z => i => :conn_idx)
            
            # D. Sample type and orientation (unchanged)
            type_idx = @trace(uniform_discrete(1, 3), :bricks_at_z => z => i => :type)
            orient = @trace(uniform_discrete(0, 1), :bricks_at_z => z => i => :orient)

            # Check if proposal is still valid (in case a previous
            # brick at this level invalidated this connection)
            if conn_idx > length(conns_at_z)
                continue
            end
            
            # --- 5. (Non-Gen) Deterministic State Update (unchanged) ---
            chosen_conn = conns_at_z[conn_idx] # No need to collect(Set)
            chosen_type = brick_types[type_idx]
            chosen_orient = orient * 90
            
            brick = BrickInstance(chosen_type, chosen_conn, chosen_orient)
            
            if is_valid_placement(state, brick)
                place_brick!(state, brick)
                
                # recalculate the connections at z
                conns_at_z = get_connections_at_z(state, z)

                # If we just used up the last connection, break
                if isempty(conns_at_z)
                    break
                end
            end
        end
    end

    # --- 6. Conditional Rendering and Likelihood ---
    if two_views
        # === Mode A: Two Views (Front + Side) ===
        
        # 1. Render Front
        scene_d_front = build_scene_dict(state, global_pos, global_rot_y, global_scale, img_size)
        modelparams_front = ModelParams(scene_d=scene_d_front, spp=16)
        mu_front = render(modelparams_front)
        
        # 2. Render Side (Rotate world by 90)
        scene_d_side = build_scene_dict(state, global_pos, global_rot_y + 90.0, global_scale, img_size)
        modelparams_side = ModelParams(scene_d=scene_d_side, spp=16)
        mu_side = render(modelparams_side)

        # 3. Likelihoods (Note the address names!)
        @trace(broadcasted_normal(mu_front, 0.1), :pred_front)
        @trace(broadcasted_normal(mu_side, 0.1), :pred_side)

    else
        # === Mode B: Single View (Original) ===
        
        scene_d = build_scene_dict(state, global_pos, global_rot_y, global_scale, img_size)
        modelparams = ModelParams(scene_d=scene_d, spp=16)
        mu = render(modelparams)
        
        # Likelihood (Note: single address :pred)
        @trace(broadcasted_normal(mu, 0.1), :pred)
    end
end

# helper function to count number of bricks
function count_total_bricks(choices, max_height=8)
    total = 0
    for z in 1:max_height
         if has_value(choices, :bricks_at_z => z => :num_bricks)
            total += choices[:bricks_at_z => z => :num_bricks]
         end
    end
    return total
end


# ===========================================================
# View Trace Function
# ===========================================================

# FOR STRUCTURE MODEL: get render and state for some trace
function view_structure_trace(
    trace_to_view; 
    spp::Int = SPP_VAL, 
    pos::Vector{Float64} = FIXED_GLOBAL_POS, 
    rot_y::Float64 = FIXED_GLOBAL_ROT_Y, 
    scale::Float64 = FIXED_GLOBAL_SCALE,
    img_size=IMAGE_SIZE,
    width=WIDTH_VAL, depth=DEPTH_VAL, max_height=MAX_HEIGHT_VAL
)
    choices = get_choices(trace_to_view)
    
    # Re-run the deterministic logic from the model
    #width, depth, max_height = 7, 7, 8
    initial_conns = get_initial_connections(width, depth)
    state = BuildState(
        zeros(Int, width, depth, max_height),
        Vector{BrickInstance}(),
        initial_conns
    )
    brick_types = [:b1x1, :b2x1, :b4x4]
    
    num_bricks_in_trace = has_value(choices, :num_bricks) ? choices[:num_bricks] : 0

    for i in 1:num_bricks_in_trace
        # Handle cases where a brick trace doesn't exist
        if !has_value(choices, :brick => i => :conn_idx)
            continue
        end

        conn_idx = choices[:brick => i => :conn_idx]
        if conn_idx > length(state.connections)
            continue # This sample was invalid, skip
        end

        chosen_conn = collect(state.connections)[conn_idx] 
        chosen_type = brick_types[choices[:brick => i => :type]]
        chosen_orient = choices[:brick => i => :orient] * 90
        
        brick = BrickInstance(chosen_type, chosen_conn, chosen_orient)
        if is_valid_placement(state, brick)
            place_brick!(state, brick)
        end
    end
    
    # make the scene
    scene_d = build_scene_dict(state, pos, rot_y, scale, img_size)
    
    # Render with the specified SPP
    modelparams = ModelParams(scene_d=scene_d, spp=spp) 
    
    # Use your `render_bitmap` function
    bitmap = render_bitmap(modelparams)
    return bitmap, state
end

# FOR LAYER MODEL: get render and state for some trace
function view_layer_trace(
    trace_to_view; 
    spp::Int = SPP_VAL, 
    pos::Vector{Float64} = FIXED_GLOBAL_POS, 
    rot_y::Float64 = FIXED_GLOBAL_ROT_Y, 
    scale::Float64 = FIXED_GLOBAL_SCALE,
    img_size=IMAGE_SIZE,
    width=WIDTH_VAL, depth=DEPTH_VAL, max_height=MAX_HEIGHT_VAL
)
    choices = get_choices(trace_to_view)
    
    # --- 1. Re-run the deterministic logic from the model ---
    initial_conns = get_initial_connections(width, depth)
    state = BuildState(
        zeros(Int, width, depth, max_height),
        Vector{BrickInstance}(),
        initial_conns
    )
    brick_types = [:b1x1, :b2x1, :b4x4]

    # --- 2. The Replay Loop ---
    for z in 1:max_height
        
        # Get the connections that *would have been available*
        # at the start of this z-level's loop
        conns_at_z = get_connections_at_z(state, z)
        
        # Check if the trace has brick data for this level
        if !has_value(choices, :bricks_at_z => z => :num_bricks)
            continue # No bricks sampled at this level
        end
        num_bricks_at_z = choices[:bricks_at_z => z => :num_bricks]

        if isempty(conns_at_z) && num_bricks_at_z > 0
            continue # This was a "dummy" sample, no valid conns
        end
        
        for i in 1:num_bricks_at_z
            
            # Check if this specific brick's data exists
            if !has_value(choices, :bricks_at_z => z => i => :conn_idx)
                continue
            end

            conn_idx = choices[:bricks_at_z => z => i => :conn_idx]
            
            # Check for invalid index (same as in the model)
            if conn_idx > length(conns_at_z)
                continue 
            end

            # Get the actual values from the trace
            chosen_conn = conns_at_z[conn_idx] # Get from our *calculated* list
            chosen_type = brick_types[choices[:bricks_at_z => z => i => :type]]
            chosen_orient = choices[:bricks_at_z => z => i => :orient] * 90
            
            brick = BrickInstance(chosen_type, chosen_conn, chosen_orient)
            
            # Deterministically check and place
            if is_valid_placement(state, brick)
                place_brick!(state, brick)
                
                # *** CRITICAL ***
                # We must update conns_at_z *just like in the model*
                # A brick placed at (4,4,1) might cover (5,4,1)
                conns_at_z = get_connections_at_z(state, z)

                if isempty(conns_at_z)
                    break # No more connections at this z-level
                end
            end
        end
    end
    
    # --- 3. Build and Render Scene ---
    # This part correctly uses your keyword arguments
    scene_d = build_scene_dict(state, pos, rot_y, scale, img_size)
    
    modelparams = ModelParams(scene_d=scene_d, spp=spp) 
    bitmap = render_bitmap(modelparams)
    
    # Return both the image and the final state for debugging
    return bitmap, state
end
