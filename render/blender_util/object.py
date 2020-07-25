"""
Utility functions for manipulating objects in Blender

Xiuming Zhang, MIT CSAIL
July 2017

Contributor: Xingyuan Sun
"""

import logging
import re
from os.path import abspath
import numpy as np
import bpy
import bmesh
from mathutils import Matrix, Vector
# import logging_colorer # noqa: F401 # pylint: disable=unused-import

logging.basicConfig(level=logging.INFO)
thisfile = abspath(__file__)


def remove_objects(name_pattern, regex=False):
    """
    Remove object(s) from current scene

    Args:
        name_pattern: Name or name pattern of object(s) to remove
            String
        regex: Whether to interpret 'name_pattern' as a regex
            Boolean
            Optional; defaults to False
    """
    thisfunc = thisfile + '->remove_objects()'

    objs = bpy.data.objects
    removed = []

    if regex:
        assert (name_pattern != '*'), "Want to match everything? Correct regex for this is '.*'"

        name_pattern = re.compile(name_pattern)

        for obj in objs:
            if name_pattern.match(obj.name):
                obj.select = True
                removed.append(obj.name)
            else:
                obj.select = False

    else:
        for obj in objs:
            if obj.name == name_pattern:
                obj.select = True
                removed.append(obj.name)
            else:
                obj.select = False

    # Delete
    bpy.ops.object.delete()

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.scene.update()

    logging.info("%s: Removed from scene: %s", thisfunc, removed)


def import_object(model_path, rot_mat=((1, 0, 0), (0, 1, 0), (0, 0, 1)), trans_vec=(0, 0, 0), scale=1, name=None):
    """
    Import external object to current scene, the low-level way

    Args:
        model_path: Path to object to add
            String
        rot_mat: 3D rotation matrix PRECEDING translation
            3-by-3 array_like
            Optional; defaults to identity matrix
        trans_vec: 3D translation vector FOLLOWING rotation
            3-array_like
            Optional; defaults to zero vector
        scale: Scale of the object
            Float
            Optional; defaults to 1
        name: Object name after import
            String
            Optional; defaults to name specified in model

    Returns:
        obj: Handle(s) of imported object(s)
            bpy_types.Object or list thereof
    """
    thisfunc = thisfile + '->import_object()'

    # Import
    if model_path.endswith('.obj'):
        bpy.ops.import_scene.obj(filepath=model_path, axis_forward='-Z', axis_up='Y')
    else:
        raise NotImplementedError(".%s" % model_path.split('.')[-1])

    obj_list = []
    for i, obj in enumerate(bpy.context.selected_objects):

        # Rename
        if name is not None:
            if len(bpy.context.selected_objects) == 1:
                obj.name = name
            else:
                obj.name = name + '_' + str(i)

        # Compute world matrix
        trans_4x4 = Matrix.Translation(trans_vec)
        rot_4x4 = Matrix(rot_mat).to_4x4()
        scale_4x4 = Matrix(np.eye(4)) # don't scale here
        obj.matrix_world = trans_4x4 * rot_4x4 * scale_4x4

        # Scale
        obj.scale = (scale, scale, scale)

        obj_list.append(obj)

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.scene.update()

    logging.info("%s: Imported: %s", thisfunc, model_path)

    if len(obj_list) == 1:
        return obj_list[0]
    else:
        return obj_list


def add_cylinder_between(pt1, pt2, r, name=None):
    """
    Add a cylinder specified by two end points and radius
        Useful for visualizing rays in ray tracing

    Args:
        pt1: Global coordinates of point 1
            Array_like containing three floats
        pt2: Global coordinates of point 2
            Array_like containing three floats
        r: Cylinder radius
            Radius
        name: Cylinder name
            String
            Optional; defaults to Blender defaults

    Returns:
        cylinder_obj: Handle of added cylinder
            bpy_types.Object
    """
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)

    d = pt2 - pt1

    # Add cylinder at the correct location
    dist = np.linalg.norm(d)
    loc = (pt1[0] + d[0] / 2, pt1[1] + d[1] / 2, pt1[2] + d[2] / 2)
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=dist, location=loc)

    cylinder_obj = bpy.context.object

    if name is not None:
        cylinder_obj.name = name

    # Further rotate it accordingly
    phi = np.arctan2(d[1], d[0])
    theta = np.arccos(d[2] / dist)
    cylinder_obj.rotation_euler[1] = theta
    cylinder_obj.rotation_euler[2] = phi

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.scene.update()

    return cylinder_obj


def add_rectangular_plane(center_loc=(0, 0, 0), point_to=(0, 0, 1), size=(2, 2), name=None):
    """
    Add a rectangular plane specified by its center location, dimensions,
        and where its +z points to

    Args:
        center_loc: Plane center location in world coordinates
            Array_like containing three floats
            Optional; defaults to world origin
        point_to: Direction to which plane's +z points to in world coordinates
            Array_like containing three floats
            Optional; defaults to world +z
        size: Sizes in x and y directions (0 in z)
            Array_like containing two floats
            Optional; defaults to a square with side length 2
        name: Plane name
            String
            Optional; defaults to Blender defaults

    Returns:
        plane_obj: Handle of added plane
            bpy_types.Object
    """
    center_loc = np.array(center_loc)
    point_to = np.array(point_to)
    size = np.append(np.array(size), 0)

    bpy.ops.mesh.primitive_plane_add(location=center_loc)

    plane_obj = bpy.context.object

    if name is not None:
        plane_obj.name = name

    plane_obj.dimensions = size

    # Point it to target
    direction = Vector(point_to) - plane_obj.location
    # Find quaternion that rotates plane's 'Z' so that it aligns with 'direction'
    # This rotation is not unique because the rotated plane can still rotate about direction vector
    # Specifying 'Y' gives the rotation quaternion with plane's 'Y' pointing up
    rot_quat = direction.to_track_quat('Z', 'Y')
    plane_obj.rotation_euler = rot_quat.to_euler()

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.scene.update()

    return plane_obj


def create_mesh(verts, faces, name):
    """
    Create a mesh from vertices and faces

    Args:
        verts: Local coordinates of the vertices
            Array_like of shape (n, 3)
        faces: Faces specified by ordered vertex indices
            List of tuples of natural numbers < n
        name: Mesh name
            String

    Returns:
        mesh_data: Mesh data created
            bpy_types.Mesh
    """
    thisfunc = thisfile + '->create_mesh()'

    verts = np.array(verts)

    # Create mesh
    mesh_data = bpy.data.meshes.new(name)
    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()

    logging.info("%s: Mesh '%s' created", thisfunc, name)

    return mesh_data


def create_object_from_mesh(mesh_data, obj_name, location=(0, 0, 0), rotation_euler=(0, 0, 0), scale=(1, 1, 1)):
    """
    Create object from mesh data

    Args:
        mesh_data: Mesh data
            bpy_types.Mesh
        obj_name: Object name
            String
        location: Object location in world coordinates
            3-tuple of floats
            Optional; defaults to world origin
        rotation_euler: Object rotation in radians
            3-tuple of floats
            Optional; defaults to aligning with world coordinates
        scale: Object scale
            3-tuple of floats
            Optional; defaults to unit scale

    Returns:
        obj: Object created
            bpy_types.Object
    """
    thisfunc = thisfile + '->create_object_from_mesh()'

    # Create
    obj = bpy.data.objects.new(obj_name, mesh_data)

    # Link to current scene
    scene = bpy.context.scene
    scene.objects.link(obj)
    obj.select = True
    scene.objects.active = obj # make the selection effective

    # Set attributes
    obj.location = location
    obj.rotation_euler = rotation_euler
    obj.scale = scale

    logging.info("%s: Object '%s' created from mesh data and selected",
                 thisfunc, obj_name)

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    return obj


def _clear_nodetree_for_active_material(obj):
    """
    Internal helper function clears the node tree of active material
        so that desired node tree can be cleanly set up.
        If no active material, one will be created
    """
    # Create material if none
    if obj.active_material is None:
        mat = bpy.data.materials.new(name='new-mat-for-%s' % obj.name)
        if obj.data.materials:
            # Assign to first material slot
            obj.data.materials[0] = mat
        else:
            # No slots
            obj.data.materials.append(mat)

    active_mat = obj.active_material
    active_mat.use_nodes = True
    node_tree = active_mat.node_tree
    nodes = node_tree.nodes

    # Remove all nodes
    for node in nodes:
        nodes.remove(node)

    return node_tree, nodes


def color_vertices(obj, vert_ind, colors):
    """
    Color each vertex of interest with the given color; i.e., same color for all its loops
        Useful for making a 3D heatmap

    Args:
        obj: Object
            bpy_types.Object
        vert_ind: Index/indices of vertex/vertices to color
            Integer or list thereof
        colors: RGB value(s) to paint on vertex/vertices
            Tuple of three floats in [0, 1] or list thereof
                - If one tuple, this color will be applied to all
                - If list of tuples, must be of same length as vert_ind
    """
    thisfunc = thisfile + '->color_vertices()'

    # Validate inputs
    if isinstance(vert_ind, int):
        vert_ind = [vert_ind]
    if isinstance(colors, tuple):
        colors = [colors] * len(vert_ind)
    assert (len(colors) == len(vert_ind)), \
        "'colors' and 'vert_ind' must be of the same length, or 'colors' is a single tuple"

    scene = bpy.context.scene
    scene.objects.active = obj
    obj.select = True
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh = obj.data

    if mesh.vertex_colors:
        vcol_layer = mesh.vertex_colors.active
    else:
        vcol_layer = mesh.vertex_colors.new()

    # A vertex and one of its edges combined are called a loop, which has a color
    # So if a vertex has four outgoing edges, it has four colors for the four loops
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            loop_vert_idx = mesh.loops[loop_idx].vertex_index
            try:
                # In the list
                color_idx = vert_ind.index(loop_vert_idx)
                vcol_layer.data[loop_idx].color = colors[color_idx]
            except ValueError:
                # Not found
                pass

    # Set up nodes for vertex colors
    node_tree, nodes = _clear_nodetree_for_active_material(obj)
    nodes.new('ShaderNodeAttribute')
    nodes.new('ShaderNodeBsdfDiffuse')
    nodes.new('ShaderNodeOutputMaterial')
    nodes['Attribute'].attribute_name = vcol_layer.name
    node_tree.links.new(nodes['Attribute'].outputs[0], nodes['Diffuse BSDF'].inputs[0])
    node_tree.links.new(nodes['Diffuse BSDF'].outputs[0], nodes['Material Output'].inputs[0])

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logging.info("%s: Vertex color(s) added to '%s'", thisfunc, obj.name)
    logging.warning("%s:     ..., so node tree of '%s' has changed", thisfunc, obj.name)


def setup_diffuse_nodetree(obj, roughness=0, color=None):
    """
    Set up a diffuse texture node tree for imported object bundled with texture map
        or a diffuse color node tree for an object without material or texture map
        Mathematically, it's either Lambertian (no roughness) or Oren-Nayar (with roughness)

    Args:
        obj: Object bundled with texture map
            bpy_types.Object
        roughness: Roughness in Oren-Nayar model
            Float
            Optional; defaults to 0, i.e., Lambertian
        color: RGBA; only useful when object does not have a texture map
            4-tuple of floats ranging from 0 to 1
            Optional; defaults to None
    """
    thisfunc = thisfile + '->setup_diffuse_nodetree()'

    scene = bpy.context.scene
    engine = scene.render.engine
    if engine != 'CYCLES':
        raise NotImplementedError(engine)

    node_tree, nodes = _clear_nodetree_for_active_material(obj)
    texture = obj.active_material.active_texture

    if texture is not None:
        # Bundled texture found -- set up diffuse texture node tree
        nodes.new('ShaderNodeTexImage')
        nodes['Image Texture'].image = texture.image
        nodes.new('ShaderNodeBsdfDiffuse')
        nodes.new('ShaderNodeOutputMaterial')
        node_tree.links.new(nodes['Image Texture'].outputs[0], nodes['Diffuse BSDF'].inputs[0])
        node_tree.links.new(nodes['Diffuse BSDF'].outputs[0], nodes['Material Output'].inputs[0])

        if color is not None:
            logging.warning("%s: %s has a texture map associated with it -- 'color' argument ignored",
                            thisfunc, obj.name)

    else:
        # No texture found -- set up diffuse color tree
        if color is None:
            color = (1, 1, 1, 1)
            logging.warning((
                "%s: %s has no texture map associated with it, "
                "and you have not provided any value for argument 'color', "
                "so opaque white color is used"
            ), thisfunc, obj.name)

        nodes.new('ShaderNodeBsdfDiffuse')
        nodes['Diffuse BSDF'].inputs[0].default_value = color
        nodes.new('ShaderNodeOutputMaterial')
        node_tree.links.new(nodes['Diffuse BSDF'].outputs[0], nodes['Material Output'].inputs[0])

    # Roughness
    node_tree.nodes['Diffuse BSDF'].inputs[1].default_value = roughness

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logging.info("%s: Diffuse node tree set up for '%s'", thisfunc, obj.name)


def setup_emission_nodetree(obj, color=(1, 1, 1, 1), strength=1):
    """
    Set up an emission node tree for the object

    Args:
        obj: Object bundled with texture map
            bpy_types.Object
        color: Emission RGBA
            4-tuple of floats ranging from 0 to 1
            Optional; defaults to opaque white
        strength: Emission strength
            Float
            Optional; defaults to 1
    """
    thisfunc = thisfile + '->setup_emission_nodetree()'

    scene = bpy.context.scene
    engine = scene.render.engine
    if engine != 'CYCLES':
        raise NotImplementedError(engine)

    node_tree, nodes = _clear_nodetree_for_active_material(obj)

    nodes.new('ShaderNodeEmission')
    nodes['Emission'].inputs[0].default_value = color
    nodes['Emission'].inputs[1].default_value = strength
    nodes.new('ShaderNodeOutputMaterial')
    node_tree.links.new(nodes['Emission'].outputs[0], nodes['Material Output'].inputs[0])

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logging.info("%s: Emission node tree set up for '%s'", thisfunc, obj.name)


def setup_holdout_nodetree(obj):
    """
    Set up a holdout node tree for the object

    Args:
        obj: Object bundled with texture map
            bpy_types.Object
    """
    thisfunc = thisfile + '->setup_holdout_nodetree()'

    scene = bpy.context.scene
    engine = scene.render.engine
    if engine != 'CYCLES':
        raise NotImplementedError(engine)

    node_tree, nodes = _clear_nodetree_for_active_material(obj)

    nodes.new('ShaderNodeHoldout')
    nodes.new('ShaderNodeOutputMaterial')
    node_tree.links.new(nodes['Holdout'].outputs[0], nodes['Material Output'].inputs[0])

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logging.info("%s: Holdout node tree set up for '%s'", thisfunc, obj.name)


def get_bmesh(obj):
    """
    Get Blender mesh data from object

    Args:
        obj: Object
            bpy_types.Object

    Returns:
        bm: Blender mesh data
            BMesh
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.scene.update()

    return bm


def subdivide_mesh(obj, n_subdiv=2):
    """
    Subdivide mesh of object

    Args:
        obj: Object whose mesh is to be subdivided
            bpy_types.Object
        n_subdiv: Number of subdivision levels
            Integer
            Optional; defaults to 2
    """
    thisfunc = thisfile + '->subdivide_mesh()'

    scene = bpy.context.scene

    # All objects need to be in 'OBJECT' mode to apply modifiers -- maybe a Blender bug?
    for o in bpy.data.objects:
        scene.objects.active = o
        bpy.ops.object.mode_set(mode='OBJECT')
        o.select = False
    obj.select = True
    scene.objects.active = obj

    bpy.ops.object.modifier_add(type='SUBSURF')
    obj.modifiers['Subsurf'].subdivision_type = 'CATMULL_CLARK'
    obj.modifiers['Subsurf'].levels = n_subdiv
    obj.modifiers['Subsurf'].render_levels = n_subdiv

    # Apply modifier
    bpy.ops.object.modifier_apply(modifier='Subsurf', apply_as='DATA')

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logging.info("%s: Subdivided mesh of '%s'", thisfunc, obj.name)


def select_mesh_elements_by_vertices(obj, vert_ind, select_type):
    """
    Select vertices or their associated edges/faces in edit mode

    Args:
        obj: Object
            bpy_types.Object
        vert_ind: A single vertex index or a list of many
            Non-negative integer or list thereof
        select_type: Type of mesh elements to select
            'vertex', 'edge' or 'face'
    """
    thisfunc = thisfile + '->select_mesh_elements_by_vertices()'

    if isinstance(vert_ind, int):
        vert_ind = [vert_ind]

    # Edit mode
    scene = bpy.context.scene
    scene.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Deselect all
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_mode(type='EDGE')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.mesh.select_all(action='DESELECT')

    bm = bmesh.from_edit_mesh(obj.data)
    bvs = bm.verts

    bvs.ensure_lookup_table()
    for i in vert_ind:
        bv = bvs[i]

        if select_type == 'vertex':
            bv.select = True

        # Select all edges with this vertex at an end
        elif select_type == 'edge':
            for be in bv.link_edges:
                be.select = True

        # Select all faces with this vertex
        elif select_type == 'face':
            for bf in bv.link_faces:
                bf.select = True

        else:
            raise ValueError("Wrong selection type")

    # Update viewport
    scene.objects.active = scene.objects.active

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logging.info("%s: Selected %s elements of '%s'", thisfunc, select_type, obj.name)
