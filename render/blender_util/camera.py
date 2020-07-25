"""
Utility functions for manipulating cameras in Blender

Xiuming Zhang, MIT CSAIL
July 2017
"""

import logging
from os import remove, rename
from os.path import abspath, dirname, basename
from time import time
import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
import numpy as np
import cv2
from xiuminglib.blender import object as xb_object
import logging_colorer # noqa: F401 # pylint: disable=unused-import

logging.basicConfig(level=logging.INFO)
thisfile = abspath(__file__)


def add_camera(xyz=(0, 0, 0), rot_vec_rad=(0, 0, 0), name=None, proj_model='PERSP', f=35, sensor_fit='HORIZONTAL', sensor_width=32, sensor_height=18, clip_start=0.1, clip_end=100):
    """
    Add camera to current scene

    Args:
        xyz: Location
            3-tuple of floats
            Optional; defaults to (0, 0, 0)
        rot_vec_rad: Rotation angle in radians around x, y and z
            3-tuple of floats
            Optional; defaults to (0, 0, 0)
        name: Light object name
            String
            Optional
        proj_model: Camera projection model
            'PERSP', 'ORTHO', or 'PANO'
            Optional; defaults to 'PERSP'
        f: Focal length in mm
            Float
            Optional; defaults to 35
        sensor_fit: Sensor fit; also see get_camera_matrix()
            'HORIZONTAL' or 'VERTICAL'
            Optional; defaults to 'HORIZONTAL'
        sensor_width: Sensor width in mm
            Float
            Optional; defaults to 32
        sensor_height: Sensor height in mm
            Float
            Optional; defaults to 18
        clip_start: Near clipping distance
            Float
            Optional; defaults to 0.1
        clip_end: Far clipping distance
            Float
            Optional; defaults to 100

    Returns:
        cam: Handle of added camera
            bpy_types.Object
    """
    thisfunc = thisfile + '->add_camera()'

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object

    if name is not None:
        cam.name = name

    cam.location = xyz
    cam.rotation_euler = rot_vec_rad

    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height
    cam.data.clip_start = clip_start
    cam.data.clip_end = clip_end

    logging.info("%s: Camera '%s' added", thisfunc, cam.name)

    return cam


def easyset(cam, xyz=None, rot_vec_rad=None, name=None, proj_model=None, f=None, sensor_fit=None, sensor_width=None, sensor_height=None):
    """
    Set camera parameters more easily

    Args:
        cam: Handle of camera
            bpy_types.Object
        xyz: Location
            3-tuple of floats
            Optional; defaults to None (no change)
        rot_vec_rad: Rotation angle in radians around x, y and z
            3-tuple of floats
            Optional; defaults to None (no change)
        name: Light object name
            String
            Optional; defaults to None (no change)
        proj_model: Camera projection model
            'PERSP', 'ORTHO', or 'PANO'
            Optional; defaults to None (no change)
        f: Focal length in mm
            Float
            Optional; defaults to None (no change)
        sensor_fit: Sensor fit; also see get_camera_matrix()
            'HORIZONTAL' or 'VERTICAL'
            Optional; defaults to None (no change)
        sensor_width: Sensor width in mm
            Float
            Optional; defaults to None (no change)
        sensor_height: Sensor height in mm
            Float
            Optional; defaults to None (no change)
    """
    if name is not None:
        cam.name = name

    if xyz is not None:
        cam.location = xyz

    if rot_vec_rad is not None:
        cam.rotation_euler = rot_vec_rad

    if proj_model is not None:
        cam.data.type = proj_model

    if f is not None:
        cam.data.lens = f

    if sensor_fit is not None:
        cam.data.sensor_fit = sensor_fit

    if sensor_width is not None:
        cam.data.sensor_width = sensor_width

    if sensor_height is not None:
        cam.data.sensor_height = sensor_height


def point_camera_to(cam, xyz_target):
    """
    Point camera to target

    Args:
        cam: Camera object
            bpy_types.Object
        xyz_target: Target point
            3-tuple of floats
    """
    thisfunc = thisfile + '->point_camera_to()'

    xyz_target = Vector(xyz_target)
    direction = xyz_target - cam.location
    # Find quaternion that rotates '-Z' so that it aligns with 'direction'
    # This rotation is not unique because the rotated camera can still rotate about direction vector
    # Specifying 'Y' gives the rotation quaternion with camera's 'Y' pointing up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    logging.info("%s: Camera '%s' pointed to %s", thisfunc, cam.name, xyz_target)

    return cam


def intrinsics_compatible_with_scene(cam, eps=1e-6):
    """
    Check if camera intrinsic parameters (sensor size and pixel aspect ratio)
        are comptible with the current scene (render resolutions and their scale)

    Args:
        cam: Camera object
            bpy_types.Object
        eps: Epsilon for numerical comparison; considered equal if abs(a - b) / b < eps
            Float
            Optional; defaults to 1e-6

    Returns:
        comptible: Result
            Boolean
    """
    thisfunc = thisfile + '->intrinsics_compatible_with_scene()'

    # Camera
    sensor_width_mm = cam.data.sensor_width
    sensor_height_mm = cam.data.sensor_height

    # Scene
    scene = bpy.context.scene
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    # Do these parameters make sense together?
    mm_per_pix_horizontal = sensor_width_mm / (w * scale)
    mm_per_pix_vertical = sensor_height_mm / (h * scale)

    if abs(mm_per_pix_horizontal / mm_per_pix_vertical - pixel_aspect_ratio) / pixel_aspect_ratio < eps:
        logging.info("%s: OK", thisfunc)

        return True
    else:
        logging.error((
            "%s: Render resolutions (w_pix = %d; h_pix = %d), sensor size (w_mm = %f; h_mm = %f), "
            "and pixel aspect ratio (r = %f) don't make sense together. This could cause "
            "unexpected behaviors later. Consider running correct_sensor_height()"
        ), thisfunc, w, h, sensor_width_mm, sensor_height_mm, pixel_aspect_ratio)

        return False


def correct_sensor_height(cam):
    """
    Make render resolutions (w_pix, h_pix), sensor size (w_mm, h_mm), and pixel aspect ratio (r)
        compatible by changing sensor height to w_mm * h_pix / w_pix / r

    Args:
        cam: Camera object
            bpy_types.Object
    """
    thisfunc = thisfile + '->correct_sensor_height()'

    # Camera
    sensor_width_mm = cam.data.sensor_width

    # Scene
    scene = bpy.context.scene
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    # Change sensor height
    sensor_height_mm = sensor_width_mm * h / w / pixel_aspect_ratio
    cam.data.sensor_height = sensor_height_mm

    logging.info("%s: Sensor height changed to %f", thisfunc, sensor_height_mm)


def get_camera_matrix(cam, keep_disparity=False):
    """
    Get camera matrix, intrinsics and extrinsics from Blender camera
        You can ask for a 4-by-4 projection that projects (x, y, z, 1) to
            (u, v, 1, d), where d is the disparity, reciprocal of depth

    Args:
        cam: Camera object
            bpy_types.Object
        keep_disparity: Whether matrices keep disparity or not
            Boolean
            Optional; defaults to False

    Returns:
        cam_mat: Camera matrix, product of intrinsics and extrinsics
            4-by-4 Matrix if 'keep_disparity'; else, 3-by-4
        int_mat: Camera intrinsics
            4-by-4 Matrix if 'keep_disparity'; else, 3-by-3
        ext_mat: Camera extrinsics
            4-by-4 Matrix if 'keep_disparity'; else, 3-by-4
    """
    thisfunc = thisfile + '->get_camera_matrix()'

    # Necessary scene update
    scene = bpy.context.scene
    scene.update()

    # Check if camera intrinsic parameters comptible with render settings
    if not intrinsics_compatible_with_scene(cam):
        raise ValueError(("Render settings and camera intrinsic parameters mismatch. "
                          "Such computed matrices will not make sense. Make them consistent first. "
                          "See error message from 'intrinsics_compatible_with_scene()' above for advice"))

    # Intrinsics

    f_mm = cam.data.lens
    sensor_width_mm = cam.data.sensor_width
    sensor_height_mm = cam.data.sensor_height
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if cam.data.sensor_fit == 'VERTICAL':
        # h times pixel height must fit into sensor_height_mm
        # w / pixel_aspect_ratio times pixel width will then fit into sensor_width_mm
        s_v = h * scale / sensor_height_mm
        s_u = w * scale / pixel_aspect_ratio / sensor_width_mm
    else: # 'HORIZONTAL' or 'AUTO'
        # w times pixel width must fit into sensor_width_mm
        # h * pixel_aspect_ratio times pixel height will then fit into sensor_height_mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = w * scale / sensor_width_mm
        s_v = h * scale * pixel_aspect_ratio / sensor_height_mm

    skew = 0 # only use rectangular pixels

    if keep_disparity:
        # 4-by-4
        int_mat = Matrix((
            (s_u * f_mm, skew, w * scale / 2, 0),
            (0, s_v * f_mm, h * scale / 2, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1)))
    else:
        # 3-by-3
        int_mat = Matrix((
            (s_u * f_mm, skew, w * scale / 2),
            (0, s_v * f_mm, h * scale / 2),
            (0, 0, 1)))

    # Extrinsics

    # Three coordinate systems involved:
    #   1. World coordinates: "world"
    #   2. Blender camera coordinates: "cam"
    #        - x is horizontal
    #        - y is up
    #        - right-handed: negative z is look-at direction
    #   3. Desired computer vision camera coordinates: "cv"
    #        - x is horizontal
    #        - y is down (to align to the actual pixel coordinates)
    #        - right-handed: positive z is look-at direction

    rotmat_cam2cv = Matrix((
        (1, 0, 0),
        (0, -1, 0),
        (0, 0, -1)))

    # matrix_world defines local-to-world transformation, i.e.,
    # where is local (x, y, z) in world coordinate system?
    t, rot_euler = cam.matrix_world.decompose()[0:2]

    # World to Blender camera
    rotmat_world2cam = rot_euler.to_matrix().transposed() # equivalent to inverse
    t_world2cam = rotmat_world2cam * -t

    # World to computer vision camera
    rotmat_world2cv = rotmat_cam2cv * rotmat_world2cam
    t_world2cv = rotmat_cam2cv * t_world2cam

    if keep_disparity:
        # 4-by-4
        ext_mat = Matrix((
            rotmat_world2cv[0][:] + (t_world2cv[0],),
            rotmat_world2cv[1][:] + (t_world2cv[1],),
            rotmat_world2cv[2][:] + (t_world2cv[2],),
            (0, 0, 0, 1)))
    else:
        # 3-by-4
        ext_mat = Matrix((
            rotmat_world2cv[0][:] + (t_world2cv[0],),
            rotmat_world2cv[1][:] + (t_world2cv[1],),
            rotmat_world2cv[2][:] + (t_world2cv[2],)))

    # Camera matrix
    cam_mat = int_mat * ext_mat

    logging.info("%s: Done computing camera matrix for '%s'", thisfunc, cam.name)
    logging.warning("%s:     ... using w = %d; h = %d", thisfunc, w * scale, h * scale)

    return cam_mat, int_mat, ext_mat


def get_camera_zbuffer(cam, save_to=None, hide=None):
    """
    Get z-buffer of Blender camera
        Values are z components in camera-centered coordinate system:
            - x is horizontal
            - y is down (to align with the actual pixel coordinates)
            - right-handed: positive z is look-at direction and means "in front of camera"
        Origin is camera center, not image plane (focal length away from origin)

    Args:
        cam: Camera object
            bpy_types.Object
        save_to: Path to which the .exr z-buffer will be saved
            String
            Optional; defaults to None (don't save)
        hide: Names of objects to be hidden while rendering this camera's z-buffer
            String or list thereof
            Optional; defaults to None

    Returns:
        zbuffer: Camera z-buffer
            2D numpy array
    """
    thisfunc = thisfile + '->get_camera_zbuffer()'

    # Validate and standardize error-prone inputs
    if hide is not None:
        if not isinstance(hide, list):
            # A single object
            hide = [hide]
        for element in hide:
            assert isinstance(element, str), \
                "'hide' should contain object names (i.e., strings), not objects themselves"

    if save_to is None:
        outpath = '/tmp/%s_zbuffer' % time()
    elif save_to.endswith('.exr'):
        outpath = save_to[:-4]

    # Duplicate scene to avoid touching the original scene
    bpy.ops.scene.new(type='LINK_OBJECTS')

    scene = bpy.context.scene
    scene.camera = cam
    scene.use_nodes = True
    node_tree = scene.node_tree
    nodes = node_tree.nodes

    # Remove all nodes
    for node in nodes:
        nodes.remove(node)

    # Set up nodes for z pass
    nodes.new('CompositorNodeRLayers')
    nodes.new('CompositorNodeOutputFile')
    node_tree.links.new(nodes['Render Layers'].outputs[2], nodes['File Output'].inputs[0])
    nodes['File Output'].format.file_format = 'OPEN_EXR'
    nodes['File Output'].format.color_mode = 'RGB'
    nodes['File Output'].format.color_depth = '32' # full float
    nodes['File Output'].base_path = dirname(outpath)
    nodes['File Output'].file_slots[0].path = basename(outpath)

    # Hide objects from z-buffer, if necessary
    if hide is not None:
        orig_hide_render = {} # for later restoration
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                orig_hide_render[obj.name] = obj.hide_render
                obj.hide_render = obj.name in hide

    # Render
    scene.cycles.samples = 1
    scene.render.filepath = '/tmp/%s_rgb.png' % time() # redirect RGB rendering to avoid overwritting
    bpy.ops.render.render(write_still=True)

    w = scene.render.resolution_x
    h = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.

    # Delete this new scene
    bpy.ops.scene.delete()

    # Restore objects' original render hide states, if necessary
    if hide is not None:
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.hide_render = orig_hide_render[obj.name]

    # Load z-buffer as array
    exr_path = outpath + '%04d' % scene.frame_current + '.exr'
    im = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    assert (np.array_equal(im[:, :, 0], im[:, :, 1]) and np.array_equal(im[:, :, 0], im[:, :, 2])), \
        "BGR channels of the z-buffer should be all the same, but they are not"
    zbuffer = im[:, :, 0]

    # Delete or move the .exr as user wants
    if save_to is None:
        # User doesn't want it -- delete
        remove(exr_path)
    else:
        # User wants it -- rename
        rename(exr_path, outpath + '.exr')

    logging.info("%s: Got z-buffer of camera '%s'", thisfunc, cam.name)
    logging.warning("%s:     ... using w = %d; h = %d", thisfunc, w * scale, h * scale)

    return zbuffer


def backproject_uv_to_3d(uvs, cam, obj_names=None, world_coords=False):
    """
    Backproject 2D coordinates to 3D
        Since a 2D point could be projected from any point on a 3D line, this function will return
            the 3D point at which this line (ray) intersects with an object for the first time

    Args:
        uvs: UV coordinates
            Array_like of length 2 or shape (n, 2)
            (0, 0)
            +------------> (w, 0)
            |           u
            |
            |
            |
            v v (0, h)
        cam: Camera object
            bpy_types.Object
        obj_names: Names of objects of interest
            String or list thereof
            Optional; defaults to None (all objects)
        world_coords: Whether to return world or local coordinates
            Boolean
            Optional; defaults to False

    Returns:
        xyzs: 3D local coordinates
            Vector or None (if no intersections) or list thereof
        intersect_objnames: Name(s) of object(s) responsible for intersections
            String or None (if no intersections) or list thereof
    """
    thisfunc = thisfile + '->backproject_uv_to_3d()'

    # Standardize inputs
    uvs = np.array(uvs).reshape(-1, 2)
    objs = bpy.data.objects
    if isinstance(obj_names, str):
        obj_names = [obj_names]
    elif obj_names is None:
        obj_names = [o.name for o in objs if o.type == 'MESH']

    scene = bpy.context.scene
    w, h = scene.render.resolution_x, scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.

    # Get 4-by-4 invertible camera matrix
    cam_mat, _, _ = get_camera_matrix(cam, keep_disparity=True)

    # Construct BVH trees for objects of interest
    trees = {}
    for obj_name in obj_names:
        obj = objs[obj_name]
        bm = xb_object.get_bmesh(obj)
        trees[obj_name] = BVHTree.FromBMesh(bm)

    xyzs = [None] * uvs.shape[0]
    intersect_objnames = [None] * uvs.shape[0]

    for i in range(uvs.shape[0]):

        # Compute the infinitely far point on the line passing camera center and projecting to uv
        uv = uvs[i, :]
        uv1d = np.append(uv, [1, 0])
        xyzw = cam_mat.inverted() * Vector(uv1d) # w = 0; world

        # Ray start and direction in world coordinates
        ray_start_world = cam.location # origin in camera coordinates
        ray_dir_world = 1e10 * Vector(xyzw[:3]) - ray_start_world # boost it for robust matrix multiplications

        first_intersect = None
        first_intersect_objname = None
        dist_min = np.inf

        # Test intersections with each object of interest
        for obj_name, tree in trees.items():
            obj2world = objs[obj_name].matrix_world
            world2obj = obj2world.inverted()

            # Ray start and direction in local coordinates
            ray_start = world2obj * ray_start_world
            ray_dir = world2obj * ray_dir_world

            # Ray tracing
            loc, _, _, dist = tree.ray_cast(ray_start, ray_dir)

            # See if this intersection is closer to camera center
            if (dist is not None) and (dist < dist_min):
                if world_coords:
                    first_intersect = obj2world * loc
                else:
                    first_intersect = loc
                first_intersect_objname = obj_name

        xyzs[i] = first_intersect
        intersect_objnames[i] = first_intersect_objname

    logging.info("%s: Backprojection done with camera '%s'", thisfunc, cam.name)
    logging.warning("%s:     ... using w = %d; h = %d", thisfunc, w * scale, h * scale)

    if uvs.shape[0] == 1:
        return xyzs[0], intersect_objnames[0]
    else:
        return xyzs, intersect_objnames


def get_visible_vertices(cam, obj, ignore_occlusion=False, perc_z_eps=1e-6, hide=None):
    """
    Get vertices that are visible (projected within frame AND unoccluded) from Blender camera
        Rasterized z-buffer (instead of ray tracing) used for speed
        Depth considered the same within a percentage window, so inaccurate when object's
            own depth variation is small compared with its overall depth
        Since z-buffer may cover other objects, this function takes occlusion by other objects into account,
            but you can opt to ignore z-buffer such that occluded vertices are also considered visible

    Args:
        cam: Camera object
            bpy_types.Object
        obj: Object of interest
            bpy_types.Object
        ignore_occlusion: Whether to ignore occlusion (including self-occlusion)
            Boolean
            Optional; defaults to False
        perc_z_eps: Threshold for percentage difference between the query z_q and buffered z_b
                z_q considered equal to z_b when abs(z_q - z_b) / z_b < perc_z_eps
            Float
            Optional; defaults to 1e-6
            Useless if ignore_occlusion
        hide: Names of objects to be hidden while rendering this camera's z-buffer
            String or list thereof
            Optional; defaults to None
            Useless if ignore_occlusion

    Returns:
        visible_vert_ind: Indices of vertices that are visible
            List of non-negative integers
    """
    thisfunc = thisfile + '->get_visible_vertices()'

    scene = bpy.context.scene
    w, h = scene.render.resolution_x, scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.

    # Get camera matrix
    cam_mat, _, ext_mat = get_camera_matrix(cam)

    # Get z-buffer
    if not ignore_occlusion:
        zbuffer = get_camera_zbuffer(cam, hide=hide)

    # Get mesh data from object
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    visible_vert_ind = []
    # For each of its vertices
    for bv in bm.verts:

        # Check if its projection falls inside frame
        v_world = obj.matrix_world * bv.co # local to world
        uv = np.array(cam_mat * v_world) # project to 2D
        uv = uv[:-1] / uv[-1]
        if uv[0] >= 0 and uv[0] < w * scale and uv[1] >= 0 and uv[1] < h * scale:
            # Yes

            if ignore_occlusion:
                # Considered visible already
                visible_vert_ind.append(bv.index)
            else:
                # Proceed to check occlusion with z-buffer
                v_cv = ext_mat * v_world # world to camera to CV
                z = v_cv[-1]
                z_min = zbuffer[int(uv[1]), int(uv[0])]
                if (z - z_min) / z_min < perc_z_eps:
                    visible_vert_ind.append(bv.index)

    logging.info("%s: Visibility test done with camera '%s'", thisfunc, cam.name)
    logging.warning("%s:     ... using w = %d; h = %d", thisfunc, w * scale, h * scale)

    return visible_vert_ind


def get_2d_bounding_box(obj, cam):
    """
    Get 2D bounding box of the object in the camera frame
        This is different from projecting the 3D bounding box to 2D

    Args:
        obj: Object of interest
            bpy_types.Object
        cam: Camera object
            bpy_types.Object

    Returns:
        corners: 2D coordinates of bounding box corners
            Numpy array of shape (4, 2); corners are ordered counterclockwise
            (0, 0)
            +------------> (w, 0)
            |           u
            |
            |
            |
            v v (0, h)
    """
    thisfunc = thisfile + '->get_2d_bounding_box()'

    scene = bpy.context.scene
    w, h = scene.render.resolution_x, scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.

    # Get camera matrix
    cam_mat, _, _ = get_camera_matrix(cam)

    # Project all vertices to 2D
    pts_2d = np.array([], dtype=float).reshape((0, 2))
    for v in obj.data.vertices:
        uv = np.array(cam_mat * obj.matrix_world * v.co) # project to 2D
        uv = uv[:-1] / uv[-1]
        pts_2d = np.vstack((pts_2d, uv))

    # Compute bounding box
    u_min, v_min = np.min(pts_2d, axis=0)
    u_max, v_max = np.max(pts_2d, axis=0)
    corners = np.vstack((
        np.array([u_min, v_min]),
        np.array([u_max, v_min]),
        np.array([u_max, v_max]),
        np.array([u_min, v_max])))

    logging.info("%s: Got 2D bounding box of '%s' in camera '%s'", thisfunc, obj.name, cam.name)
    logging.warning("%s:     ... using w = %d; h = %d", thisfunc, w * scale, h * scale)

    return corners
