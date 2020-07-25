"""
Utility functions for manipulating Blender lighting

Xiuming Zhang, MIT CSAIL
July 2017
"""

import logging
from os.path import abspath
import numpy as np
import bpy
from mathutils import Vector
import logging_colorer # noqa: F401 # pylint: disable=unused-import

logging.basicConfig(level=logging.INFO)
thisfile = abspath(__file__)


def point_light_to(light, target):
    """
    Point directional light to a target

    Args:
        light: Light object
            bpy_types.Object
        target: Target location to which light rays point
            3-tuple of floats
    """
    thisfunc = thisfile + '->point_light_to()'

    target = Vector(target)

    # Point it to target
    direction = target - light.location
    # Find quaternion that rotates lamp facing direction '-Z' so that it aligns with 'direction'
    # This rotation is not unique because the rotated lamp can still rotate about direction vector
    # Specifying 'Y' gives the rotation quaternion with lamp's 'Y' pointing up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    light.rotation_euler = rot_quat.to_euler()

    logging.info("%s: Lamp '%s' points to %s now", thisfunc, light.name, target)


def add_light_sun(xyz=(0, 0, 0), rot_vec_rad=(0, 0, 0), name=None, energy=1, size=0.1):
    """
    Add a sun lamp that emits parallel light rays

    Args:
        xyz: Location only used to compute light ray direction
            3-tuple of floats
            Optional; defaults to (0, 0, 0)
        rot_vec_rad: Rotation angle in radians around x, y and z
            3-tuple of floats
            Optional; defaults to (0, 0, 0)
        name: Light object name
            String
            Optional
        energy: Light intensity
            Float
            Optional; defaults to 1
        size: Light size for ray shadow tracing; larger for softer shadows
            Float
            Optional; defaults to 0.1

    Returns:
        sun: Handle of added light
            bpy_types.Object
    """
    thisfunc = thisfile + '->add_light_sun()'

    bpy.ops.object.lamp_add(type='SUN', location=xyz, rotation=rot_vec_rad)
    sun = bpy.context.active_object

    if name is not None:
        sun.name = name

    sun.data.shadow_soft_size = size # larger means softer shadows

    # Strength
    engine = bpy.data.scenes['Scene'].render.engine
    if engine == 'CYCLES':
        sun.data.node_tree.nodes['Emission'].inputs[1].default_value = energy
    else:
        raise NotImplementedError(engine)

    logging.info("%s: Sun lamp (parallel light) added", thisfunc)

    return sun


def add_light_area(xyz=(0, 0, 0), rot_vec_rad=(0, 0, 0), name=None, energy=100, size=0.1):
    """
    Add area light that emits light rays the lambertian way

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
        energy: Light intensity
            Float
            Optional; defaults to 100
        size: Light size for ray shadow tracing; larger for softer shadows
            Float
            Optional; defaults to 0.1

    Returns:
        area: Handle of added light
            bpy_types.Object
    """
    thisfunc = thisfile + '->add_light_area()'

    if (np.abs(rot_vec_rad) > 2 * np.pi).any():
        logging.warning("%s: Some input value falls outside [-2pi, 2pi]. Sure inputs are in radians?", thisfunc)

    bpy.ops.object.lamp_add(type='AREA', location=xyz, rotation=rot_vec_rad)
    area = bpy.context.active_object

    if name is not None:
        area.name = name

    area.data.size = size # larger means softer shadows

    # Strength
    engine = bpy.data.scenes['Scene'].render.engine
    if engine == 'CYCLES':
        area.data.node_tree.nodes['Emission'].inputs[1].default_value = energy
    else:
        raise NotImplementedError(engine)

    logging.info("%s: Area light added", thisfunc)

    return area


def add_light_point(xyz=(0, 0, 0), name=None, energy=100):
    """
    Add omnidirectional point lamp

    Args:
        xyz: Location
            3-tuple of floats
            Optional; defaults to (0, 0, 0)
        name: Light object name
            String
            Optional
        energy: Light intensity
            Float
            Optional; defaults to 100

    Returns:
        point: Handle of added light
            bpy_types.Object
    """
    thisfunc = thisfile + '->add_light_point()'

    bpy.ops.object.lamp_add(type='POINT', location=xyz)
    point = bpy.context.active_object

    if name is not None:
        point.name = name

    point.data.shadow_soft_size = 0 # hard shadows

    # Strength
    engine = bpy.data.scenes['Scene'].render.engine
    if engine == 'CYCLES':
        point.data.node_tree.nodes['Emission'].inputs[1].default_value = energy
    else:
        raise NotImplementedError(engine)

    logging.info("%s: Omnidirectional point light added", thisfunc)

    return point
