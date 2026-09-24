"""Blender (5.2) headless script: build a preview .blend from s2_segment outputs.

Run:  Blender --background --factory-startup --python preview_scene.py -- --segment <dir> --out <file.blend>

Contents (all in meters, Z up):
  Collection "SF_Points"      fused point cloud (Geometry Nodes Mesh to Points, emission color)
  Collection "SF_Cameras"     one camera per keyframe (sub-collection per shot), frame as background image
  Collection "SF_Objects"     wireframe upright boxes per detected object (custom props: label, quality...)
  Collection "SF_Room"        wireframe room box
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # must precede bmesh when running as the pip `bpy` module
import bmesh
import numpy as np
from mathutils import Matrix


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--point-radius", type=float, default=0.006)
    ap.add_argument("--max-points", type=int, default=1_500_000)
    return ap.parse_args(argv)


def collection(name: str, parent: bpy.types.Collection | None = None) -> bpy.types.Collection:
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    return col


def points_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("SF_PointColor")
    if mat.node_tree is None:  # Blender < 5 needs use_nodes; 5.x always has a node tree
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def points_node_group(radius: float, mat: bpy.types.Material) -> bpy.types.NodeTree:
    ng = bpy.data.node_groups.new("SF_PointsView", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nin = ng.nodes.new("NodeGroupInput")
    nout = ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    m2p.inputs["Radius"].default_value = radius
    setmat = ng.nodes.new("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = mat
    ng.links.new(nin.outputs["Geometry"], m2p.inputs["Mesh"])
    ng.links.new(m2p.outputs["Points"], setmat.inputs["Geometry"])
    ng.links.new(setmat.outputs["Geometry"], nout.inputs["Geometry"])
    return ng


def build_points(seg: Path, col: bpy.types.Collection, radius: float, max_points: int) -> bpy.types.Object:
    d = np.load(seg / "points_scene.npz")
    xyz, rgb = d["xyz"].astype(np.float32), d["rgb"].astype(np.float32) / 255.0
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    me = bpy.data.meshes.new("SF_Points")
    me.vertices.add(len(xyz))
    me.vertices.foreach_set("co", xyz.ravel())
    ca = me.color_attributes.new("Col", "BYTE_COLOR", "POINT")
    rgba = np.concatenate([rgb, np.ones((len(rgb), 1), np.float32)], axis=1)
    ca.data.foreach_set("color", rgba.ravel())
    me.color_attributes.active_color = ca
    me.update()
    obj = bpy.data.objects.new("SF_Points", me)
    col.objects.link(obj)
    mod = obj.modifiers.new("PointsView", "NODES")
    mod.node_group = points_node_group(radius, points_material())
    return obj


def build_cameras(seg: Path, col: bpy.types.Collection) -> list[bpy.types.Object]:
    data = json.loads((seg / "cameras_scene.json").read_text())
    cams = []
    shot_cols: dict[str, bpy.types.Collection] = {}
    for f in data["frames"]:
        W, H = int(f["width"]), int(f["height"])
        K = f["K_full"]
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        cd = bpy.data.cameras.new(f"SF_Cam_{f['frame_id']}")
        cd.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"
        cd.sensor_width = 36.0
        cd.sensor_height = 36.0
        big = max(W, H)
        cd.lens = (fx if W >= H else fy) * 36.0 / big
        cd.shift_x = (W / 2.0 - cx) / big
        cd.shift_y = (cy - H / 2.0) / big
        cd.display_size = 0.15
        cd.clip_start = 0.01
        cd.clip_end = 200.0
        cd.show_background_images = True
        bg = cd.background_images.new()
        bg.image = bpy.data.images.load(f["image"], check_existing=True)
        bg.alpha = 0.6
        obj = bpy.data.objects.new(f"SF_Cam_{f['frame_id']}", cd)
        obj.matrix_world = Matrix(f["matrix_world_blender"])
        obj["frame_id"] = f["frame_id"]
        obj["shot_id"] = f["shot_id"]
        obj["time_s"] = f["time_s"]
        obj["fy_over_fx"] = fy / fx
        sc = shot_cols.get(f["shot_id"]) or collection(f"SF_Shot_{f['shot_id']}", col)
        shot_cols[f["shot_id"]] = sc
        sc.objects.link(obj)
        cams.append(obj)
    if data["frames"]:
        scn = bpy.context.scene
        scn.render.resolution_x = int(data["frames"][0]["width"])
        scn.render.resolution_y = int(data["frames"][0]["height"])
        scn.render.resolution_percentage = 100
        scn.camera = cams[0]
    return cams


def box_object(name: str, center, size, yaw: float, col: bpy.types.Collection) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(me)
    bm.free()
    me.update()
    obj = bpy.data.objects.new(name, me)
    obj.location = center
    obj.rotation_euler = (0.0, 0.0, yaw)
    obj.scale = [max(s, 1e-3) for s in size]
    obj.display_type = "WIRE"
    obj.hide_render = True
    col.objects.link(obj)
    return obj


def main() -> None:
    args = parse_args()
    seg = Path(args.segment)
    print("Blender", bpy.app.version_string)
    scn = bpy.context.scene
    for o in list(bpy.data.objects):  # factory startup scene: cube, light, camera
        bpy.data.objects.remove(o)
    scn.unit_settings.system = "METRIC"
    scn.unit_settings.scale_length = 1.0
    scn.view_settings.view_transform = "Standard"  # show frame colors as-is (default AgX washes them out)

    pts = build_points(seg, collection("SF_Points"), args.point_radius, args.max_points)
    cams = build_cameras(seg, collection("SF_Cameras"))

    ocol = collection("SF_Objects")
    objects = json.loads((seg / "objects.json").read_text())["objects"]
    for o in objects:
        b = o["bbox"]
        obj = box_object(f"SF_{o['id']}", b["center"], b["size"], b["yaw"], ocol)
        for k in ("label", "category", "source", "coverage", "quality", "n_views", "best_frame"):
            obj[k] = o[k]
        obj.color = (0.1, 0.8, 0.1, 1.0) if o["coverage"] == "good" else (0.9, 0.2, 0.1, 1.0)

    room = json.loads((seg / "room.json").read_text())
    lo, hi = np.array(room["min"]), np.array(room["max"])
    robj = box_object("SF_RoomBox", ((lo + hi) / 2).tolist(), (hi - lo).tolist(), 0.0, collection("SF_Room"))
    for face, info in room["faces"].items():
        robj[f"face_{face}"] = info["source"]

    frame = json.loads((seg / "scene_frame.json").read_text())
    scn["sceneforge_scale"] = frame["scale"]
    scn["sceneforge_scale_source"] = frame["scale_source"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"saved {out}: {len(pts.data.vertices)} points, {len(cams)} cameras, {len(objects)} objects")


if __name__ == "__main__":
    main()
