import numpy as np

from sceneforge.core import geom, scene_frame


def test_model_input_size_multiple_of_patch():
    w, h = geom.model_input_size(1920, 1080, 518)
    assert (w, h) == (518, 294)
    assert w % 14 == 0 and h % 14 == 0


def test_k_roundtrip_full_to_model():
    w0, h0 = 1920, 1080
    w, h = geom.model_input_size(w0, h0, 518)
    p = geom.resize_crop_params(w0, h0, w, h)
    K_full = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    s = p["scale"]
    K_model = K_full.copy()
    K_model[:2] *= s
    K_model[0, 2] -= p["crop_x"]
    K_model[1, 2] -= p["crop_y"]
    np.testing.assert_allclose(geom.k_model_to_full(K_model, p), K_full, atol=1e-9)


def test_backproject_project_roundtrip():
    rng = np.random.default_rng(0)
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    c2w = np.eye(4)
    c2w[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0] * np.sign(np.linalg.det(np.linalg.qr(rng.normal(size=(3, 3)))[0]))
    c2w[:3, 3] = [1, 2, 3]
    if np.linalg.det(c2w[:3, :3]) < 0:
        c2w[:3, 0] *= -1
    depth = rng.uniform(1, 5, size=(240, 320))
    pts = geom.backproject(depth, K, c2w)
    uv, z = geom.project(pts.reshape(-1, 3), K, c2w)
    np.testing.assert_allclose(z, depth.reshape(-1), rtol=1e-9)
    u, v = np.meshgrid(np.arange(320) + 0.5, np.arange(240) + 0.5)
    np.testing.assert_allclose(uv[:, 0], u.reshape(-1), atol=1e-6)


def test_opencv_to_blender_flips_axes():
    c2w = np.eye(4)
    b = geom.opencv_to_blender_cam(c2w)
    # OpenCV forward (+Z) must equal Blender -Z of the camera
    assert np.allclose(c2w[:3, 2], -b[:3, 2])
    assert np.allclose(c2w[:3, 1], -b[:3, 1])


def test_manhattan_yaw_recovers_rotation():
    up = np.array([0, 0, 1.0])
    yaw_true = np.deg2rad(25)
    normals = []
    for k in range(4):
        a = yaw_true + k * np.pi / 2
        normals += [[np.cos(a), np.sin(a), 0.0]] * 100
    yaw = scene_frame.manhattan_yaw(np.array(normals), up)
    R = scene_frame.rotation_from_up_yaw(up, yaw)
    # scene x axis must align with one of the wall normals (mod 90°)
    ang = np.degrees(np.arctan2(R[0, 1], R[0, 0])) % 90
    assert min(abs(ang - 25), abs(ang - 25 - 90)) < 1.0


def test_combine_anchors_weighted_median():
    ms = [scene_frame.AnchorMeasurement("d", "door", "height", 1.0, 2.03, 0.07),
          scene_frame.AnchorMeasurement("t", "desk", "top_height", 0.37, 0.75, 0.05)]
    s, spread = scene_frame.combine_anchors(ms)
    assert 2.0 <= s <= 2.03 + 1e-9
    assert spread < 0.02
