"""Equidistant fisheye projection: the lens model the HPWREN Mobotix cameras follow.

A rectilinear model (pixel offset proportional to tan(angle from boresight)) is a fair local
approximation near the image centre, but it is not what these lenses do, and the failure is
not subtle. Projected rectilinearly, Orion's belt at 52 deg elevation -- well inside a
published 90 deg field of view -- lands 1000+ px above the top of a 2048 px frame, for a star
plainly visible mid-frame in the real image (plume-triangulation's NOTES.md, "Night sky
(stars)"). tan() diverges as its argument approaches 90 deg; a fisheye lens is built so that
light from those angles still lands on the sensor, via a mapping that stays finite (and
roughly linear in angle) out to and past the edge of the field. The model here is pixel
radius r = k * theta * (1 + k1 * theta^2): equidistant, plus one radial term.

The star solves put every 90 deg, 3072 px unit on nearly the same lens, K_RATIO and K1 below
(k = K_RATIO times the nameplate scale, `initial_k`). They are the starting lens for a solve
and the shared lens for a camera that has none of its own.

Geometry: build the camera's boresight as a unit vector from (az, pitch), then a local
(right, up) image-plane basis perpendicular to it, rotated by roll. Any target (az, el) is
also a unit vector; theta is the angle between it and the boresight (via the dot product,
valid at any angle, unlike a tangent-plane difference), and phi is its bearing within the
local basis. Pixel offset is (k*theta) in the direction of phi -- polar coordinates in the
image plane, not a small-angle tangent-plane approximation.

The optical centre is where the boresight lands. It defaults to the frame's middle; a camera
record with `cx`, `cy` (pixels right and down from the middle) moves it there. HPWREN's
units sit a median 32 px off (hpwren/intrinsics.json).
"""
from __future__ import annotations

import math

import numpy as np

# The shared lens of the 90 deg, 3072 px units: 0.886 of the nameplate scale, k1 -0.078. All
# nine of the first lens-free solves agreed on it (k 0.882-0.890, k1 -0.074 to -0.084;
# plume-triangulation's NOTES.md, 2026-09-13), and 79 cameras since sit at 0.877-0.894.
K_RATIO, K1 = 0.886, -0.078

# No direction further than this from the boresight is in the picture: even the 180 deg units
# see only 90 deg off-axis. Past it `project_fisheye` returns NaN, and sooner if the lens
# polynomial folds first (below).
MAX_THETA_DEG = 100.0


def theta_limit_deg(k1: float) -> float:
    """How far off the boresight `project_fisheye` projects. With k1 < 0 the radius
    k*theta*(1 + k1*theta^2) peaks at theta = sqrt(-1/(3 k1)) (~118 deg at K1) and then
    shrinks, so a star behind the camera would land back in the frame: up to 4-5 phantom
    bright stars per pose in the wide search's coarse scan."""
    return min(MAX_THETA_DEG, math.degrees(math.sqrt(-1 / (3 * k1)))) if k1 < 0 else MAX_THETA_DEG


def _unit(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.sin(az),   # east
                      np.cos(el) * np.cos(az),   # north
                      np.sin(el)], axis=-1)      # up


def initial_k(cam: dict, width: int) -> float:
    """px per radian, calibrated so the published horizontal FOV reaches the frame edge."""
    return (width / 2) / math.radians(cam["fov"] / 2.0)


def _basis(cam: dict, d_az: float, d_pitch: float, d_roll: float):
    """Boresight, right and up unit vectors (east, north, up) for the camera's solved pose."""
    az0 = cam["az"] + cam.get("yaw", 0.0) + d_az
    el0 = (cam.get("pitch") or 0.0) + d_pitch
    boresight = _unit(az0, el0)

    world_up = np.array([0.0, 0.0, 1.0])
    right0 = np.cross(boresight, world_up)  # facing north -> right = east, not west
    right0 /= np.linalg.norm(right0)
    up0 = np.cross(right0, boresight)       # right x forward = up, not forward x right

    roll = math.radians((cam.get("roll") or 0.0) + d_roll)
    right = right0 * math.cos(roll) + up0 * math.sin(roll)
    up = -right0 * math.sin(roll) + up0 * math.cos(roll)
    return boresight, right, up


def project_fisheye(cam: dict, az_deg: np.ndarray, elev_deg: np.ndarray, width: int, height: int,
                     d_az: float = 0.0, d_pitch: float = 0.0, d_roll: float = 0.0,
                     k_scale: float | None = None, k1: float = 0.0):
    """Equidistant-fisheye (az, el) -> fractional image coords. Mirrors terrain.project()'s
    signature (minus cam-table pitch/roll, folded into d_pitch/d_roll by the caller) so the
    two models are interchangeable in fit/plot code. NaN for a direction further off the
    boresight than `theta_limit_deg(k1)`: it isn't in the picture, and the fold would put it
    there."""
    if k_scale is None:
        k_scale = initial_k(cam, width)
    boresight, right, up = _basis(cam, d_az, d_pitch, d_roll)

    t = _unit(np.asarray(az_deg, float), np.asarray(elev_deg, float))
    cos_theta = np.clip(t @ boresight, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    t_perp_x = t @ right
    t_perp_y = t @ up
    sin_theta = np.sin(theta)
    safe = np.where(sin_theta > 1e-9, sin_theta, 1.0)
    ux, uy = t_perp_x / safe, t_perp_y / safe  # unit bearing direction, well-defined at theta=0

    r = k_scale * theta * (1.0 + k1 * theta ** 2)
    dx_px = r * ux + cam.get("cx", 0.0)
    dy_px = -r * uy + cam.get("cy", 0.0)  # image y grows downward; "up" should decrease y

    out = theta > math.radians(theta_limit_deg(k1))
    return (np.where(out, np.nan, 0.5 + dx_px / width),
            np.where(out, np.nan, 0.5 + dy_px / height))


def unproject_fisheye(cam: dict, x_frac, y_frac, width: int, height: int,
                      d_az: float = 0.0, d_pitch: float = 0.0, d_roll: float = 0.0,
                      k_scale: float | None = None, k1: float = 0.0):
    """Fractional image coords -> (az, el) in degrees: the exact inverse of `project_fisheye`.

    The pixel's radius from the centre gives the angle off the boresight (inverting
    r = k*theta*(1 + k1*theta^2) by Newton), its direction the bearing within the image
    plane; the pose's basis turns that back into a direction in the world. With the camera
    pitched or rolled, a pixel's azimuth depends on its row, which reading along the middle
    row (`geom.offset_bearing_deg`) cannot see.
    """
    if k_scale is None:
        k_scale = initial_k(cam, width)
    boresight, right, up = _basis(cam, d_az, d_pitch, d_roll)
    dx = (np.asarray(x_frac, float) - 0.5) * width - cam.get("cx", 0.0)
    dy = (np.asarray(y_frac, float) - 0.5) * height - cam.get("cy", 0.0)
    r = np.hypot(dx, dy)
    theta = r / k_scale
    for _ in range(30):
        f = k_scale * theta * (1.0 + k1 * theta ** 2) - r
        theta = theta - f / (k_scale * (1.0 + 3.0 * k1 * theta ** 2))
    safe = np.where(r > 1e-9, r, 1.0)
    ux, uy = dx / safe, -dy / safe
    st, ct = np.sin(theta)[..., None], np.cos(theta)[..., None]
    t = ct * boresight + st * (ux[..., None] * right + uy[..., None] * up)
    az = np.degrees(np.arctan2(t[..., 0], t[..., 1])) % 360.0
    el = np.degrees(np.arcsin(np.clip(t[..., 2], -1.0, 1.0)))
    return az, el


if __name__ == "__main__":
    from star_calibration import catalog as SG
    from star_calibration.hpwren import cameras

    cam = cameras()["hp-s-mobo-c"]
    W, H = 3072, 2048
    alt, az = SG.stars_altaz(SG.ORION, 1729513594, cam["lat"], cam["lon"])
    x, y = project_fisheye(cam, az, alt, W, H)
    for name, xi, yi in zip(SG.ORION, x * W, y * H):
        onframe = 0 <= xi < W and 0 <= yi < H
        print(f"{name:12s} px=({xi:7.1f},{yi:7.1f})  {'ON' if onframe else 'off'}")
