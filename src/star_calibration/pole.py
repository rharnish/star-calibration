"""Where is the celestial pole, from the shape of the trails alone -- no star named.

`solve.py` starts from a coarse grid over d_az/d_pitch/d_roll, +-30/15/10 deg around the
published pose, scored by how many *bright catalog stars* land within 20 px of some track.
Two failures follow from that. A camera whose published azimuth is wrong by more than 30 deg
is outside the box and can never be found (bm-s-mobo-c has 35 clean arcs and still failed).
And the score needs identified bright stars, so it is blind on a night whose arcs are real
but faint.

This module removes both limits by using a fact the grid never exploits: *every* track,
named or not, bright or faint, is carried by the same rigid rotation of the sky. A star's
direction d obeys

    d_dot = omega * (p x d)

with omega the sidereal rate (known, 7.2921e-5 rad/s) and p the pole. Map each track point
into the *camera's own frame* through the lens alone -- which needs no pose, since pose is
exactly the rotation between camera frame and world -- and that equation becomes linear in
p. Least squares over every sample of every track then gives the pole in camera coordinates
in closed form, with no search at all.

Knowing the pole fixes two of the three pose angles. What is left is the rotation about the
polar axis itself, which the flow field cannot see (turning the sky about the pole leaves
every velocity unchanged); that one angle is a 1-D search, and *that* is where naming stars
earns its keep. So the 3-D grid becomes closed form + a 1-D scan, and the search stops being
local to the published pose.

Two numbers come out of the fit for free and are worth as much as the pole itself:
|p| should be 1, and the residuals are in rad/s against a known rate. A field of jagged
noise tracks (the lit-cloud failures, marconi-n-mobo-c and starr-n-mobo-c) has no consistent
rotation in it at all and shows up as |p| far from 1 with huge residuals -- so this also
tells an overcast night from a solvable one before any solving is attempted.
"""
from __future__ import annotations

import math

import numpy as np

OMEGA = 7.2921159e-5      # sidereal rotation, rad/s
_THETA_TAB = np.linspace(0.0, math.pi, 4096)


def pixel_to_cam(x: np.ndarray, y: np.ndarray, W: int, H: int, k: float, k1: float) -> np.ndarray:
    """Pixels -> unit vectors in the camera's own (right, up, boresight) frame.

    The inverse of `fisheye.project_fisheye`'s lens stage, and pose-free by construction:
    r = k*theta*(1 + k1*theta^2) is inverted against a dense monotone table, and the bearing
    within the image plane comes straight from the pixel offset.
    """
    dx = np.asarray(x, float) - 0.5 * W
    dy = np.asarray(y, float) - 0.5 * H
    r = np.hypot(dx, dy)
    r_tab = k * _THETA_TAB * (1.0 + k1 * _THETA_TAB ** 2)
    if np.any(np.diff(r_tab) <= 0):           # k1 strong enough to fold the table over
        good = np.r_[True, np.diff(r_tab) > 0]
        theta = np.interp(r, r_tab[good], _THETA_TAB[good])
    else:
        theta = np.interp(r, r_tab, _THETA_TAB)
    safe = np.where(r > 1e-9, r, 1.0)
    ux, uy = dx / safe, -dy / safe             # image y grows downward; "up" decreases y
    s = np.sin(theta)
    return np.stack([s * ux, s * uy, np.cos(theta)], axis=-1)


def samples(tracks: list[dict], W: int, H: int, k: float, k1: float, t0_step: float = 1.0):
    """(d, d_dot) pairs from consecutive points of every track, in camera coordinates.

    Consecutive *observed* offsets are differenced, so a track with a gap contributes a
    correspondingly longer baseline rather than a wrong rate. Offsets are seconds.
    """
    D, V = [], []
    for t in tracks:
        offs = sorted(t)
        if len(offs) < 2:
            continue
        xy = np.array([t[o][:2] for o in offs], float)
        d = pixel_to_cam(xy[:, 0], xy[:, 1], W, H, k, k1)
        dt = np.diff(np.asarray(offs, float)) * t0_step
        ok = dt > 0
        D.append(0.5 * (d[:-1] + d[1:])[ok])
        V.append((np.diff(d, axis=0)[ok] / dt[ok, None]))
    if not D:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(D), np.concatenate(V)


def _skew(d: np.ndarray) -> np.ndarray:
    """Stack of [d]_x matrices so that [d]_x @ p == d x p."""
    Z = np.zeros(len(d))
    return np.array([[Z, -d[:, 2], d[:, 1]],
                     [d[:, 2], Z, -d[:, 0]],
                     [-d[:, 1], d[:, 0], Z]]).transpose(2, 0, 1)


# The (right, up, boresight) basis is left-handed, and a cross product's components pick up
# a sign under a reflection: (Ja) x (Jb) = -J(a x b). So the textbook d_dot = OMEGA*(p x d)
# appears with a minus in these coordinates. Measured, not assumed -- fitting both signs
# against the 86 known-good solves puts this one at 0.29 deg median pole error and the other
# at 179.7 deg, i.e. exactly antipodal, as a pure sign error must be.
HANDEDNESS = -1


def fit_pole(D: np.ndarray, V: np.ndarray, sign: int = HANDEDNESS, iters: int = 8):
    """Least-squares pole in camera coordinates, with iterative outlier rejection.

    Solves V = sign*OMEGA*(p x D) for p. Reweighting is a hard trim rather than a soft loss:
    a noise track's velocity is unrelated to any rotation, so there is nothing to be gained
    by letting it pull on the fit at reduced weight.
    """
    if len(D) < 12:
        return None
    A = sign * OMEGA * _skew(D)               # (n, 3, 3), rows of the linear system
    w = np.ones(len(D))
    p = np.zeros(3)
    for it in range(iters):
        M = (A * w[:, None, None]).reshape(-1, 3)
        b = (V * w[:, None]).reshape(-1)
        p, *_ = np.linalg.lstsq(M, b, rcond=None)
        res = np.linalg.norm(V - (A @ p), axis=1)
        keep = w > 0
        if keep.sum() < 12:
            break
        scale = max(np.median(res[keep]), 1e-9)
        w = (res < max(3.0 * scale, 0.15 * OMEGA)).astype(float)
        if w.sum() < 12:
            w = (res <= np.partition(res, 11)[11]).astype(float)
    inl = w > 0
    res = np.linalg.norm(V - (A @ p), axis=1)
    return {"p": p, "norm": float(np.linalg.norm(p)),
            "p_hat": p / max(np.linalg.norm(p), 1e-12),
            "n": int(len(D)), "n_inliers": int(inl.sum()),
            "inlier_frac": float(inl.mean()),
            "median_res_rel": float(np.median(res[inl]) / OMEGA) if inl.any() else float("inf"),
            "inliers": inl}


def estimate(tracks: list[dict], W: int, H: int, k: float, k1: float):
    """The pole in camera coordinates from a sequence's tracks, sign resolved."""
    D, V = samples(tracks, W, H, k, k1)
    f = fit_pole(D, V)
    if f is None:
        return None
    # |p| is the headline quality number: the fit is told the sidereal rate, so a real star
    # field returns |p| within a percent of 1, and a field with no coherent rotation in it
    # cannot. Combined with the inlier fraction it separates a solvable night from an
    # overcast one before any star is named.
    f["quality"] = f["inlier_frac"] / (1.0 + abs(f["norm"] - 1.0) + f["median_res_rel"])
    return f


def cam_from_pole(p_cam: np.ndarray, lat: float, psi_deg: np.ndarray):
    """Camera-frame axes for every pose consistent with the pole landing at `p_cam`.

    The pole in world coordinates is (az 0, alt = latitude). Any rotation M taking that to
    `p_cam` differs from any other by a turn about the polar axis, so the whole consistent
    family is M(psi) = M0 @ Rot(p_world, psi) -- one free angle, which is what the caller
    scans. Returns (n, 3, 3) with rows (right, up, boresight) in world coordinates.
    """
    la = math.radians(lat)
    p_w = np.array([0.0, math.cos(la), math.sin(la)])      # east, north, up
    p_c = np.asarray(p_cam, float)
    p_c = p_c / np.linalg.norm(p_c)

    # (right, up, boresight) is LEFT-handed: right x up = -boresight, so the basis matrix has
    # determinant -1 and is not a rotation. Flipping the third row makes it one, and the whole
    # argument above ("two such rotations differ by a turn about the polar axis") only holds
    # for rotations. Work in the flipped frame and flip back at the end.
    J = np.diag([1.0, 1.0, -1.0])
    p_c = J @ p_c

    v = np.cross(p_w, p_c)
    c = float(np.dot(p_w, p_c))
    if np.linalg.norm(v) < 1e-12:
        M0 = np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(p_w, p_w)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        M0 = np.eye(3) + vx + vx @ vx / (1 + c)

    psi = np.radians(np.atleast_1d(psi_deg))
    K = np.array([[0, -p_w[2], p_w[1]], [p_w[2], 0, -p_w[0]], [-p_w[1], p_w[0], 0]])
    R = (np.eye(3)[None] + np.sin(psi)[:, None, None] * K[None]
         + (1 - np.cos(psi))[:, None, None] * (K @ K)[None])
    return J[None] @ (M0[None] @ R)


def pose_from_axes(M: np.ndarray, cam: dict):
    """(d_az, d_pitch, d_roll) in degrees for camera-frame axes M, relative to cams.json.

    Inverts the basis construction in `fisheye.project_fisheye`: rows of M are
    (right, up, boresight) as world vectors.
    """
    M = np.atleast_3d(M).reshape(-1, 3, 3)
    b = M[:, 2, :]
    el0 = np.degrees(np.arcsin(np.clip(b[:, 2], -1, 1)))
    az0 = np.degrees(np.arctan2(b[:, 0], b[:, 1])) % 360
    up_w = np.array([0.0, 0.0, 1.0])
    r0 = np.cross(b, up_w)
    n = np.linalg.norm(r0, axis=1, keepdims=True)
    r0 = r0 / np.where(n > 1e-12, n, 1.0)
    u0 = np.cross(r0, b)
    right = M[:, 0, :]
    roll = np.degrees(np.arctan2(np.einsum("ij,ij->i", right, u0),
                                 np.einsum("ij,ij->i", right, r0)))
    d_az = (az0 - cam["az"] - cam.get("yaw", 0.0) + 180) % 360 - 180
    d_pitch = el0 - (cam.get("pitch") or 0.0)
    d_roll = (roll - (cam.get("roll") or 0.0) + 180) % 360 - 180
    return np.stack([d_az, d_pitch, d_roll], axis=1)


def axes_from_pose(cam: dict, d_az: float, d_pitch: float, d_roll: float) -> np.ndarray:
    """Rows (right, up, boresight) for a pose -- the basis `project_fisheye` builds, exposed
    so `pose_from_axes` can be round-tripped and a known solve can be checked against a fit."""
    az0 = math.radians(cam["az"] + cam.get("yaw", 0.0) + d_az)
    el0 = math.radians((cam.get("pitch") or 0.0) + d_pitch)
    b = np.array([math.cos(el0) * math.sin(az0), math.cos(el0) * math.cos(az0), math.sin(el0)])
    r0 = np.cross(b, [0.0, 0.0, 1.0])
    r0 /= np.linalg.norm(r0)
    u0 = np.cross(r0, b)
    roll = math.radians((cam.get("roll") or 0.0) + d_roll)
    return np.array([r0 * math.cos(roll) + u0 * math.sin(roll),
                     -r0 * math.sin(roll) + u0 * math.cos(roll), b])


def pole_in_cam(cam: dict, d_az: float, d_pitch: float, d_roll: float) -> np.ndarray:
    """Where a known pose puts the celestial pole in camera coordinates -- the fit's target."""
    la = math.radians(cam["lat"])
    return axes_from_pose(cam, d_az, d_pitch, d_roll) @ np.array(
        [0.0, math.cos(la), math.sin(la)])


def lens_scale(tracks: list[dict], W: int, H: int, k0: float, k1: float,
               lo: float = 0.55, hi: float = 1.45, tol: float = 1e-3):
    """The lens scale this night's trails imply, from |p| alone -- no star identified.

    `fit_pole` is told the sidereal rate, so |p| is not free: it comes out at 1 only when the
    pixel-to-angle map is right. Too large a k makes every angle too small, the apparent rate
    too slow, and |p| less than 1. So |p| is a *measurement of the lens*, from the same
    trails, before anything is matched to a catalog.

    That matters because `solve.py` holds k at 0.886x nameplate for every camera, which the
    2026-09-13 batch established from nine cameras that happened to solve. Cameras outside
    that group are not obliged to agree, and the ones that failed mostly do not: all three
    Big Black Mountain units want ~0.80x and 69bravo-w ~0.78x, which is a 10% radial error --
    150 px out at the edge of the frame, far past any matching tolerance, and enough on its
    own to explain why no star was ever found.

    Scans coarsely for the bracket, then bisects on |p| - 1, which is monotone decreasing in k.
    """
    grid = np.arange(lo, hi + 1e-9, 0.02)
    norms = []
    for kr in grid:
        f = estimate(tracks, W, H, kr * k0, k1)
        norms.append(f["norm"] if f else np.nan)
    norms = np.array(norms)
    if not np.any(np.isfinite(norms)):
        return None
    i = int(np.nanargmin(np.abs(norms - 1.0)))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    for _ in range(24):
        if b - a < tol:
            break
        m = 0.5 * (a + b)
        f = estimate(tracks, W, H, m * k0, k1)
        if f is None:
            break
        if f["norm"] > 1.0:
            a = m
        else:
            b = m
    kr = 0.5 * (a + b)
    f = estimate(tracks, W, H, kr * k0, k1)
    return {"k_ratio": float(kr), "norm": f["norm"], "fit": f} if f else None
