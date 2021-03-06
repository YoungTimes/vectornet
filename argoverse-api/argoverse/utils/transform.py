# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Utility functions for converting quaternions to 3d rotation matrices.

Unit quaternions are a way to compactly represent 3D rotations
while avoiding singularities or discontinuities (e.g. gimbal lock).

If a quaternion is not normalized beforehand to be unit-length, we will
re-normalize it on the fly.
"""

import logging

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def quat2rotmat(q: np.ndarray) -> np.ndarray:
    """Normalizes a quaternion to unit-length, then converts it into a rotation matrix.

    Note that libraries such as Scipy expect a quaternion in scalar-last [x, y, z, w] format,
    whereas at Argo we work with scalar-first [w, x, y, z] format, so we convert between the
    two formats here. We use the [w, x, y, z] order because this corresponds to the
    multidimensional complex number `w + ix + jy + kz`.

    Args:
        q: Array of shape (4,) representing (w, x, y, z) coordinates

    Returns:
        R: Array of shape (3, 3) representing a rotation matrix.
    """
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0, atol=1e-12):
        logger.info("Forced to re-normalize quaternion, since its norm was not equal to 1.")
        if np.isclose(norm, 0.0):
            raise ZeroDivisionError("Normalize quaternioning with norm=0 would lead to division by zero.")
        q /= norm

    quat_xyzw = quat_argo2scipy(q)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def quat_argo2scipy(q: np.ndarray) -> np.ndarray:
    """Re-order Argoverse's scalar-first [w,x,y,z] quaternion order to Scipy's scalar-last [x,y,z,w]"""
    w, x, y, z = q
    q_scipy = np.array([x, y, z, w])
    return q_scipy


def quat_argo2scipy_vectorized(q: np.ndarray) -> np.ndarray:
    """Re-order Argoverse's scalar-first [w,x,y,z] quaternion order to Scipy's scalar-last [x,y,z,w]"""
    return q[..., [1, 2, 3, 0]]


def quat_scipy2argo_vectorized(q: np.ndarray) -> np.ndarray:
    """Re-order Scipy's scalar-last [x,y,z,w] quaternion order to Argoverse's scalar-first [w,x,y,z]."""
    return q[..., [3, 0, 1, 2]]
