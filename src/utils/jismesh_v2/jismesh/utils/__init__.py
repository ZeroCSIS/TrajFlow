# Keep this file exactly as it was in the previous correct **kwargs** version.
# It correctly handles kwargs dispatch and uses function-local imports.

from __future__ import absolute_import
import numpy as _np

# Define public functions first, using function-local imports

def unit_lat(level):
    """(Docstring unchanged)"""
    from . import _scalar, _vector
    if _np.isscalar(level): unit_lat_func = _scalar.unit_lat
    else: unit_lat_func = _vector.unit_lat
    return unit_lat_func(level)

def unit_lon(level):
    """(Docstring unchanged)"""
    from . import _scalar, _vector
    if _np.isscalar(level): unit_lon_func = _scalar.unit_lon
    else: unit_lon_func = _vector.unit_lon
    return unit_lon_func(level)

def to_meshcode(lat, lon, level, astype=_np.int64, **kwargs):
    """
    緯度経度から指定次の地域メッシュコードを算出する。
    オプションでメッシュ内での相対的な位置（緯度・経度方向の倍率）も返すことができる。

    Args:
        lat: 世界測地系の緯度(度単位) (scalar or numpy array)
        lon: 世界測地系の経度(度単位) (scalar or numpy array)
        level: 地域メッシュコードの次数 (scalar or numpy array)
        astype: 戻り値メッシュコードの型 (default: numpy.int64)
        **kwargs:
            return_multipliers (bool): Trueの場合、(meshcode, lat_multiplier, lon_multiplier) のタプルを返す。
                                       Falseまたは未指定の場合、meshcode のみを返す (デフォルト、後方互換性のため)。

    Returns:
        meshcode (scalar or numpy array) OR
        tuple (meshcode, lat_multiplier, lon_multiplier):
               戻り値は `return_multipliers` の値に依存する。
    """
    from . import _scalar, _vector
    # Check if all inputs relevant for dispatch are scalar
    is_scalar_input = _np.isscalar(lat) and _np.isscalar(lon) and _np.isscalar(level)

    if is_scalar_input:
        to_meshcode_func = _scalar.to_meshcode
    else:
        # Use vector implementation if any input is array-like
        to_meshcode_func = _vector.to_meshcode

    # Pass kwargs down to the chosen implementation
    return to_meshcode_func(lat, lon, level, astype, **kwargs)

def to_meshlevel(meshcode):
    """(Docstring unchanged)"""
    from . import _scalar, _vector
    if _np.isscalar(meshcode): to_meshlevel_func = _scalar.to_meshlevel
    else: to_meshlevel_func = _vector.to_meshlevel
    return to_meshlevel_func(meshcode)

def to_meshpoint(meshcode, lat_multiplier, lon_multiplier):
    """(Docstring unchanged)"""
    from . import _scalar, _vector
    is_scalar_input = _np.isscalar(meshcode) and _np.isscalar(lat_multiplier) and _np.isscalar(lon_multiplier)
    if is_scalar_input: to_meshpoint_func = _scalar.to_meshpoint
    else: to_meshpoint_func = _vector.to_meshpoint
    return to_meshpoint_func(meshcode, lat_multiplier, lon_multiplier)

def to_envelope(meshcode_sw, meshcode_ne):
    """(Docstring unchanged)"""
    from . import _scalar
    if not (_np.isscalar(meshcode_sw) and _np.isscalar(meshcode_ne)):
        raise TypeError("to_envelope currently only supports scalar inputs.")
    # _scalar.to_envelope calls public API via _make_envelope, should be fine
    to_envelope_func = _scalar.to_envelope
    return to_envelope_func(meshcode_sw, meshcode_ne)

def to_intersects(meshcode, to_level):
    """(Docstring unchanged)"""
    from . import _scalar
    if not (_np.isscalar(meshcode) and _np.isscalar(to_level)):
         raise TypeError("to_intersects currently only supports scalar inputs.")
    # _scalar.to_intersects calls public API via _make_envelope, should be fine
    to_intersects_func = _scalar.to_intersects
    return to_intersects_func(meshcode, to_level)
