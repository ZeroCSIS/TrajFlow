# -*- coding: utf-8 -*-

# Keep original imports
from __future__ import division as _division
import sys as _sys
import numpy as _np
if _sys.version_info.major < 3:
    import functools32 as _functools
else:
    import functools as _functools

# Keep original helper functions
def _get_num_digits(t):
    # Note: _np.log10(0) is -inf. Add small epsilon or handle 0?
    # Original seems to assume t > 0. Let's keep it.
    with _np.errstate(divide='ignore'): # Ignore log10(0) warning if t can be 0
        digits = _np.floor(_np.log10(t)+1)
    return _np.where(t > 0, digits, 1).astype(int) # Treat 0 as 1 digit? Or handle error? Assume t>0.

def _slice(t, start, stop):
    num_digits = _get_num_digits(t)
    # Ensure safety for varying digits
    power_start = _np.maximum(num_digits - start, 0)
    power_stop = _np.maximum(num_digits - stop, 0)
    return (t % 10 ** power_start) // 10 ** power_stop


# Keep original unit definitions and _supported_levels
_unit_lat_lv1 = _functools.lru_cache(1)(lambda: 2/3)
_unit_lon_lv1 = _functools.lru_cache(1)(lambda: 1)
_unit_lat_40000 = _functools.lru_cache(1)(lambda: _unit_lat_lv1()/2)
_unit_lon_40000 = _functools.lru_cache(1)(lambda: _unit_lon_lv1()/2)
# ... (all other unit definitions as in original _vector.py) ...
_unit_lat_20000 = _functools.lru_cache(1)(lambda: _unit_lat_40000()/2)
_unit_lon_20000 = _functools.lru_cache(1)(lambda: _unit_lon_40000()/2)
_unit_lat_16000 = _functools.lru_cache(1)(lambda: _unit_lat_lv1()/5)
_unit_lon_16000 = _functools.lru_cache(1)(lambda: _unit_lon_lv1()/5)
_unit_lat_lv2 = _functools.lru_cache(1)(lambda: _unit_lat_lv1()/8)
_unit_lon_lv2 = _functools.lru_cache(1)(lambda: _unit_lon_lv1()/8)
_unit_lat_8000 = _functools.lru_cache(1)(lambda: _unit_lat_lv1()/10)
_unit_lon_8000 = _functools.lru_cache(1)(lambda: _unit_lon_lv1()/10)
_unit_lat_5000 = _functools.lru_cache(1)(lambda: _unit_lat_lv2()/2)
_unit_lon_5000 = _functools.lru_cache(1)(lambda: _unit_lon_lv2()/2)
_unit_lat_4000 = _functools.lru_cache(1)(lambda: _unit_lat_8000()/2)
_unit_lon_4000 = _functools.lru_cache(1)(lambda: _unit_lon_8000()/2)
_unit_lat_2500 = _functools.lru_cache(1)(lambda: _unit_lat_5000()/2)
_unit_lon_2500 = _functools.lru_cache(1)(lambda: _unit_lon_5000()/2)
_unit_lat_2000 = _functools.lru_cache(1)(lambda: _unit_lat_lv2()/5)
_unit_lon_2000 = _functools.lru_cache(1)(lambda: _unit_lon_lv2()/5)
_unit_lat_lv3 = _functools.lru_cache(1)(lambda: _unit_lat_lv2()/10)
_unit_lon_lv3 = _functools.lru_cache(1)(lambda: _unit_lon_lv2()/10)
_unit_lat_lv4 = _functools.lru_cache(1)(lambda: _unit_lat_lv3()/2)
_unit_lon_lv4 = _functools.lru_cache(1)(lambda: _unit_lon_lv3()/2)
_unit_lat_lv5 = _functools.lru_cache(1)(lambda: _unit_lat_lv4()/2)
_unit_lon_lv5 = _functools.lru_cache(1)(lambda: _unit_lon_lv4()/2)
_unit_lat_lv6 = _functools.lru_cache(1)(lambda: _unit_lat_lv5()/2)
_unit_lon_lv6 = _functools.lru_cache(1)(lambda: _unit_lon_lv5()/2)

_supported_levels = [1, 40000, 20000, 16000, 2, 8000, 5000, 4000, 2500, 2000, 3, 4, 5, 6]
# Import scalar unit funcs needed for vector version
from ._scalar import unit_lat as _unit_lat_scalar, unit_lon as _unit_lon_scalar

# Keep original unit_lat and unit_lon vector functions
def unit_lat(level):
    level = _np.atleast_1d(level).astype(_np.int64)
    if not _np.all(_np.isin(level, _supported_levels)): raise ValueError('Unsupported level.')
    lat = _np.zeros(level.size, dtype=_np.float64)
    for lvl_val in _supported_levels:
        if _np.any(level == lvl_val):
             lat[level==lvl_val] = _unit_lat_scalar(lvl_val) # Use scalar lookup
    if lat.size == 1: lat = _np.asscalar(lat)
    return lat

def unit_lon(level):
    level = _np.atleast_1d(level).astype(_np.int64)
    if not _np.all(_np.isin(level, _supported_levels)): raise ValueError('Unsupported level.')
    lon = _np.zeros(level.size, dtype=_np.float64)
    for lvl_val in _supported_levels:
         if _np.any(level == lvl_val):
              lon[level==lvl_val] = _unit_lon_scalar(lvl_val) # Use scalar lookup
    if lon.size == 1: lon = _np.asscalar(lon)
    return lon


# Keep original docstring for to_meshcode, but add kwargs part
def to_meshcode(lat, lon, level, astype, **kwargs): # Add **kwargs
    """緯度経度から指定次の地域メッシュコードを算出する。
       (Original docstring preserved) ...
       オプションでメッシュ内での相対的な位置（緯度・経度方向の倍率）も返す。

    Args:
        lat: 世界測地系の緯度(度単位) (scalar or numpy array)
        lon: 世界測地系の経度(度単位) (scalar or numpy array)
        level: 地域メッシュコードの次数 (scalar or numpy array)
        astype: 戻り値メッシュコードの型
        **kwargs:
            return_multipliers (bool): Trueの場合、(meshcode, lat_multiplier, lon_multiplier) を返す。
                                       Falseまたは未指定の場合、meshcode のみを返す (デフォルト)。
    Return:
        指定次の地域メッシュコード (numpy array or scalar) または
        tuple: (meshcode, lat_multiplier, lon_multiplier) (arrays or scalars)
    """
    return_multipliers = kwargs.get('return_multipliers', False) # Get the flag

    # Keep original remainder lambda definitions (operating on arrays)
    rem_lat_lv0 = lambda lat_arr: lat_arr
    rem_lon_lv0 = lambda lon_arr: lon_arr % 100
    rem_lat_lv1 = lambda lat_arr: rem_lat_lv0(lat_arr) % _unit_lat_lv1()
    rem_lon_lv1 = lambda lon_arr: rem_lon_lv0(lon_arr) % _unit_lon_lv1()
    rem_lat_40000 = lambda lat_arr: rem_lat_lv1(lat_arr) % _unit_lat_40000()
    rem_lon_40000 = lambda lon_arr: rem_lon_lv1(lon_arr) % _unit_lon_40000()
    rem_lat_20000 = lambda lat_arr: rem_lat_40000(lat_arr) % _unit_lat_20000()
    rem_lon_20000 = lambda lon_arr: rem_lon_40000(lon_arr) % _unit_lon_20000()
    rem_lat_16000 = lambda lat_arr: rem_lat_lv1(lat_arr) % _unit_lat_16000()
    rem_lon_16000 = lambda lon_arr: rem_lon_lv1(lon_arr) % _unit_lon_16000()
    rem_lat_lv2 = lambda lat_arr: rem_lat_lv1(lat_arr) % _unit_lat_lv2()
    rem_lon_lv2 = lambda lon_arr: rem_lon_lv1(lon_arr) % _unit_lon_lv2()
    rem_lat_8000 = lambda lat_arr: rem_lat_lv1(lat_arr) % _unit_lat_8000()
    rem_lon_8000 = lambda lon_arr: rem_lon_lv1(lon_arr) % _unit_lon_8000()
    rem_lat_5000 = lambda lat_arr: rem_lat_lv2(lat_arr) % _unit_lat_5000()
    rem_lon_5000 = lambda lon_arr: rem_lon_lv2(lon_arr) % _unit_lon_5000()
    rem_lat_4000 = lambda lat_arr: rem_lat_8000(lat_arr) % _unit_lat_4000()
    rem_lon_4000 = lambda lon_arr: rem_lon_8000(lon_arr) % _unit_lon_4000()
    rem_lat_2500 = lambda lat_arr: rem_lat_5000(lat_arr) % _unit_lat_2500()
    rem_lon_2500 = lambda lon_arr: rem_lon_5000(lon_arr) % _unit_lon_2500()
    rem_lat_2000 = lambda lat_arr: rem_lat_lv2(lat_arr) % _unit_lat_2000()
    rem_lon_2000 = lambda lon_arr: rem_lon_lv2(lon_arr) % _unit_lon_2000()
    rem_lat_lv3 = lambda lat_arr: rem_lat_lv2(lat_arr) % _unit_lat_lv3()
    rem_lon_lv3 = lambda lon_arr: rem_lon_lv2(lon_arr) % _unit_lon_lv3()
    rem_lat_lv4 = lambda lat_arr: rem_lat_lv3(lat_arr) % _unit_lat_lv4()
    rem_lon_lv4 = lambda lon_arr: rem_lon_lv3(lon_arr) % _unit_lon_lv4()
    rem_lat_lv5 = lambda lat_arr: rem_lat_lv4(lat_arr) % _unit_lat_lv5()
    rem_lon_lv5 = lambda lon_arr: rem_lon_lv4(lon_arr) % _unit_lon_lv5()
    rem_lat_lv6 = lambda lat_arr: rem_lat_lv5(lat_arr) % _unit_lat_lv6()
    rem_lon_lv6 = lambda lon_arr: rem_lon_lv5(lon_arr) % _unit_lon_lv6()

    # Map level to its specific remainder function (vector version)
    remainder_funcs_vec = {
        1: (rem_lat_lv1, rem_lon_lv1), 40000: (rem_lat_40000, rem_lon_40000),
        20000: (rem_lat_20000, rem_lon_20000), 16000: (rem_lat_16000, rem_lon_16000),
        2: (rem_lat_lv2, rem_lon_lv2), 8000: (rem_lat_8000, rem_lon_8000),
        5000: (rem_lat_5000, rem_lon_5000), 4000: (rem_lat_4000, rem_lon_4000),
        2500: (rem_lat_2500, rem_lon_2500), 2000: (rem_lat_2000, rem_lon_2000),
        3: (rem_lat_lv3, rem_lon_lv3), 4: (rem_lat_lv4, rem_lon_lv4),
        5: (rem_lat_lv5, rem_lon_lv5), 6: (rem_lat_lv6, rem_lon_lv6)
    }

    # Keep original meshcode_... function definitions (operating on arrays)
    def meshcode_lv1(lat_arr, lon_arr):
        ab = (rem_lat_lv0(lat_arr) // _unit_lat_lv1())
        cd = rem_lon_lv0(lon_arr) // _unit_lon_lv1()
        return ab*100 + cd # Returns array

    def meshcode_40000(lat_arr, lon_arr):
        e = (rem_lat_lv1(lat_arr) // _unit_lat_40000())*2 + (rem_lon_lv1(lon_arr) // _unit_lon_40000()) + 1
        return meshcode_lv1(lat_arr, lon_arr)*10 + e # Returns array

    def meshcode_20000(lat_arr, lon_arr):
        f = (rem_lat_40000(lat_arr) // _unit_lat_20000())*2 + (rem_lon_40000(lon_arr) // _unit_lon_20000()) + 1
        g = 5
        return meshcode_40000(lat_arr, lon_arr)*100 + f*10 + g # Returns array

    def meshcode_16000(lat_arr, lon_arr):
        e = (rem_lat_lv1(lat_arr) // _unit_lat_16000())*2
        f = (rem_lon_lv1(lon_arr) // _unit_lon_16000())*2
        g = 7
        return meshcode_lv1(lat_arr, lon_arr)*1000 + e*100 + f*10 + g # Returns array

    def meshcode_lv2(lat_arr, lon_arr):
        e = (rem_lat_lv1(lat_arr) // _unit_lat_lv2())
        f = (rem_lon_lv1(lon_arr) // _unit_lon_lv2())
        return meshcode_lv1(lat_arr, lon_arr)*100 + e*10 + f # Returns array

    def meshcode_8000(lat_arr, lon_arr):
        e = (rem_lat_lv1(lat_arr) // _unit_lat_8000())
        f = (rem_lon_lv1(lon_arr) // _unit_lon_8000())
        g = 6
        return meshcode_lv1(lat_arr, lon_arr)*1000 + e*100 + f*10 + g # Returns array

    def meshcode_5000(lat_arr, lon_arr):
        g = (rem_lat_lv2(lat_arr) // _unit_lat_5000())*2 + (rem_lon_lv2(lon_arr) // _unit_lon_5000()) + 1
        return meshcode_lv2(lat_arr, lon_arr)*10 + g # Returns array

    def meshcode_4000(lat_arr, lon_arr):
        h = (rem_lat_8000(lat_arr) // _unit_lat_4000())*2 + (rem_lon_8000(lon_arr) // _unit_lon_4000()) + 1
        i = 7
        return meshcode_8000(lat_arr, lon_arr)*100 + h*10 + i # Returns array

    def meshcode_2500(lat_arr, lon_arr):
        h = (rem_lat_5000(lat_arr) // _unit_lat_2500())*2 + (rem_lon_5000(lon_arr) // _unit_lon_2500()) + 1
        i = 6
        return meshcode_5000(lat_arr, lon_arr)*100 + h*10 + i # Returns array

    def meshcode_2000(lat_arr, lon_arr):
        g = (rem_lat_lv2(lat_arr) // _unit_lat_2000())*2
        h = (rem_lon_lv2(lon_arr) // _unit_lon_2000())*2
        i = 5
        return meshcode_lv2(lat_arr, lon_arr)*1000 + g*100 + h*10 + i # Returns array

    def meshcode_lv3(lat_arr, lon_arr):
        g = (rem_lat_lv2(lat_arr) // _unit_lat_lv3())
        h = (rem_lon_lv2(lon_arr) // _unit_lon_lv3())
        return meshcode_lv2(lat_arr, lon_arr)*100 + g*10 + h # Returns array

    def meshcode_lv4(lat_arr, lon_arr):
        i = (rem_lat_lv3(lat_arr) // _unit_lat_lv4())*2 + (rem_lon_lv3(lon_arr) // _unit_lon_lv4()) + 1
        return meshcode_lv3(lat_arr, lon_arr)*10 + i # Returns array

    def meshcode_lv5(lat_arr, lon_arr):
        j = (rem_lat_lv4(lat_arr) // _unit_lat_lv5())*2 + (rem_lon_lv4(lon_arr) // _unit_lon_lv5()) + 1
        return meshcode_lv4(lat_arr, lon_arr)*10 + j # Returns array

    def meshcode_lv6(lat_arr, lon_arr):
        k = (rem_lat_lv5(lat_arr) // _unit_lat_lv6())*2 + (rem_lon_lv5(lon_arr) // _unit_lon_lv6()) + 1
        return meshcode_lv5(lat_arr, lon_arr)*10 + k # Returns array

    # Keep original input processing and validation
    lat_in = _np.atleast_1d(lat).astype(_np.float64) # Use _in suffix
    lon_in = _np.atleast_1d(lon).astype(_np.float64)
    level_in = _np.atleast_1d(level).astype(_np.int64)

    if not _np.all(_np.isin(level_in, _supported_levels)): raise ValueError('Unsupported level.')
    if _np.any(lat_in < 0) | _np.any(lat_in >= 66.66): raise ValueError('Latitude out of bound.')
    if _np.any(lon_in < 100) | _np.any(lon_in >= 180): raise ValueError('Longitude out of bound.')

    # Broadcast inputs
    try:
        b_lat, b_lon, b_level = _np.broadcast_arrays(lat_in, lon_in, level_in)
    except ValueError as e:
        raise ValueError(f"Input arrays could not be broadcast. Shapes: {lat_in.shape}, {lon_in.shape}, {level_in.shape}. Error: {e}")

    # Keep original meshcode calculation using masking
    meshcode = _np.zeros(b_lat.shape, dtype=_np.float64) # Start with float for calculation

    if _np.any(_np.isin(b_level, 1)): meshcode += meshcode_lv1(b_lat, b_lon) * (b_level == 1)
    if _np.any(_np.isin(b_level, 40000)): meshcode += meshcode_40000(b_lat, b_lon) * (b_level == 40000)
    if _np.any(_np.isin(b_level, 20000)): meshcode += meshcode_20000(b_lat, b_lon) * (b_level == 20000)
    if _np.any(_np.isin(b_level, 16000)): meshcode += meshcode_16000(b_lat, b_lon) * (b_level == 16000)
    if _np.any(_np.isin(b_level, 2)): meshcode += meshcode_lv2(b_lat, b_lon) * (b_level == 2)
    if _np.any(_np.isin(b_level, 8000)): meshcode += meshcode_8000(b_lat, b_lon) * (b_level == 8000)
    if _np.any(_np.isin(b_level, 5000)): meshcode += meshcode_5000(b_lat, b_lon) * (b_level == 5000)
    if _np.any(_np.isin(b_level, 4000)): meshcode += meshcode_4000(b_lat, b_lon) * (b_level == 4000)
    if _np.any(_np.isin(b_level, 2500)): meshcode += meshcode_2500(b_lat, b_lon) * (b_level == 2500)
    if _np.any(_np.isin(b_level, 2000)): meshcode += meshcode_2000(b_lat, b_lon) * (b_level == 2000)
    if _np.any(_np.isin(b_level, 3)): meshcode += meshcode_lv3(b_lat, b_lon) * (b_level == 3)
    if _np.any(_np.isin(b_level, 4)): meshcode += meshcode_lv4(b_lat, b_lon) * (b_level == 4)
    if _np.any(_np.isin(b_level, 5)): meshcode += meshcode_lv5(b_lat, b_lon) * (b_level == 5)
    if _np.any(_np.isin(b_level, 6)): meshcode += meshcode_lv6(b_lat, b_lon) * (b_level == 6)

    # Keep original type conversion
    meshcode = meshcode.astype(_np.int64)
    meshcode = meshcode.astype(astype)

    # --- Add multiplier calculation conditionally ---
    if return_multipliers:
        lat_multiplier_arr = _np.zeros(b_lat.shape, dtype=_np.float64)
        lon_multiplier_arr = _np.zeros(b_lat.shape, dtype=_np.float64)
        unique_levels_in_input = _np.unique(b_level)

        for lvl in unique_levels_in_input:
            if lvl not in _supported_levels: continue # Should have been caught earlier
            mask = (b_level == lvl)
            if _np.any(mask):
                rem_lat_func, rem_lon_func = remainder_funcs_vec[lvl]
                rem_lat_vals = rem_lat_func(b_lat[mask])
                rem_lon_vals = rem_lon_func(b_lon[mask])
                unit_lat_val = _unit_lat_scalar(lvl) # Use scalar unit value
                unit_lon_val = _unit_lon_scalar(lvl) # Use scalar unit value

                # Avoid division by zero
                lat_mult = _np.divide(rem_lat_vals, unit_lat_val, out=_np.zeros_like(rem_lat_vals), where=unit_lat_val!=0)
                lon_mult = _np.divide(rem_lon_vals, unit_lon_val, out=_np.zeros_like(rem_lon_vals), where=unit_lon_val!=0)

                lat_multiplier_arr[mask] = lat_mult
                lon_multiplier_arr[mask] = lon_mult

        # Handle scalar return if input was scalar
        if lat_in.ndim == 0 and lon_in.ndim == 0 and level_in.ndim == 0:
            return _np.asscalar(meshcode), _np.asscalar(lat_multiplier_arr), _np.asscalar(lon_multiplier_arr)
        else:
            return meshcode, lat_multiplier_arr, lon_multiplier_arr
    else:
        # Original return path
        # Handle scalar return if input was scalar
        if lat_in.ndim == 0 and lon_in.ndim == 0 and level_in.ndim == 0:
            return _np.asscalar(meshcode)
        else:
            return meshcode

# --- Keep original to_meshlevel and to_meshpoint vector functions ---
# These call the scalar versions internally or use slicing, should be unaffected

def to_meshlevel(meshcode):
    # Keep original implementation (lines 296-350)
    meshcode_arr = _np.array(meshcode).astype(_np.int64) # Use _arr suffix
    level_arr = _np.full(meshcode_arr.shape, _np.int64(-1))
    num_digits = _get_num_digits(meshcode_arr)

    # Slicing needs to handle arrays now
    g = _slice(meshcode_arr, 6, 7)
    i = _slice(meshcode_arr, 8, 9)
    j = _slice(meshcode_arr, 9, 10)
    k = _slice(meshcode_arr, 10, 11)

    level_arr[(num_digits==4)] = 1
    level_arr[(num_digits==5)] = 40000
    level_arr[(num_digits==6)] = 2
    level_arr[(num_digits==7) & (_np.isin(g, [1,2,3,4]))] = 5000
    level_arr[(num_digits==7) & (g == 6)] = 8000
    level_arr[(num_digits==7) & (g == 5)] = 20000
    level_arr[(num_digits==7) & (g == 7)] = 16000
    level_arr[(num_digits==8)] = 3
    level_arr[(num_digits==9) & (_np.isin(i, [1,2,3,4]))] = 4
    level_arr[(num_digits==9) & (i == 5)] = 2000
    level_arr[(num_digits==9) & (i == 6)] = 2500
    level_arr[(num_digits==9) & (i == 7)] = 4000
    level_arr[(num_digits==10) & (_np.isin(j, [1,2,3,4]))] = 5
    level_arr[(num_digits==11) & (_np.isin(k, [1,2,3,4]))] = 6

    if level_arr.size == 1: level_arr = _np.asscalar(level_arr)
    return level_arr


def to_meshpoint(meshcode, lat_multiplier, lon_multiplier):
    # Keep original implementation (lines 352-496)
    # This function is complex and uses array broadcasting/masking heavily.
    # It calls the vector to_meshlevel. It should remain correct.
    meshcode_arr = _np.array(meshcode).astype(_np.int64) # Use _arr suffix
    lat_multiplier_arr = _np.atleast_1d(lat_multiplier) # Ensure array
    lon_multiplier_arr = _np.atleast_1d(lon_multiplier) # Ensure array

    # Original slicing
    ab = _slice(meshcode_arr, 0, 2); cd = _slice(meshcode_arr, 2, 4)
    e = _slice(meshcode_arr, 4, 5); f = _slice(meshcode_arr, 5, 6)
    g = _slice(meshcode_arr, 6, 7); h = _slice(meshcode_arr, 7, 8)
    i = _slice(meshcode_arr, 8, 9); j = _slice(meshcode_arr, 9, 10)
    k = _slice(meshcode_arr, 10, 11)

    level_arr = to_meshlevel(meshcode_arr) # Calls vector version

    # Original target masks
    target_lv1 = level_arr == 1; target_40000 = level_arr == 40000; target_20000 = level_arr == 20000
    target_16000 = level_arr == 16000; target_8000 = level_arr == 8000; target_4000 = level_arr == 4000
    target_lv2 = level_arr == 2; target_5000 = level_arr == 5000; target_2500 = level_arr == 2500
    target_2000 = level_arr == 2000; target_lv3 = level_arr == 3; target_lv4 = level_arr == 4
    target_lv5 = level_arr == 5; target_lv6 = level_arr == 6

    # Original lat/lon calculation using broadcasting and masking
    lat = _np.zeros(meshcode_arr.shape, dtype=float)
    lon = _np.zeros(meshcode_arr.shape, dtype=float)

    targets = target_lv1 | target_40000 | target_20000 | target_16000 | target_8000 | target_4000 | target_lv2 | target_5000 | target_2500 | target_2000 | target_lv3 | target_lv4 | target_lv5 | target_lv6
    lat += (ab * _unit_lat_lv1()) * targets
    lon += (cd * _unit_lon_lv1() + 100) * targets

    targets = target_40000 | target_20000
    lat += ((e//3 == 1) * _unit_lat_40000()) * targets
    lon += ((e%2 == 0) * _unit_lon_40000()) * targets

    targets = target_20000
    lat += ((f//3 == 1) * _unit_lat_20000()) * targets
    lon += ((f%2 == 0) * _unit_lon_20000()) * targets

    targets = target_16000
    lat += (e//2 * _unit_lat_16000()) * targets # Original used //
    lon += (f//2 * _unit_lon_16000()) * targets # Original used //

    targets = target_8000 | target_4000
    lat += (e * _unit_lat_8000()) * targets
    lon += (f * _unit_lon_8000()) * targets

    targets = target_4000
    lat += ((h//3 == 1) * _unit_lat_4000()) * targets
    lon += ((h%2 == 0) * _unit_lon_4000()) * targets

    targets = target_lv2 | target_5000 | target_2500 | target_2000 | target_lv3 | target_lv4 | target_lv5 | target_lv6
    lat += (e * _unit_lat_lv2()) * targets
    lon += (f * _unit_lon_lv2()) * targets

    targets = target_5000 | target_2500
    lat += ((g//3 == 1) * _unit_lat_5000()) * targets
    lon += ((g%2 == 0) * _unit_lon_5000()) * targets

    targets = target_2500
    lat += ((h//3 == 1) * _unit_lat_2500()) * targets
    lon += ((h%2 == 0) * _unit_lon_2500()) * targets

    targets = target_2000
    lat += (g//2 * _unit_lat_2000()) * targets # Original used //
    lon += (h//2 * _unit_lon_2000()) * targets # Original used //

    targets = target_lv3 | target_lv4 | target_lv5 | target_lv6
    lat += (g * _unit_lat_lv3()) * targets
    lon += (h * _unit_lon_lv3()) * targets

    targets = target_lv4 | target_lv5 | target_lv6
    lat += ((i//3 == 1) * _unit_lat_lv4()) * targets
    lon += ((i%2 == 0) * _unit_lon_lv4()) * targets

    targets = target_lv5 | target_lv6
    lat += ((j//3 == 1) * _unit_lat_lv5()) * targets
    lon += ((j%2 == 0) * _unit_lon_lv5()) * targets

    targets = target_lv6
    lat += ((k//3 == 1) * _unit_lat_lv6()) * targets
    lon += ((k%2 == 0) * _unit_lon_lv6()) * targets

    # Original final addition of multipliers
    # Need to use the vector unit_lat/unit_lon functions here
    lat += unit_lat(level_arr) * lat_multiplier_arr
    lon += unit_lon(level_arr) * lon_multiplier_arr

    # Handle scalar return if input was scalar
    if meshcode_arr.ndim == 0 and lat_multiplier_arr.ndim <=1 and lon_multiplier_arr.ndim <=1: # Check original input shapes? Safer to check output shape.
         if lat.size == 1 and lon.size == 1:
              return _np.asscalar(lat), _np.asscalar(lon)
    return lat, lon
