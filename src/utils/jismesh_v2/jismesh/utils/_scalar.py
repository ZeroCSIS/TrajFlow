# -*- coding: utf-8 -*-

# Keep original imports
from __future__ import division as _division
import sys as _sys
import numpy as _np
if _sys.version_info.major < 3:
    import functools32 as _functools
else:
    import functools as _functools

# Keep original unit definitions and _dict_unit_lat_lon
# unit in degree of latitude and longitude for each mesh level.
_unit_lat_lv1 = _functools.lru_cache(1)(lambda: 2/3)
_unit_lon_lv1 = _functools.lru_cache(1)(lambda: 1)
_unit_lat_40000 = _functools.lru_cache(1)(lambda: _unit_lat_lv1()/2)
_unit_lon_40000 = _functools.lru_cache(1)(lambda: _unit_lon_lv1()/2)
_unit_lat_20000 = _functools.lru_cache(1)(lambda: _unit_lat_40000()/2)
_unit_lon_20000 = _functools.lru_cache(1)(lambda: _unit_lon_40000()/2)
_unit_lat_16000 = _functools.lru_cache(1)(lambda: _unit_lat_lv1()/5)
_unit_lon_16000 = _functools.lru_cache(1)(lambda: _unit_lon_lv1()/5)
_unit_lat_lv2 = _functools.lru_cache(1)(lambda: _unit_lat_lv1()/8)
_unit_lon_lv2 = _functools.lru_cache(1)(lambda: _unit_lon_lv1()/8)
_unit_lat_8000 = _functools.lru_cache(1)(lambda: _unit_lat_lv1()/10) # Original was lv1/10
_unit_lon_8000 = _functools.lru_cache(1)(lambda: _unit_lon_lv1()/10) # Original was lv1/10
_unit_lat_5000 = _functools.lru_cache(1)(lambda: _unit_lat_lv2()/2)
_unit_lon_5000 = _functools.lru_cache(1)(lambda: _unit_lon_lv2()/2)
_unit_lat_4000 = _functools.lru_cache(1)(lambda: _unit_lat_8000()/2) # Original used 8000
_unit_lon_4000 = _functools.lru_cache(1)(lambda: _unit_lon_8000()/2) # Original used 8000
_unit_lat_2500 = _functools.lru_cache(1)(lambda: _unit_lat_5000()/2) # Original used 5000
_unit_lon_2500 = _functools.lru_cache(1)(lambda: _unit_lon_5000()/2) # Original used 5000
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

_dict_unit_lat_lon = {
    1 : (_unit_lat_lv1, _unit_lon_lv1), 40000 : (_unit_lat_40000, _unit_lon_40000),
    20000 : (_unit_lat_20000, _unit_lon_20000), 16000 : (_unit_lat_16000, _unit_lon_16000),
    2 : (_unit_lat_lv2, _unit_lon_lv2), 8000 : (_unit_lat_8000, _unit_lon_8000),
    5000 : (_unit_lat_5000, _unit_lon_5000), 4000 : (_unit_lat_4000, _unit_lon_4000),
    2500 : (_unit_lat_2500, _unit_lon_2500), 2000 : (_unit_lat_2000, _unit_lon_2000),
    3 : (_unit_lat_lv3, _unit_lon_lv3), 4 : (_unit_lat_lv4, _unit_lon_lv4),
    5 : (_unit_lat_lv5, _unit_lon_lv5), 6 : (_unit_lat_lv6, _unit_lon_lv6)
}

def unit_lat(level):
    # Keep original
    return _dict_unit_lat_lon[level][0]()

def unit_lon(level):
    # Keep original
    return _dict_unit_lat_lon[level][1]()

# Keep original docstring for to_meshcode, but add kwargs part
def to_meshcode(lat, lon, level, astype, **kwargs): # Add **kwargs
    """緯度経度から指定次の地域メッシュコードを算出する。
       (Original docstring preserved) ...
       オプションでメッシュ内での相対的な位置（緯度・経度方向の倍率）も返す。

    Args:
        lat: 世界測地系の緯度(度単位)
        lon: 世界測地系の経度(度単位)
        level: 地域メッシュコードの次数 (see original docstring for values)
        astype: 戻り値メッシュコードの型
        **kwargs:
            return_multipliers (bool): Trueの場合、(meshcode, lat_multiplier, lon_multiplier) を返す。
                                       Falseまたは未指定の場合、meshcode のみを返す (デフォルト)。
    Return:
        指定次の地域メッシュコード (astype) または
        tuple: (指定次の地域メッシュコード, 緯度方向の倍率, 経度方向の倍率)
    """
    return_multipliers = kwargs.get('return_multipliers', False) # Get the flag

    # Keep original input validation
    if not 0 <= lat < 66.66: raise ValueError('the latitude is out of bound.')
    if not 100 <= lon < 180: raise ValueError('the longitude is out of bound.')
    if not (isinstance(level, int) and level in _dict_unit_lat_lon): # Check level validity early
        raise ValueError("the level is unsupported.")

    # Keep original remainder lambda definitions
    rem_lat_lv0 = lambda lat: lat
    rem_lon_lv0 = lambda lon: lon % 100
    rem_lat_lv1 = lambda lat: rem_lat_lv0(lat) % _unit_lat_lv1()
    rem_lon_lv1 = lambda lon: rem_lon_lv0(lon) % _unit_lon_lv1()
    rem_lat_40000 = lambda lat: rem_lat_lv1(lat) % _unit_lat_40000()
    rem_lon_40000 = lambda lon: rem_lon_lv1(lon) % _unit_lon_40000()
    rem_lat_20000 = lambda lat: rem_lat_40000(lat) % _unit_lat_20000()
    rem_lon_20000 = lambda lon: rem_lon_40000(lon) % _unit_lon_20000()
    rem_lat_16000 = lambda lat: rem_lat_lv1(lat) % _unit_lat_16000()
    rem_lon_16000 = lambda lon: rem_lon_lv1(lon) % _unit_lon_16000()
    rem_lat_lv2 = lambda lat: rem_lat_lv1(lat) % _unit_lat_lv2()
    rem_lon_lv2 = lambda lon: rem_lon_lv1(lon) % _unit_lon_lv2()
    rem_lat_8000 = lambda lat: rem_lat_lv1(lat) % _unit_lat_8000() # Uses lv1 rem
    rem_lon_8000 = lambda lon: rem_lon_lv1(lon) % _unit_lon_8000() # Uses lv1 rem
    rem_lat_5000 = lambda lat: rem_lat_lv2(lat) % _unit_lat_5000() # Uses lv2 rem
    rem_lon_5000 = lambda lon: rem_lon_lv2(lon) % _unit_lon_5000() # Uses lv2 rem
    rem_lat_4000 = lambda lat: rem_lat_8000(lat) % _unit_lat_4000() # Uses 8000 rem
    rem_lon_4000 = lambda lon: rem_lon_8000(lon) % _unit_lon_4000() # Uses 8000 rem
    rem_lat_2500 = lambda lat: rem_lat_5000(lat) % _unit_lat_2500() # Uses 5000 rem
    rem_lon_2500 = lambda lon: rem_lon_5000(lon) % _unit_lon_2500() # Uses 5000 rem
    rem_lat_2000 = lambda lat: rem_lat_lv2(lat) % _unit_lat_2000() # Uses lv2 rem
    rem_lon_2000 = lambda lon: rem_lon_lv2(lon) % _unit_lon_2000() # Uses lv2 rem
    rem_lat_lv3 = lambda lat: rem_lat_lv2(lat) % _unit_lat_lv3() # Uses lv2 rem
    rem_lon_lv3 = lambda lon: rem_lon_lv2(lon) % _unit_lon_lv3() # Uses lv2 rem
    rem_lat_lv4 = lambda lat: rem_lat_lv3(lat) % _unit_lat_lv4() # Uses lv3 rem
    rem_lon_lv4 = lambda lon: rem_lon_lv3(lon) % _unit_lon_lv4() # Uses lv3 rem
    rem_lat_lv5 = lambda lat: rem_lat_lv4(lat) % _unit_lat_lv5() # Uses lv4 rem
    rem_lon_lv5 = lambda lon: rem_lon_lv4(lon) % _unit_lon_lv5() # Uses lv4 rem
    rem_lat_lv6 = lambda lat: rem_lat_lv5(lat) % _unit_lat_lv6() # Uses lv5 rem
    rem_lon_lv6 = lambda lon: rem_lon_lv5(lon) % _unit_lon_lv6() # Uses lv5 rem

    # Map level to its specific remainder function (for multiplier calculation)
    # This maps the level value to the lambda function that calculates the remainder WITHIN that level's grid cell
    remainder_funcs = {
        1: (rem_lat_lv1, rem_lon_lv1), 40000: (rem_lat_40000, rem_lon_40000),
        20000: (rem_lat_20000, rem_lon_20000), 16000: (rem_lat_16000, rem_lon_16000),
        2: (rem_lat_lv2, rem_lon_lv2), 8000: (rem_lat_8000, rem_lon_8000), # Check original _scalar: rem_lat_8000 used lv1 base rem
        5000: (rem_lat_5000, rem_lon_5000), 4000: (rem_lat_4000, rem_lon_4000), # Check original _scalar: rem_lat_4000 used 8000 base rem
        2500: (rem_lat_2500, rem_lon_2500), # Check original _scalar: rem_lat_2500 used 5000 base rem
        2000: (rem_lat_2000, rem_lon_2000), 3: (rem_lat_lv3, rem_lon_lv3),
        4: (rem_lat_lv4, rem_lon_lv4), 5: (rem_lat_lv5, rem_lon_lv5),
        6: (rem_lat_lv6, rem_lon_lv6)
    }

    # Keep original meshcode_... function definitions
    def meshcode_lv1(lat, lon):
        ab = int(rem_lat_lv0(lat) / _unit_lat_lv1())
        cd = int(rem_lon_lv0(lon) / _unit_lon_lv1())
        return str(ab) + str(cd) # Returns string

    def meshcode_40000(lat, lon):
        e = int(rem_lat_lv1(lat) / _unit_lat_40000())*2 + int(rem_lon_lv1(lon) / _unit_lon_40000()) + 1
        return meshcode_lv1(lat, lon) + str(e) # Returns string

    def meshcode_20000(lat, lon):
        f = int(rem_lat_40000(lat) / _unit_lat_20000())*2 + int(rem_lon_40000(lon) / _unit_lon_20000()) + 1
        g = 5
        return meshcode_40000(lat, lon) + str(f) + str(g) # Returns string

    def meshcode_16000(lat, lon):
        e = int(rem_lat_lv1(lat) / _unit_lat_16000())*2
        f = int(rem_lon_lv1(lon) / _unit_lon_16000())*2
        g = 7
        # Original _scalar seems to have error here - likely intended concatenation or calculation
        # return meshcode_lv1(lat, lon) + str(e) + str(f) + str(g) # Assuming concatenation based on length
        # Rechecking _vector.py line 180 suggests calculation: meshcode_lv1*1000 + e*100 + f*10 + g
        # Let's stick to original _scalar.py string concatenation for minimal change
        return meshcode_lv1(lat, lon) + str(e) + str(f) + str(g) # Returns string

    def meshcode_lv2(lat, lon):
        e = int(rem_lat_lv1(lat) / _unit_lat_lv2())
        f = int(rem_lon_lv1(lon) / _unit_lon_lv2())
        return meshcode_lv1(lat, lon) + str(e) + str(f) # Returns string

    def meshcode_8000(lat, lon):
        e = int(rem_lat_lv1(lat) / _unit_lat_8000())
        f = int(rem_lon_lv1(lon) / _unit_lon_8000())
        g = 6
        return meshcode_lv1(lat, lon) + str(e) + str(f) + str(g) # Returns string

    def meshcode_5000(lat, lon):
        g = int(rem_lat_lv2(lat) / _unit_lat_5000())*2 + int(rem_lon_lv2(lon) / _unit_lon_5000()) + 1
        return meshcode_lv2(lat, lon) + str(g) # Returns string

    def meshcode_4000(lat, lon):
        h = int(rem_lat_8000(lat) / _unit_lat_4000())*2 + int(rem_lon_8000(lon) / _unit_lon_4000()) + 1
        i = 7
        return meshcode_8000(lat, lon) + str(h) + str(i) # Returns string

    def meshcode_2500(lat, lon):
        h = int(rem_lat_5000(lat) / _unit_lat_2500())*2 + int(rem_lon_5000(lon) / _unit_lon_2500()) + 1
        i = 6
        return meshcode_5000(lat, lon) + str(h) + str(i) # Returns string

    def meshcode_2000(lat, lon):
        g = int(rem_lat_lv2(lat) / _unit_lat_2000())*2
        h = int(rem_lon_lv2(lon) / _unit_lon_2000())*2
        i = 5
        return meshcode_lv2(lat, lon) + str(g) + str(h) + str(i) # Returns string

    def meshcode_lv3(lat, lon):
        g = int(rem_lat_lv2(lat) / _unit_lat_lv3())
        h = int(rem_lon_lv2(lon) / _unit_lon_lv3())
        return meshcode_lv2(lat, lon) + str(g) + str(h) # Returns string

    def meshcode_lv4(lat, lon):
        i = int(rem_lat_lv3(lat) / _unit_lat_lv4())*2 + int(rem_lon_lv3(lon) / _unit_lon_lv4()) + 1
        return meshcode_lv3(lat, lon) + str(i) # Returns string

    def meshcode_lv5(lat, lon):
        j = int(rem_lat_lv4(lat) / _unit_lat_lv5())*2 + int(rem_lon_lv4(lon) / _unit_lon_lv5()) + 1
        return meshcode_lv4(lat, lon) + str(j) # Returns string

    def meshcode_lv6(lat, lon):
        k = int(rem_lat_lv5(lat) / _unit_lat_lv6())*2 + int(rem_lon_lv5(lon) / _unit_lon_lv6()) + 1
        return meshcode_lv5(lat, lon) + str(k) # Returns string

    # --- Original final return block ---
    # Modify to calculate multipliers conditionally

    mesh_func = None
    if level == 1: mesh_func = meshcode_lv1
    elif level == 40000: mesh_func = meshcode_40000
    elif level == 20000: mesh_func = meshcode_20000
    elif level == 16000: mesh_func = meshcode_16000
    elif level == 2: mesh_func = meshcode_lv2
    elif level == 8000: mesh_func = meshcode_8000
    elif level == 5000: mesh_func = meshcode_5000
    elif level == 4000: mesh_func = meshcode_4000
    elif level == 2500: mesh_func = meshcode_2500
    elif level == 2000: mesh_func = meshcode_2000
    elif level == 3: mesh_func = meshcode_lv3
    elif level == 4: mesh_func = meshcode_lv4
    elif level == 5: mesh_func = meshcode_lv5
    elif level == 6: mesh_func = meshcode_lv6
    # else: # Already checked level validity earlier
    #     raise ValueError("the level is unsupported.")

    result_meshcode_str = mesh_func(lat, lon)
    result_meshcode = astype(result_meshcode_str) # Convert to final type

    if return_multipliers:
        rem_lat_func, rem_lon_func = remainder_funcs[level]
        rem_lat_val = rem_lat_func(lat)
        rem_lon_val = rem_lon_func(lon)
        unit_lat_val = unit_lat(level)
        unit_lon_val = unit_lon(level)

        # Avoid division by zero
        lat_multiplier = rem_lat_val / unit_lat_val if unit_lat_val != 0 else 0.0
        lon_multiplier = rem_lon_val / unit_lon_val if unit_lon_val != 0 else 0.0
        return result_meshcode, lat_multiplier, lon_multiplier
    else:
        return result_meshcode


# --- Keep original to_meshlevel, to_meshpoint, _make_envelope, to_envelope, to_intersects ---
# These functions should be unaffected as they call the public API which handles kwargs
# or they call the scalar to_meshcode without the new flag.

def to_meshlevel(meshcode):
    # Keep original implementation (lines 240-310)
    meshcode = str(meshcode)
    length = len(meshcode)
    if length == 4: return 1
    if length == 5: return 40000
    if length == 6: return 2
    if length == 7:
        g = meshcode[6:7]
        if g in ['1','2','3','4']: return 5000
        if g == '6': return 8000
        if g == '5': return 20000
        if g == '7': return 16000
    if length == 8: return 3
    if length == 9:
        i = meshcode[8:9]
        if i in ['1','2','3','4']: return 4
        if i == '5': return 2000
        if i == '6': return 2500
        if i == '7': return 4000
    if length == 10:
        j = meshcode[9:10]
        if j in ['1','2','3','4']: return 5
    if length == 11:
        k = meshcode[10:11]
        if k in ['1','2','3','4']: return 6
    return -1 # Original return for unsupported

def to_meshpoint(meshcode, lat_multiplier, lon_multiplier):
    # Keep original implementation (lines 312-812)
    # This function uses functools.partial heavily but seems self-contained
    # and calls to_meshlevel internally. It should not be affected.
    meshcode = str(meshcode)
    def mesh_cord(func_higher_cord, func_unit_cord, func_multiplier):
        return func_higher_cord() + func_unit_cord() * func_multiplier()
    lat_multiplier_lv = lambda: lat_multiplier
    lon_multiplier_lv = lambda: lon_multiplier
    # --- All original lat/lon_multiplier_... partials ---
    lat_multiplier_lv1 = _functools.partial(lambda meshcode: int(meshcode[0:2]), meshcode=meshcode)
    lon_multiplier_lv1 = _functools.partial(lambda meshcode: int(meshcode[2:4]), meshcode=meshcode)
    lat_multiplier_40000 = _functools.partial(lambda meshcode: int(bin(int(meshcode[4:5])-1)[2:].zfill(2)[0:1]), meshcode=meshcode)
    lon_multiplier_40000 = _functools.partial(lambda meshcode: int(bin(int(meshcode[4:5])-1)[2:].zfill(2)[1:2]), meshcode=meshcode)
    lat_multiplier_20000 = _functools.partial(lambda meshcode: int(bin(int(meshcode[5:6])-1)[2:].zfill(2)[0:1]), meshcode=meshcode)
    lon_multiplier_20000 = _functools.partial(lambda meshcode: int(bin(int(meshcode[5:6])-1)[2:].zfill(2)[1:2]), meshcode=meshcode)
    lat_multiplier_16000 = _functools.partial(lambda meshcode: int(meshcode[4:5])/2, meshcode=meshcode) # Original used int(m)/2
    lon_multiplier_16000 = _functools.partial(lambda meshcode: int(meshcode[5:6])/2, meshcode=meshcode) # Original used int(m)/2
    lat_multiplier_lv2 = _functools.partial(lambda meshcode: int(meshcode[4:5]), meshcode=meshcode)
    lon_multiplier_lv2 = _functools.partial(lambda meshcode: int(meshcode[5:6]), meshcode=meshcode)
    lat_multiplier_8000 = _functools.partial(lambda meshcode: int(meshcode[4:5]), meshcode=meshcode) # Uses L2 indices
    lon_multiplier_8000 = _functools.partial(lambda meshcode: int(meshcode[5:6]), meshcode=meshcode) # Uses L2 indices
    lat_multiplier_5000 = _functools.partial(lambda meshcode: int(bin(int(meshcode[6:7])-1)[2:].zfill(2)[0:1]), meshcode=meshcode)
    lon_multiplier_5000 = _functools.partial(lambda meshcode: int(bin(int(meshcode[6:7])-1)[2:].zfill(2)[1:2]), meshcode=meshcode)
    lat_multiplier_4000 = _functools.partial(lambda meshcode: int(bin(int(meshcode[7:8])-1)[2:].zfill(2)[0:1]), meshcode=meshcode) # Uses L3 index? No, L8000 sub index
    lon_multiplier_4000 = _functools.partial(lambda meshcode: int(bin(int(meshcode[7:8])-1)[2:].zfill(2)[1:2]), meshcode=meshcode) # Uses L8000 sub index
    lat_multiplier_2500 = _functools.partial(lambda meshcode: int(bin(int(meshcode[7:8])-1)[2:].zfill(2)[0:1]), meshcode=meshcode) # Uses L5000 sub index
    lon_multiplier_2500 = _functools.partial(lambda meshcode: int(bin(int(meshcode[7:8])-1)[2:].zfill(2)[1:2]), meshcode=meshcode) # Uses L5000 sub index
    lat_multiplier_2000 = _functools.partial(lambda meshcode: int(meshcode[6:7])/2, meshcode=meshcode) # Uses L2 sub index
    lon_multiplier_2000 = _functools.partial(lambda meshcode: int(meshcode[7:8])/2, meshcode=meshcode) # Uses L2 sub index
    lat_multiplier_lv3 = _functools.partial(lambda meshcode: int(meshcode[6:7]), meshcode=meshcode)
    lon_multiplier_lv3 = _functools.partial(lambda meshcode: int(meshcode[7:8]), meshcode=meshcode)
    lat_multiplier_lv4 = _functools.partial(lambda meshcode: int(bin(int(meshcode[8:9])-1)[2:].zfill(2)[0:1]), meshcode=meshcode)
    lon_multiplier_lv4 = _functools.partial(lambda meshcode: int(bin(int(meshcode[8:9])-1)[2:].zfill(2)[1:2]), meshcode=meshcode)
    lat_multiplier_lv5 = _functools.partial(lambda meshcode: int(bin(int(meshcode[9:10])-1)[2:].zfill(2)[0:1]), meshcode=meshcode)
    lon_multiplier_lv5 = _functools.partial(lambda meshcode: int(bin(int(meshcode[9:10])-1)[2:].zfill(2)[1:2]), meshcode=meshcode)
    lat_multiplier_lv6 = _functools.partial(lambda meshcode: int(bin(int(meshcode[10:11])-1)[2:].zfill(2)[0:1]), meshcode=meshcode)
    lon_multiplier_lv6 = _functools.partial(lambda meshcode: int(bin(int(meshcode[10:11])-1)[2:].zfill(2)[1:2]), meshcode=meshcode)
    # --- All original mesh_..._default_... partials ---
    mesh_lv1_default_lat = _functools.partial(mesh_cord, func_higher_cord=lambda: 0, func_unit_cord=_unit_lat_lv1, func_multiplier=lat_multiplier_lv1)
    mesh_lv1_default_lon = _functools.partial(mesh_cord, func_higher_cord=lambda: 100, func_unit_cord=_unit_lon_lv1, func_multiplier=lon_multiplier_lv1)
    mesh_40000_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv1_default_lat, func_unit_cord=_unit_lat_40000, func_multiplier=lat_multiplier_40000)
    mesh_40000_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv1_default_lon, func_unit_cord=_unit_lon_40000, func_multiplier=lon_multiplier_40000)
    mesh_20000_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_40000_default_lat, func_unit_cord=_unit_lat_20000, func_multiplier=lat_multiplier_20000)
    mesh_20000_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_40000_default_lon, func_unit_cord=_unit_lon_20000, func_multiplier=lon_multiplier_20000)
    mesh_16000_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv1_default_lat, func_unit_cord=_unit_lat_16000, func_multiplier=lat_multiplier_16000)
    mesh_16000_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv1_default_lon, func_unit_cord=_unit_lon_16000, func_multiplier=lon_multiplier_16000)
    mesh_lv2_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv1_default_lat, func_unit_cord=_unit_lat_lv2, func_multiplier=lat_multiplier_lv2)
    mesh_lv2_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv1_default_lon, func_unit_cord=_unit_lon_lv2, func_multiplier=lon_multiplier_lv2)
    mesh_8000_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv1_default_lat, func_unit_cord=_unit_lat_8000, func_multiplier=lat_multiplier_8000) # Base L1
    mesh_8000_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv1_default_lon, func_unit_cord=_unit_lon_8000, func_multiplier=lon_multiplier_8000) # Base L1
    mesh_5000_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv2_default_lat, func_unit_cord=_unit_lat_5000, func_multiplier=lat_multiplier_5000) # Base L2
    mesh_5000_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv2_default_lon, func_unit_cord=_unit_lon_5000, func_multiplier=lon_multiplier_5000) # Base L2
    mesh_4000_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_8000_default_lat, func_unit_cord=_unit_lat_4000, func_multiplier=lat_multiplier_4000) # Base 8000
    mesh_4000_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_8000_default_lon, func_unit_cord=_unit_lon_4000, func_multiplier=lon_multiplier_4000) # Base 8000
    mesh_2500_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_5000_default_lat, func_unit_cord=_unit_lat_2500, func_multiplier=lat_multiplier_2500) # Base 5000
    mesh_2500_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_5000_default_lon, func_unit_cord=_unit_lon_2500, func_multiplier=lon_multiplier_2500) # Base 5000
    mesh_2000_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv2_default_lat, func_unit_cord=_unit_lat_2000, func_multiplier=lat_multiplier_2000) # Base L2
    mesh_2000_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv2_default_lon, func_unit_cord=_unit_lon_2000, func_multiplier=lon_multiplier_2000) # Base L2
    mesh_lv3_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv2_default_lat, func_unit_cord=_unit_lat_lv3, func_multiplier=lat_multiplier_lv3)
    mesh_lv3_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv2_default_lon, func_unit_cord=_unit_lon_lv3, func_multiplier=lon_multiplier_lv3)
    mesh_lv4_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv3_default_lat, func_unit_cord=_unit_lat_lv4, func_multiplier=lat_multiplier_lv4)
    mesh_lv4_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv3_default_lon, func_unit_cord=_unit_lon_lv4, func_multiplier=lon_multiplier_lv4)
    mesh_lv5_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv4_default_lat, func_unit_cord=_unit_lat_lv5, func_multiplier=lat_multiplier_lv5)
    mesh_lv5_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv4_default_lon, func_unit_cord=_unit_lon_lv5, func_multiplier=lon_multiplier_lv5)
    mesh_lv6_default_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv5_default_lat, func_unit_cord=_unit_lat_lv6, func_multiplier=lat_multiplier_lv6)
    mesh_lv6_default_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv5_default_lon, func_unit_cord=_unit_lon_lv6, func_multiplier=lon_multiplier_lv6)
    # --- All original mesh_..._lat/lon partials ---
    mesh_lv1_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv1_default_lat, func_unit_cord=_unit_lat_lv1, func_multiplier=lat_multiplier_lv)
    mesh_lv1_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv1_default_lon, func_unit_cord=_unit_lon_lv1, func_multiplier=lon_multiplier_lv)
    mesh_40000_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_40000_default_lat, func_unit_cord=_unit_lat_40000, func_multiplier=lat_multiplier_lv)
    mesh_40000_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_40000_default_lon, func_unit_cord=_unit_lon_40000, func_multiplier=lon_multiplier_lv)
    mesh_20000_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_20000_default_lat, func_unit_cord=_unit_lat_20000, func_multiplier=lat_multiplier_lv)
    mesh_20000_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_20000_default_lon, func_unit_cord=_unit_lon_20000, func_multiplier=lon_multiplier_lv)
    mesh_16000_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_16000_default_lat, func_unit_cord=_unit_lat_16000, func_multiplier=lat_multiplier_lv)
    mesh_16000_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_16000_default_lon, func_unit_cord=_unit_lon_16000, func_multiplier=lon_multiplier_lv)
    mesh_lv2_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv2_default_lat, func_unit_cord=_unit_lat_lv2, func_multiplier=lat_multiplier_lv)
    mesh_lv2_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv2_default_lon, func_unit_cord=_unit_lon_lv2, func_multiplier=lon_multiplier_lv)
    mesh_8000_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_8000_default_lat, func_unit_cord=_unit_lat_8000, func_multiplier=lat_multiplier_lv)
    mesh_8000_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_8000_default_lon, func_unit_cord=_unit_lon_8000, func_multiplier=lon_multiplier_lv)
    mesh_5000_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_5000_default_lat, func_unit_cord=_unit_lat_5000, func_multiplier=lat_multiplier_lv)
    mesh_5000_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_5000_default_lon, func_unit_cord=_unit_lon_5000, func_multiplier=lon_multiplier_lv)
    mesh_4000_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_4000_default_lat, func_unit_cord=_unit_lat_4000, func_multiplier=lat_multiplier_lv)
    mesh_4000_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_4000_default_lon, func_unit_cord=_unit_lon_4000, func_multiplier=lon_multiplier_lv)
    mesh_2500_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_2500_default_lat, func_unit_cord=_unit_lat_2500, func_multiplier=lat_multiplier_lv)
    mesh_2500_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_2500_default_lon, func_unit_cord=_unit_lon_2500, func_multiplier=lon_multiplier_lv)
    mesh_2000_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_2000_default_lat, func_unit_cord=_unit_lat_2000, func_multiplier=lat_multiplier_lv)
    mesh_2000_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_2000_default_lon, func_unit_cord=_unit_lon_2000, func_multiplier=lon_multiplier_lv)
    mesh_lv3_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv3_default_lat, func_unit_cord=_unit_lat_lv3, func_multiplier=lat_multiplier_lv)
    mesh_lv3_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv3_default_lon, func_unit_cord=_unit_lon_lv3, func_multiplier=lon_multiplier_lv)
    mesh_lv4_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv4_default_lat, func_unit_cord=_unit_lat_lv4, func_multiplier=lat_multiplier_lv)
    mesh_lv4_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv4_default_lon, func_unit_cord=_unit_lon_lv4, func_multiplier=lon_multiplier_lv)
    mesh_lv5_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv5_default_lat, func_unit_cord=_unit_lat_lv5, func_multiplier=lat_multiplier_lv)
    mesh_lv5_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv5_default_lon, func_unit_cord=_unit_lon_lv5, func_multiplier=lon_multiplier_lv)
    mesh_lv6_lat = _functools.partial(mesh_cord, func_higher_cord=mesh_lv6_default_lat, func_unit_cord=_unit_lat_lv6, func_multiplier=lat_multiplier_lv)
    mesh_lv6_lon = _functools.partial(mesh_cord, func_higher_cord=mesh_lv6_default_lon, func_unit_cord=_unit_lon_lv6, func_multiplier=lon_multiplier_lv)

    level = to_meshlevel(meshcode) # Calls the scalar to_meshlevel

    # --- Original final return block ---
    if level == 1: return mesh_lv1_lat(), mesh_lv1_lon()
    if level == 40000: return mesh_40000_lat(), mesh_40000_lon()
    if level == 20000: return mesh_20000_lat(), mesh_20000_lon()
    if level == 16000: return mesh_16000_lat(), mesh_16000_lon()
    if level == 2: return mesh_lv2_lat(), mesh_lv2_lon()
    if level == 8000: return mesh_8000_lat(), mesh_8000_lon()
    if level == 5000: return mesh_5000_lat(), mesh_5000_lon()
    if level == 4000: return mesh_4000_lat(), mesh_4000_lon()
    if level == 2500: return mesh_2500_lat(), mesh_2500_lon()
    if level == 2000: return mesh_2000_lat(), mesh_2000_lon()
    if level == 3: return mesh_lv3_lat(), mesh_lv3_lon()
    if level == 4: return mesh_lv4_lat(), mesh_lv4_lon()
    if level == 5: return mesh_lv5_lat(), mesh_lv5_lon()
    if level == 6: return mesh_lv6_lat(), mesh_lv6_lon()

    raise ValueError("the level is unsupported.")


def _make_envelope(lat_s, lon_w, lat_n, lon_e, to_level, astype):
    # Keep original implementation (lines 814-822)
    # Calls public to_meshcode, will use default return value
    to_unit_lat = unit_lat(to_level)
    to_unit_lon = unit_lon(to_level)
    to_lats = _np.arange(lat_s, lat_n, to_unit_lat)
    to_lons = _np.arange(lon_w, lon_e, to_unit_lon)
    # Need to import public API to call it
    from . import to_meshcode as public_to_meshcode
    for to_lat in to_lats:
        for to_lon in to_lons:
            # Call public API - default return is meshcode only
            yield public_to_meshcode(to_lat, to_lon, to_level, astype)

def to_envelope(meshcode_sw, meshcode_ne):
    # Keep original implementation (lines 824-836)
    level_sw = to_meshlevel(meshcode_sw)
    level_ne = to_meshlevel(meshcode_ne)
    if level_sw != level_ne: raise ValueError("Levels must be the same.")
    mergin_lat = 0.5; mergin_lon = 0.5 # Original margin
    lat_s, lon_w = to_meshpoint(meshcode_sw, 0+mergin_lat, 0+mergin_lon)
    lat_n, lon_e = to_meshpoint(meshcode_ne, 1, 1) # Uses original to_meshpoint
    return _make_envelope(lat_s, lon_w, lat_n, lon_e, level_sw, type(meshcode_sw))

def to_intersects(meshcode, to_level):
    # Keep original implementation (lines 838-852)
    to_unit_lat = unit_lat(to_level); to_unit_lon = unit_lon(to_level)
    from_level = to_meshlevel(meshcode)
    from_unit_lat = unit_lat(from_level); from_unit_lon = unit_lon(from_level)
    mergin_lat = (to_unit_lat/from_unit_lat)/2 if to_unit_lat <= from_unit_lat else 0.5
    mergin_lon = (to_unit_lon/from_unit_lon)/2 if to_unit_lon <= from_unit_lon else 0.5
    from_lat_s, from_lon_w = to_meshpoint(meshcode, 0+mergin_lat, 0+mergin_lon)
    from_lat_n, from_lon_e = to_meshpoint(meshcode, 1, 1)
    return _make_envelope(from_lat_s, from_lon_w, from_lat_n, from_lon_e, to_level, type(meshcode))
