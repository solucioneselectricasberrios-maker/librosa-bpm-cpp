"""
export_golden_bin.py — Vuelca golden.npz a .bin crudos + manifiesto .json.

Formato .bin: datos little-endian, C-contiguous (row-major), dtype float64 o int64.
El manifiesto golden_manifest.json registra shape y dtype de cada array.

Esto permite a los tests C++ comparar contra el golden SIN depender de cnpy/zlib
(npz esta comprimido). Solo fopen + fread.
"""
import json
import os
import numpy as np

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_NPZ = os.path.join(ROOT, "tests", "golden", "golden.npz")
OUT_DIR = os.path.join(ROOT, "tests", "golden", "bin")
os.makedirs(OUT_DIR, exist_ok=True)

g = np.load(GOLDEN_NPZ)
manifest = {"_comment": "Arrays volcados como little-endian C-contiguous. dtypes: <f8 / <i8 / '?' (bool)."}

for key in g.files:
    a = np.ascontiguousarray(g[key])
    fname = f"{key}.bin"
    # Mapear dtype numpy -> etiqueta
    if a.dtype == np.float64:
        dtype = "<f8"
    elif a.dtype == np.int64:
        dtype = "<i8"
    elif a.dtype == np.bool_:
        dtype = "?"
        a = a.astype("<u1")  # bool como uint8 (0/1)
    else:
        dtype = str(a.dtype)
    a.tofile(os.path.join(OUT_DIR, fname))
    manifest[key] = {
        "file": fname,
        "shape": list(a.shape),
        "dtype": dtype,
        "nbytes": int(a.nbytes),
    }
    print(f"  {key:18} -> {fname:24} {str(a.shape):16} {dtype}")

with open(os.path.join(OUT_DIR, "golden_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest: {os.path.join(OUT_DIR, 'golden_manifest.json')}")
