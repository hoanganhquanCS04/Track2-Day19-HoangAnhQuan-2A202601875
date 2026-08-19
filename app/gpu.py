"""Make the CUDA execution provider actually loadable on Windows.

Why this exists: `pip install onnxruntime-gpu` ships
`onnxruntime_providers_cuda.dll`, but that DLL links against the CUDA runtime
(`cublasLt64_13.dll`, cuDNN, …) which pip installs under
`site-packages/nvidia/**/bin` — a directory Windows does not search. The load
fails with `Error 126: The specified module could not be found`, onnxruntime
prints a warning, and then **silently falls back to CPU**. Nothing crashes; you
just quietly pay 50 ms per query instead of 7 ms.

That is the exact "silent regression" failure mode VIBE-CODING.md warns about,
so this module makes the state observable instead of guessable:
`setup_cuda_dlls()` reports which directories it registered, and
`cuda_available()` answers whether the provider can really be created rather
than whether it merely appears in `get_available_providers()`.

Must run BEFORE onnxruntime is first imported — importing `app.embeddings`
triggers it, and `notebooks/_setup.py` calls it for the notebooks.

No-op on non-Windows and on machines with no GPU packages installed, so the
lite CPU path behaves exactly as before.
"""
from __future__ import annotations

import os
import site
import sys
from functools import lru_cache
from pathlib import Path

# Relative to site-packages. cu13 holds cuBLAS/cuFFT/cuRAND/nvrtc; cudnn is
# packaged separately by NVIDIA and therefore lands in its own tree.
_DLL_SUBDIRS = (
    "nvidia/cu13/bin/x86_64",
    "nvidia/cudnn/bin",
    "nvidia/cu12/bin",          # CUDA 12 builds of onnxruntime-gpu
    "nvidia/cublas/bin",
    "nvidia/cuda_runtime/bin",
)


def _site_dirs() -> list[Path]:
    dirs = [Path(p) for p in site.getsitepackages()]
    user = site.getusersitepackages()
    if isinstance(user, str):
        dirs.append(Path(user))
    return dirs


@lru_cache(maxsize=1)
def setup_cuda_dlls() -> list[str]:
    """Register NVIDIA DLL directories with the loader. Returns what it found.

    Cached: `os.add_dll_directory` leaks a handle per call, and the second call
    would add nothing anyway.
    """
    if sys.platform != "win32":
        return []

    registered: list[str] = []
    for base in _site_dirs():
        for sub in _DLL_SUBDIRS:
            d = base / sub
            if not d.is_dir():
                continue
            os.add_dll_directory(str(d))
            # PATH too: child processes (uvicorn under NB3) inherit the env,
            # but they do NOT inherit add_dll_directory registrations.
            if str(d) not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"
            registered.append(str(d))
    return registered


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """True only if a CUDA session can really be created.

    `'CUDAExecutionProvider' in ort.get_available_providers()` is NOT enough —
    it is true whenever onnxruntime-gpu is installed, including when every
    dependent DLL is missing. Building a one-node session is the honest test.
    """
    setup_cuda_dlls()
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return False

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False

    try:
        from onnx import TensorProto, helper
    except ImportError:
        # onnx isn't a lab dependency; fall back to trusting the provider list
        # plus the presence of the runtime DLLs we just registered.
        return any("cu1" in d for d in setup_cuda_dlls())

    try:
        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph(
            [node], "probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph)
        opts = ort.SessionOptions()
        opts.log_severity_level = 4          # keep the probe quiet
        sess = ort.InferenceSession(
            model.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        ok = "CUDAExecutionProvider" in sess.get_providers()
        del sess
        _ = np
        return ok
    except Exception:
        return False


def describe() -> str:
    dirs = setup_cuda_dlls()
    if not dirs:
        return "cuda: no NVIDIA DLL directories found (CPU path)"
    return f"cuda: registered {len(dirs)} DLL dir(s); usable={cuda_available()}"
