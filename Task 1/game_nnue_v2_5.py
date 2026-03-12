"""
game_nnue.py
============
Python bridge for engine_nnue.dll – the NNUE-powered RoboGambit engine.
Drop-in replacement for game.py; same external API (get_best_move).

On Windows + MinGW, the gcc runtime DLLs live in the MinGW bin directory.
Python 3.8+ requires os.add_dll_directory() before ctypes.CDLL() to find them.
"""

import numpy as np
import ctypes
import os
import platform
from typing import Optional

# ── 0. Add MinGW runtime DLL search path (Windows only) ──────────────────────
if platform.system() == "Windows":
    _mingw_candidates = [
        # Common MinGW install paths – adjust if your install is elsewhere
        r"C:\Users\Hp\OneDrive - IIT Delhi\Desktop\C++ Compilers\mingw64\bin",
        r"C:\mingw64\bin",
        r"C:\msys64\mingw64\bin",
    ]
    for _p in _mingw_candidates:
        if os.path.isdir(_p):
            os.add_dll_directory(_p)
            break

# ── 1. Auto-Compile & Load the NNUE C++ Engine ───────────────────────────────
# ── 1. Auto-Compile & Load the NNUE C++ Engine ───────────────────────────────
system  = platform.system()
lib_ext = ".dll" if system == "Windows" else ".so"

# Force the script to look in its OWN directory
base_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(base_dir, "engine_nnue_v2_5" + lib_ext)
cpp_path = os.path.join(base_dir, "engine_nnue_v2_5.cpp")

# Pre-define to prevent NameError if compilation fails
engine_lib = None 

if not os.path.exists(lib_path):
    print(f"[game_nnue.py] Compiling NNUE engine for {system}...")
    if system == "Darwin":
        cmd = f'g++ -std=c++17 -O3 -shared -fPIC "{cpp_path}" -o "{lib_path}"'
    elif system == "Windows":
        cmd = (f'g++ -std=c++17 -O3 -shared -fPIC '
               f'-static-libgcc -static-libstdc++ "{cpp_path}" -o "{lib_path}"')
    else:   
        cmd = (f'g++ -std=c++17 -O3 -march=native -ffast-math '
               f'-shared -fPIC "{cpp_path}" -o "{lib_path}"')
               
    ret = os.system(cmd)
    if ret != 0:
        print(f"WARNING: [game_nnue.py] Compilation FAILED (exit code {ret}).")

if os.path.exists(lib_path):
    try:
        engine_lib = ctypes.CDLL(lib_path)
        engine_lib.search_best_move.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_bool, ctypes.c_int,
        ctypes.c_double, ctypes.c_bool]
        engine_lib.search_best_move.restype = ctypes.c_char_p
        
        engine_lib.reset_game_history.argtypes = []
        engine_lib.reset_game_history.restype = None
        print(f"[game_nnue.py] NNUE engine loaded from {lib_path}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load NNUE engine: {e}")
        engine_lib = None
# ── 2. Python State Tracking ──────────────────────────────────────────────────
GAME_HISTORY: dict = {}
_move_number: int = 0   # simple counter; not inflated by repeated positions

def get_board_hash_simple(board: np.ndarray) -> int:
    return hash(board.tobytes())

# ── 3. Public API ─────────────────────────────────────────────────────────────
def get_best_move(board: np.ndarray, playing_white: bool = True
                  ) -> Optional[str]:
    """
    RoboGambit Task 1 Entry  (NNUE edition).
    Same signature as game.py so it can be swapped in without changes.

    Parameters
    ----------
    board        : 6x6 numpy int32 board array
    playing_white: True if this side to move is White
    is_new_game  : Set True at the first call of each new game to flush the
                   C++ GAME_HISTORY and prevent cross-game repetition phantoms
    time_limit   : Maximum time (in seconds) the C++ engine is allowed to think
    use_noise    : If True, play a random legal move during the opening (ply ≤ 3)
                   to introduce opening diversity (default False)
    """
    global GAME_HISTORY, _move_number
    time_limit = 3.0
    is_new_game = False
    use_noise = False

    if engine_lib is None:
        print("Error: NNUE engine not loaded.")
        return None

    # Flush both history tables at the start of a new game
    if is_new_game:
        engine_lib.reset_game_history()
        GAME_HISTORY.clear()
        _move_number = 0

    h = get_board_hash_simple(board)
    GAME_HISTORY[h] = GAME_HISTORY.get(h, 0) + 1
    
    _move_number += 1          # real move counter, unaffected by repeated positions
    current_ply = _move_number

    flat_board = board.flatten().astype(np.int32)
    c_board = flat_board.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Pass the time_limit and use_noise dynamically to the C++ engine
    move_bytes = engine_lib.search_best_move(
        c_board, playing_white, current_ply,
        ctypes.c_double(time_limit), ctypes.c_bool(use_noise))

    if move_bytes:
        move_str = move_bytes.decode("utf-8")
        return None if move_str == "NONE" else move_str
    return None
# ── 4. Sandbox Tester ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing RoboGambit NNUE Python -> C++ Bridge...")
    initial_board = np.array([
        [ 2,  3,  4,  5,  3,  2],
        [ 1,  1,  1,  1,  1,  1],
        [ 0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0],
        [ 6,  6,  6,  6,  6,  6],
        [ 7,  8,  9, 10,  8,  7],
    ], dtype=np.int32)

    best_move = get_best_move(initial_board, playing_white=True)
