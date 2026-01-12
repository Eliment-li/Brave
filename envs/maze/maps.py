RESET = R = 'r'
GOAL = G = 'g'


U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, G, G, 1],
          [1, 1, 1, G, 1],
          [1, G, G, G, 1],
          [1, 1, 1, 1, 1]]

U_MAZE_EVAL = [[1, 1, 1, 1, 1],
               [1, R, 0, 0, 1],
               [1, 1, 1, 0, 1],
               [1, G, G, G, 1],
               [1, 1, 1, 1, 1]]

U_MAZE_SINGLE_EVAL = [[1, 1, 1, 1, 1],
               [1, R, 0, 0, 1],
               [1, 1, 1, 0, 1],
               [1, G, 0, 0, 1],
               [1, 1, 1, 1, 1]]

# --- map registry (name -> map) ---
MAPS = {
    "U_MAZE": U_MAZE,
    "U_MAZE_EVAL": U_MAZE_EVAL,
    "U_MAZE_SINGLE_EVAL": U_MAZE_SINGLE_EVAL,
}

def get_map(name: str):
    if not name:
        raise ValueError("map name is empty")
    try:
        return MAPS[name]
    except KeyError as e:
        raise KeyError(f"Unknown map name: {name}. Available: {sorted(MAPS.keys())}") from e
