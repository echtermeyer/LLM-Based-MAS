from typing import List

TOPOLOGY_NAMES = ["fc", "ring", "chain", "star"]


def fully_connected(n: int) -> List[List[int]]:
    return [[0 if i == j else 1 for j in range(n)] for i in range(n)]


def ring(n: int) -> List[List[int]]:
    # Directed: agent i receives only from (i-1) % n.
    # Information flows i-1 → i → i+1 → … (limit cycle).
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        mat[i][(i - 1) % n] = 1
    return mat


def chain(n: int) -> List[List[int]]:
    # Undirected line: agent i receives from i-1 and i+1 where they exist.
    # Endpoints have exactly one neighbor.
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        if i > 0:
            mat[i][i - 1] = 1
        if i < n - 1:
            mat[i][i + 1] = 1
    return mat


def star(n: int, hub: int = 0) -> List[List[int]]:
    # Hub receives from all leaves and is received by all leaves.
    # Leaves receive only from the hub.
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        if i != hub:
            mat[i][hub] = 1
            mat[hub][i] = 1
    return mat


def neighbors(adjacency: List[List[int]], agent_id: int) -> List[int]:
    return [j for j, v in enumerate(adjacency[agent_id]) if v == 1]
