from typing import List


def fully_connected(n: int) -> List[List[int]]:
    """Return a fully connected adjacency matrix (no self-loops)."""
    return [[0 if i == j else 1 for j in range(n)] for i in range(n)]


def neighbors(adjacency: List[List[int]], agent_id: int) -> List[int]:
    """Return the list of agent indices that agent_id receives messages from."""
    return [j for j, v in enumerate(adjacency[agent_id]) if v == 1]
