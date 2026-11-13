"""Rook contiguity geometry tests."""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components

from bstr.grid import adjacency_edge_list, build_grid, rook_adjacency


def test_rook_symmetric_and_connected(tiny_env):
    grid = build_grid(tiny_env.cfg)
    W = rook_adjacency(grid)
    assert (W != W.T).nnz == 0
    n_components, _ = connected_components(W, directed=False)
    assert n_components == 1
    assert W.diagonal().sum() == 0


def test_rook_neighbour_counts(tiny_env):
    W = rook_adjacency(build_grid(tiny_env.cfg))
    degree = np.asarray(W.sum(axis=1)).ravel()
    n_cols = tiny_env.cfg.N_COLS
    assert degree[0] == 2  # corner
    assert degree[1] == 3  # edge
    assert degree[n_cols + 1] == 4  # interior
    assert degree[-1] == 2  # opposite corner


def test_rook_edge_list_is_upper_triangle(tiny_env):
    W = rook_adjacency(build_grid(tiny_env.cfg))
    e1, e2 = adjacency_edge_list(W)
    assert len(e1) == len(e2) == 2 * tiny_env.cfg.N_ROWS * (tiny_env.cfg.N_COLS - 1)
    assert np.all(e1 < e2)
    assert np.all(W[e1, e2] == 1)
