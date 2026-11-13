"""Square-cell geometry and rook contiguity for the operating design domain (ODD)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from shapely import STRtree
from shapely.geometry import box

from .config import SimConfig


def build_grid(cfg: SimConfig) -> gpd.GeoDataFrame:
    """Build the N_ROWS x N_COLS unit-cell grid.

    cell_id is row-major: cell_id = row * N_COLS + col. Centroids and polygons are
    in a local planar kilometre coordinate system anchored at (0, 0).
    """
    cell_ids = np.arange(cfg.N_CELLS, dtype=int)
    rows = cell_ids // cfg.N_COLS
    cols = cell_ids % cfg.N_COLS
    x0 = cols * cfg.CELL_KM
    y0 = rows * cfg.CELL_KM

    geoms = [box(x, y, x + cfg.CELL_KM, y + cfg.CELL_KM) for x, y in zip(x0, y0, strict=True)]
    return gpd.GeoDataFrame(
        {
            "cell_id": cell_ids,
            "row": rows,
            "col": cols,
            "centroid_x": x0 + cfg.CELL_KM / 2.0,
            "centroid_y": y0 + cfg.CELL_KM / 2.0,
            "geometry": geoms,
        },
        geometry="geometry",
        crs="EPSG:32633",  # local UTM-like planar CRS; distances are kilometres
    )


def rook_adjacency(gdf: gpd.GeoDataFrame) -> sparse.csr_matrix:
    """Rook contiguity: cells are neighbours iff they share a boundary segment.

    Corner-only (queen) contacts have intersection length zero and are excluded.
    The returned matrix is symmetric with a zero diagonal; the graph is asserted
    connected because the ICAR prior is improper on disconnected components.
    """
    n = len(gdf)
    geoms = gdf.geometry.values
    strtree = STRtree(geoms)
    left, right = strtree.query(geoms, predicate="intersects")
    pairs = [(int(i), int(j)) for i, j in zip(left, right, strict=True) if i < j]

    shared_km = 0.1  # a shared edge is the full 1 km side; corner touches are 0
    edges: list[tuple[int, int]] = []
    for i, j in pairs:
        inter = geoms[i].intersection(geoms[j])
        if inter.length >= shared_km:
            edges.append((i, j))

    if not edges:
        raise ValueError("no rook neighbours found; check the grid geometry")
    src = np.array([e[0] for e in edges], dtype=int)
    dst = np.array([e[1] for e in edges], dtype=int)
    data = np.ones(2 * len(edges), dtype=np.int8)
    W = sparse.coo_matrix(
        (data, (np.concatenate([src, dst]), np.concatenate([dst, src]))),
        shape=(n, n),
    ).tocsr()

    n_comp, _ = connected_components(W, directed=False)
    if n_comp != 1:
        raise AssertionError(f"rook graph has {n_comp} connected components; must be connected")
    return W


def adjacency_edge_list(W: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Upper-triangular edge list (node1, node2) for an ICAR pairwise-difference prior."""
    coo = sparse.triu(W, k=1).tocoo()
    order = np.lexsort((coo.col, coo.row))
    return coo.row[order].astype(int), coo.col[order].astype(int)


def assign_regions(gdf: gpd.GeoDataFrame, cfg: SimConfig) -> np.ndarray:
    """Assign cells to 6 contiguous 2 x 3 rectangular regions (row-major numbering)."""
    row_blocks = 2
    col_blocks = 3
    rb = (gdf["row"].to_numpy() * row_blocks) // cfg.N_ROWS
    cb = (gdf["col"].to_numpy() * col_blocks) // cfg.N_COLS
    return (rb * col_blocks + cb).astype(int)
