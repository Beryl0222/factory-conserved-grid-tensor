"""工作区容量规划：按制造端规模估算峰值暂存字节。

口径（与 README 一致）：只读输入和已消费输出不计入工作区，
引擎内部为处理而分配的索引、瓦片副本与归约缓冲计入。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityPlan:
    rows: int
    cols: int
    target_count: int
    tiles: int
    groups: int
    bytes_per_cell_grid: int
    peak_scratch_bytes: int
    limit_bytes: int

    @property
    def fits(self) -> bool:
        return self.peak_scratch_bytes <= self.limit_bytes


def plan_capacity(
    rows: int,
    cols: int,
    target_count: int,
    tiles: int,
    groups: int,
    limit_bytes: int,
) -> CapacityPlan:
    """估算最坏工作区占用。

    组成部分：

    * 每格两张 1 字节格网：获胜瓦片索引（int8）与目标接触判定（bool）；
    * 目标归约器：每目标 2 个 int64（分母/有效面积）、2 个 float64
      （总量/强度权值），按互斥组分摊；
    * 瓦片暂存：每瓦片按全网格 1/tiles 估算一份 valid/amount/intensity；
    * 几何与事件缓冲的固定开销。

    明确不包含源格 × 目标的配对矩阵——算法只做每格的候选查询。
    """
    cells = rows * cols
    # int8 获胜索引 + bool 目标接触判定，各 1 字节/格。
    grid_bytes = cells * 2
    reducer_bytes = target_count * (2 * 8 + 2 * 8)
    per_tile_cells = max(1, cells // max(tiles, 1))
    tile_scratch = tiles * per_tile_cells * (1 + 8 + 8)
    fixed_overhead = 2 * 1024 * 1024  # 事件、几何碎片与输出缓冲
    peak = grid_bytes + reducer_bytes + tile_scratch + fixed_overhead
    return CapacityPlan(
        rows=rows,
        cols=cols,
        target_count=target_count,
        tiles=tiles,
        groups=groups,
        bytes_per_cell_grid=grid_bytes // max(cells, 1),
        peak_scratch_bytes=peak,
        limit_bytes=limit_bytes,
    )
