"""数据结构与输出状态定义。"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rect:
    """左闭右开的微米矩形 ``[x0, x1) × [y0, y1)``。"""

    x0: int
    x1: int
    y0: int
    y1: int

    def as_list(self) -> list[int]:
        return [self.x0, self.x1, self.y0, self.y1]


class ValueState(str, enum.Enum):
    """模型团队需要的三种资料状态。

    * ``NO_COVERAGE``：归属区域内没有任何源格覆盖（空区域不产生输出）；
    * ``ZERO``：有有效覆盖，且重映射后的数值本身就是零；
    * ``PENDING``：覆盖到的源格尚未交齐有效资料，等待中不能定量；
    * ``MEASURED``：有有效覆盖且数值非零，可用于定量判断。
    """

    NO_COVERAGE = "no_coverage"
    ZERO = "zero"
    PENDING = "pending"
    MEASURED = "measured"


@dataclass(frozen=True)
class TargetSpec:
    """采集资料声明的一个目标框。"""

    id: str
    group: str
    box: Rect
    priority: int
    coverage_threshold: float


@dataclass
class Reconciliation:
    """单个互斥组的对账结果。

    恒等式（容差 1e-9）：
    ``sum(allocated_amount) + uncovered_amount == effective_source_amount``，
    裁剪窗外的有效源量计入 ``uncovered_amount``；面积侧同理。
    """

    group: str
    effective_source_amount: float
    effective_source_area: int
    allocated_amount: float
    uncovered_amount: float
    covered_effective_area: int
    uncovered_effective_area: int
    residual: float = 0.0

    @property
    def balanced(self) -> bool:
        return abs(self.residual) <= 1e-9


@dataclass
class TargetResult:
    """单个目标格的两类张量与覆盖判定。"""

    id: str
    group: str
    box: Rect
    denominator_area: int
    """组内扣除更高优先级占有后的归属分母面积（裁剪窗内）。"""
    effective_overlap_area: int
    """归属碎片上实际拿到有效源资料的面积。"""
    amount: float | None
    intensity: float | None
    amount_state: ValueState
    intensity_state: ValueState
    coverage_ratio: float | None
    coverage_threshold: float
    sources: tuple[str, ...]
    """贡献了有效面积的来源版本。"""

    @property
    def covered(self) -> bool:
        """是否达到声明的有效覆盖率阈值。"""
        return (
            self.coverage_ratio is not None
            and self.coverage_ratio >= self.coverage_threshold
        )


@dataclass
class DatasetView:
    """某个数据集在一个时点的完整重映射视图。"""

    dataset: str
    is_preview: bool
    """True 表示数据集仍开放、目标归属仍可能因瓦片更正而变化。"""
    targets: dict[str, TargetResult]
    reconciliations: dict[str, Reconciliation]
    pending_cells: int
    """与任一目标重叠、尚未交齐有效资料的源格数量。"""
    delivered_cells: int
    snapshot_id: str | None = None
    """关单快照的可回查标识；预览视图为 ``None``。"""
    closed: bool = False

    def target(self, target_id: str) -> TargetResult:
        return self.targets[target_id]
