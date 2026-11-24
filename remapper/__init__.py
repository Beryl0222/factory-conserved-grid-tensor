"""面向不等宽栅格瓦片的保量张量重映射。

核心能力：

* 局部瓦片按来源版本声明的拥有者合入全局源格网，缺测与失效显式区分；
* 总量型按实际重叠面积比例分配并整组对账，强度型按有效重叠面积加权；
* 互斥目标组内按优先级、再按稳定 ID 归属，组间互不影响；
* 关单生成不可变、可回查的定量快照，未关单的视图标明预览身份。

入口类为 :class:`~remapper.engine.Remapper`。
"""
from __future__ import annotations

from .engine import Remapper
from .model import (
    DatasetView,
    Reconciliation,
    TargetResult,
    ValueState,
)

__all__ = [
    "Remapper",
    "DatasetView",
    "Reconciliation",
    "TargetResult",
    "ValueState",
]
