"""整数微米坐标下的矩形几何。

所有区间左闭右开，坐标为整数微米。边界相接（零面积）一律视为不覆盖，
因此重叠面积必须严格为正才会计入。
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import Rect


def intersect(a: Rect, b: Rect) -> Rect | None:
    """返回两个矩形的严格正面积交集，零面积接触返回 ``None``。"""
    x0 = max(a.x0, b.x0)
    x1 = min(a.x1, b.x1)
    y0 = max(a.y0, b.y0)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return Rect(x0, x1, y0, y1)


def area(rect: Rect) -> int:
    """矩形面积（平方微米）。"""
    return (rect.x1 - rect.x0) * (rect.y1 - rect.y0)


def subtract(rect: Rect, hole: Rect) -> list[Rect]:
    """从 ``rect`` 中扣除 ``hole``，返回互不重叠的剩余矩形。

    先沿 x 切成左/中/右三条，再把中间一条沿 y 切成下/上，
    保证碎片互不重叠、零面积者丢弃，且切割顺序确定。
    """
    cut = intersect(rect, hole)
    if cut is None:
        return [rect]
    strips: list[Rect] = []
    if rect.x0 < cut.x0:
        strips.append(Rect(rect.x0, cut.x0, rect.y0, rect.y1))
    if cut.x1 < rect.x1:
        strips.append(Rect(cut.x1, rect.x1, rect.y0, rect.y1))
    if rect.y0 < cut.y0:
        strips.append(Rect(cut.x0, cut.x1, rect.y0, cut.y0))
    if cut.y1 < rect.y1:
        strips.append(Rect(cut.x0, cut.x1, cut.y1, rect.y1))
    return strips


def assign_by_priority(region: Rect, targets: list[Rect]) -> list[tuple[int, Rect]]:
    """把 ``region`` 在同一互斥组内按目标顺序归属。

    ``targets`` 必须已按优先级降序、稳定 ID 升序排好。每个目标只拿到
    尚未被更高优先级占有的部分；返回 ``(目标在输入列表中的序号, 碎片)``。
    """
    owned: list[tuple[int, Rect]] = []
    pending = [region]
    for idx, box in enumerate(targets):
        if not pending:
            break
        remaining: list[Rect] = []
        for piece in pending:
            overlap = intersect(piece, box)
            if overlap is not None:
                owned.append((idx, overlap))
            remaining.extend(subtract(piece, box))
        pending = remaining
    return owned
