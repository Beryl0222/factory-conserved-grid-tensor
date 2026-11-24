"""重映射引擎：瓦片合入、归属分配、对账与快照。

空间约束（对应 ``capacity.json`` 的 ``peak_scratch_bytes``）：

* 不构造源格 × 目标的配对矩阵，每个源格只通过区间索引访问候选目标；
* 不物化整张源格网的数值副本，数值保留在已接受瓦片内，重算时用一张
  获胜瓦片索引表（每格一个整数）解决同格多次更正的去重；
* 归约量按目标保存，与目标数线性相关。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import Rect, area, assign_by_priority, intersect
from .model import (
    DatasetView,
    Reconciliation,
    TargetResult,
    TargetSpec,
    ValueState,
)

_STATE_MISSING = -1


@dataclass(frozen=True)
class AcceptedTile:
    """一次被接受的瓦片交付（非拥有者格已在合入时剔除）。"""

    source: str
    rev: int
    r0: int
    r1: int
    c0: int
    c1: int
    valid: np.ndarray
    amount: np.ndarray
    intensity: np.ndarray


class Remapper:
    """持有采集资料并按数据集提供重映射视图。

    参数
    ----
    x, y:
        源格网轴边界（整数微米），长度分别为列数 + 1、行数 + 1。
    crop:
        公共裁剪窗 ``[x0, x1, y0, y1)``。
    owners:
        来源版本到其拥有者矩形索引窗 ``[r0, r1, c0, c1)`` 的映射。
    targets:
        目标声明列表，每项含 id/group/box/priority/coverage。
    default_dataset:
        未显式标注数据集的事件归属名称（关单记录会给出正式名称）。
    """

    def __init__(
        self,
        x: list[int],
        y: list[int],
        crop: list[int],
        owners: dict[str, list[int]],
        targets: list[dict],
        default_dataset: str = "dataset",
    ):
        self.x = [int(v) for v in x]
        self.y = [int(v) for v in y]
        self.cols = len(self.x) - 1
        self.rows = len(self.y) - 1
        self.crop = Rect(*(int(v) for v in crop))
        self.owners = {
            src: tuple(int(v) for v in window) for src, window in owners.items()
        }
        self.targets: list[TargetSpec] = [
            TargetSpec(
                id=item["id"],
                group=item["group"],
                box=Rect(*(int(v) for v in item["box"])),
                priority=int(item["priority"]),
                coverage_threshold=float(item["coverage"]),
            )
            for item in targets
        ]
        if len({t.id for t in self.targets}) != len(self.targets):
            raise ValueError("目标 ID 必须唯一")
        self._groups = self._index_groups()
        self._target_index = {t.id: t for t in self.targets}

        self._tiles: dict[str, list[AcceptedTile]] = {}
        self._priority_overrides: dict[str, dict[str, int]] = {}
        self._closed: set[str] = set()
        self._snapshots: dict[str, DatasetView] = {}
        self._cache: dict[str, DatasetView] = {}
        self._default_dataset = default_dataset

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_records(cls, records: dict) -> "Remapper":
        """按采集记录的静态资料构造引擎（不含事件与张量）。"""
        return cls(
            x=records["x"],
            y=records["y"],
            crop=records["crop"],
            owners=records["owners"],
            targets=records["targets"],
        )

    def _index_groups(self) -> dict[str, list[TargetSpec]]:
        groups: dict[str, list[TargetSpec]] = {}
        for target in self.targets:
            groups.setdefault(target.group, []).append(target)
        return groups

    # ------------------------------------------------------------------ 写入
    def _require_open(self, dataset: str) -> None:
        if dataset in self._closed:
            raise ValueError(f"数据集 {dataset} 已关单，来源版本不得再修改")

    def _owner_window(self, source: str) -> tuple[int, int, int, int]:
        if source not in self.owners:
            raise ValueError(f"未知来源版本 {source}")
        return self.owners[source]

    def deliver(
        self,
        dataset: str,
        source: str,
        rev: int,
        slice_: list[int],
        tensors: dict[str, np.ndarray],
        prefix: str,
    ) -> int:
        """合入一块瓦片，返回实际接受的拥有者格数。

        瓦片内不属于该来源拥有者窗的格被静默忽略；同一格的后到修订
        （更大的事件次序，rev 仅作版本登记）在重算时覆盖先前值。
        """
        self._require_open(dataset)
        r0, r1, c0, c1 = (int(v) for v in slice_)
        if not (0 <= r0 < r1 <= self.rows and 0 <= c0 < c1 <= self.cols):
            raise ValueError(f"瓦片索引窗越界: {slice_}")
        or0, or1, oc0, oc1 = self._owner_window(source)
        if not (or0 <= r0 and r1 <= or1 and oc0 <= c0 and c1 <= oc1):
            # 允许瓦片比拥有者窗大，但必须与拥有者窗相交；非拥有格后续忽略。
            if r1 <= or0 or r0 >= or1 or c1 <= oc0 or c0 >= oc1:
                raise ValueError(f"来源 {source} 不拥有瓦片 {slice_} 内的任何格")
        try:
            valid = np.asarray(tensors[f"{prefix}_valid"], dtype=bool)
            amount = np.asarray(tensors[f"{prefix}_amount"], dtype=np.float64)
            intensity = np.asarray(
                tensors[f"{prefix}_intensity"], dtype=np.float64
            )
        except KeyError as exc:
            raise ValueError(f"缺少字段张量: {exc.args[0]}") from exc
        shape = (r1 - r0, c1 - c0)
        for name, array in (("valid", valid), ("amount", amount),
                            ("intensity", intensity)):
            if array.shape != shape:
                raise ValueError(
                    f"{prefix}_{name} 形状 {array.shape} 与瓦片 {shape} 不一致"
                )
        tile = AcceptedTile(
            source=source,
            rev=int(rev),
            r0=r0, r1=r1, c0=c0, c1=c1,
            valid=valid,
            amount=amount,
            intensity=intensity,
        )
        self._tiles.setdefault(dataset, []).append(tile)
        self._cache.pop(dataset, None)
        accepted = 0
        for r in range(r0, r1):
            for c in range(c0, c1):
                if or0 <= r < or1 and oc0 <= c < oc1:
                    accepted += 1
        return accepted

    def update_priority(self, dataset: str, new_priority: dict[str, int]) -> None:
        """更正开放数据集内的目标归属优先级。"""
        self._require_open(dataset)
        for target_id in new_priority:
            if target_id not in self._target_index:
                raise ValueError(f"未知目标 {target_id}")
        self._priority_overrides.setdefault(dataset, {}).update(
            {k: int(v) for k, v in new_priority.items()}
        )
        self._cache.pop(dataset, None)

    def close(self, dataset: str) -> DatasetView:
        """关单：生成不可变、可回查的定量快照。"""
        view = self._build_view(dataset)
        payload = self._canonical_payload(view)
        snapshot_id = hashlib.sha256(payload).hexdigest()[:16]
        closed_view = DatasetView(
            dataset=dataset,
            is_preview=False,
            targets=view.targets,
            reconciliations=view.reconciliations,
            pending_cells=view.pending_cells,
            delivered_cells=view.delivered_cells,
            snapshot_id=snapshot_id,
            closed=True,
        )
        self._closed.add(dataset)
        self._snapshots[dataset] = closed_view
        self._cache[dataset] = closed_view
        return closed_view

    def ingest(self, records: dict, tensors: dict[str, np.ndarray]) -> dict[str, DatasetView]:
        """回放一份采集资料的完整事件流，返回各数据集最终视图。

        未标注数据集的事件先缓冲，直到下一条关单记录给出正式名称；
        显式标注 ``dataset`` 的事件直接合入对应数据集。
        """
        current: str | None = None
        buffered: list[dict] = []
        results: dict[str, DatasetView] = {}

        def flush(dataset: str) -> None:
            for buffered_event in buffered:
                if "tile" in buffered_event:
                    self._apply_tile_event(dataset, buffered_event, tensors)
                else:
                    self.update_priority(
                        dataset, buffered_event["new_priority"]
                    )
            buffered.clear()

        for event in records["inputs"]:
            if "tile" in event or "new_priority" in event:
                if event.get("dataset") is None:
                    buffered.append(event)
                elif "tile" in event:
                    self._apply_tile_event(event["dataset"], event, tensors)
                else:
                    self.update_priority(event["dataset"], event["new_priority"])
            elif event.get("close_sources"):
                name = event["dataset"]
                flush(name)
                results[name] = self.close(name)
                current = name
            else:
                raise ValueError(f"无法识别的交换记录: {event}")
        if buffered:
            name = current or self._default_dataset
            flush(name)
            results[name] = self.view(name)
        for dataset in self._tiles:
            results.setdefault(dataset, self.view(dataset))
        return results

    def _apply_tile_event(
        self, dataset: str, event: dict, tensors: dict[str, np.ndarray]
    ) -> None:
        self.deliver(
            dataset=dataset,
            source=event["source"],
            rev=event["rev"],
            slice_=event["slice"],
            tensors=tensors,
            prefix=event["fields"],
        )

    # ------------------------------------------------------------------ 读取
    def view(self, dataset: str) -> DatasetView:
        """取得数据集当前视图；已关单返回快照，未关单返回预览。"""
        if dataset in self._snapshots:
            return self._snapshots[dataset]
        if dataset in self._cache:
            return self._cache[dataset]
        view = self._build_view(dataset)
        self._cache[dataset] = view
        return view

    def snapshot(self, dataset: str) -> DatasetView:
        """按数据集名称回查关单快照。"""
        if dataset not in self._snapshots:
            raise KeyError(f"数据集 {dataset} 没有关单快照")
        return self._snapshots[dataset]

    @property
    def closed_datasets(self) -> tuple[str, ...]:
        return tuple(sorted(self._closed))

    # ------------------------------------------------------------ 重算核心
    def _priority_of(self, target: TargetSpec, dataset: str) -> int:
        return self._priority_overrides.get(dataset, {}).get(
            target.id, target.priority
        )

    def _ordered_targets(self, dataset: str) -> dict[str, list[TargetSpec]]:
        ordered = {}
        for group, members in self._groups.items():
            ordered[group] = sorted(
                members,
                key=lambda t: (-self._priority_of(t, dataset), t.id),
            )
        return ordered

    def _build_view(self, dataset: str) -> DatasetView:
        tiles = self._tiles.get(dataset, [])
        ordered_groups = self._ordered_targets(dataset)

        # 获胜瓦片索引：同一格被多次交付时，事件次序靠后者胜出。
        # -1 表示从未交付；int8 可容纳 127 个瓦片，超出则升级 int32。
        winner_dtype = np.int8 if len(tiles) <= 126 else np.int32
        winner = np.full((self.rows, self.cols), _STATE_MISSING, dtype=winner_dtype)
        for tile_idx, tile in enumerate(tiles):
            or0, or1, oc0, oc1 = self.owners[tile.source]
            for r in range(tile.r0, tile.r1):
                if not (or0 <= r < or1):
                    continue
                for c in range(tile.c0, tile.c1):
                    if oc0 <= c < oc1:
                        winner[r, c] = tile_idx

        group_acc = {
            group: {
                "denom": np.zeros(len(members), dtype=np.int64),
                "eff": np.zeros(len(members), dtype=np.int64),
                "amount": np.zeros(len(members), dtype=np.float64),
                "wnum": np.zeros(len(members), dtype=np.float64),
                "sources": [set() for _ in members],
            }
            for group, members in ordered_groups.items()
        }

        effective_amount = 0.0
        effective_area = 0
        delivered_cells = 0
        pending_cells = 0

        # 预计算每个源格是否与任一目标有正面积接触（裁剪窗内），
        # 供等待资料计数使用，避免在每个互斥组内重复判定。
        touches_any_target = np.zeros((self.rows, self.cols), dtype=bool)
        for r in range(self.rows):
            cell_y0, cell_y1 = self.y[r], self.y[r + 1]
            for c in range(self.cols):
                cell = Rect(self.x[c], self.x[c + 1], cell_y0, cell_y1)
                if intersect(cell, self.crop) is None:
                    continue
                touches_any_target[r, c] = any(
                    intersect(cell, target.box) is not None
                    for target in self.targets
                )

        for r in range(self.rows):
            cell_y0, cell_y1 = self.y[r], self.y[r + 1]
            for c in range(self.cols):
                cell = Rect(self.x[c], self.x[c + 1], cell_y0, cell_y1)
                tile_idx = int(winner[r, c])
                if tile_idx == _STATE_MISSING:
                    # 从未交付的格：若几何上能服务任一目标，计入等待资料。
                    if touches_any_target[r, c]:
                        pending_cells += 1
                    continue
                delivered_cells += 1
                tile = tiles[tile_idx]
                lr, lc = r - tile.r0, c - tile.c0
                is_valid = bool(tile.valid[lr, lc])
                cell_area = area(cell)
                cell_amount = float(tile.amount[lr, lc])
                cell_intensity = float(tile.intensity[lr, lc])
                if is_valid:
                    effective_amount += cell_amount
                    effective_area += cell_area

                inside = intersect(cell, self.crop)
                if inside is None:
                    # 裁剪窗外的有效量无法分配，自然留在各组未覆盖余量中。
                    continue

                touches_target = False
                for group, members in ordered_groups.items():
                    boxes = [t.box for t in members]
                    fragments = assign_by_priority(inside, boxes)
                    acc = group_acc[group]
                    for target_pos, fragment in fragments:
                        frag_area = area(fragment)
                        acc["denom"][target_pos] += frag_area
                        touches_target = True
                        if not is_valid:
                            continue
                        ratio = frag_area / cell_area
                        acc["eff"][target_pos] += frag_area
                        acc["amount"][target_pos] += cell_amount * ratio
                        acc["wnum"][target_pos] += cell_intensity * frag_area
                        acc["sources"][target_pos].add(tile.source)
                # 已交付但失效、且确实压到某个目标的格，仍是等待有效资料。
                if not is_valid and touches_target:
                    pending_cells += 1

        results: dict[str, TargetResult] = {}
        reconciliations: dict[str, Reconciliation] = {}
        for group, members in ordered_groups.items():
            acc = group_acc[group]
            allocated = 0.0
            covered_area = 0
            for pos, target in enumerate(members):
                denom = int(acc["denom"][pos])
                eff = int(acc["eff"][pos])
                amount_sum = float(acc["amount"][pos])
                intensity = (
                    float(acc["wnum"][pos]) / eff if eff > 0 else None
                )
                amount_value: float | None
                intensity_value: float | None
                if denom == 0:
                    amount_state = intensity_state = ValueState.NO_COVERAGE
                    amount_value = intensity_value = None
                    coverage_ratio = None
                elif eff == 0:
                    amount_state = intensity_state = ValueState.PENDING
                    amount_value = intensity_value = None
                    coverage_ratio = 0.0
                else:
                    amount_value = amount_sum
                    intensity_value = intensity
                    amount_state = (
                        ValueState.ZERO if amount_value == 0.0 else ValueState.MEASURED
                    )
                    intensity_state = (
                        ValueState.ZERO if intensity_value == 0.0 else ValueState.MEASURED
                    )
                    coverage_ratio = eff / denom
                result = TargetResult(
                    id=target.id,
                    group=group,
                    box=target.box,
                    denominator_area=denom,
                    effective_overlap_area=eff,
                    amount=amount_value,
                    intensity=intensity_value,
                    amount_state=amount_state,
                    intensity_state=intensity_state,
                    coverage_ratio=coverage_ratio,
                    coverage_threshold=target.coverage_threshold,
                    sources=tuple(sorted(acc["sources"][pos])),
                )
                results[target.id] = result
                allocated += amount_sum
                covered_area += eff
            uncovered_amount = effective_amount - allocated
            uncovered_area = effective_area - covered_area
            residual = (allocated + uncovered_amount) - effective_amount
            reconciliations[group] = Reconciliation(
                group=group,
                effective_source_amount=effective_amount,
                effective_source_area=effective_area,
                allocated_amount=allocated,
                uncovered_amount=uncovered_amount,
                covered_effective_area=covered_area,
                uncovered_effective_area=uncovered_area,
                residual=residual,
            )

        return DatasetView(
            dataset=dataset,
            is_preview=dataset not in self._closed,
            targets=results,
            reconciliations=reconciliations,
            pending_cells=pending_cells,
            delivered_cells=delivered_cells,
        )

    # ------------------------------------------------------------ 序列化
    @staticmethod
    def _canonical_payload(view: DatasetView) -> bytes:
        return json.dumps(
            view_to_json(view),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def snapshot_hash(self, dataset: str) -> str:
        return self.snapshot(dataset).snapshot_id  # type: ignore[return-value]


def view_to_json(view: DatasetView) -> dict:
    """把视图转成可交给模型团队的 JSON 友好结构（三态显式区分）。"""
    return {
        "dataset": view.dataset,
        "is_preview": view.is_preview,
        "closed": view.closed,
        "snapshot_id": view.snapshot_id,
        "pending_cells": view.pending_cells,
        "delivered_cells": view.delivered_cells,
        "targets": [
            {
                "id": target.id,
                "group": target.group,
                "box": target.box.as_list(),
                "denominator_area": target.denominator_area,
                "effective_overlap_area": target.effective_overlap_area,
                "coverage_ratio": target.coverage_ratio,
                "coverage_threshold": target.coverage_threshold,
                "covered": target.covered,
                "sources": list(target.sources),
                "amount": _value_entry(target.amount, target.amount_state),
                "intensity": _value_entry(
                    target.intensity, target.intensity_state
                ),
            }
            for target in sorted(view.targets.values(), key=lambda t: t.id)
        ],
        "reconciliation": [
            {
                "group": rec.group,
                "effective_source_amount": rec.effective_source_amount,
                "effective_source_area": rec.effective_source_area,
                "allocated_amount": rec.allocated_amount,
                "uncovered_amount": rec.uncovered_amount,
                "covered_effective_area": rec.covered_effective_area,
                "uncovered_effective_area": rec.uncovered_effective_area,
                "residual": rec.residual,
                "balanced": rec.balanced,
            }
            for rec in sorted(
                view.reconciliations.values(), key=lambda r: r.group
            )
        ],
    }


def _value_entry(value: float | None, state: ValueState) -> dict:
    return {"state": state.value, "value": value}


def write_view(view: DatasetView, path: Path) -> None:
    """把视图写成 UTF-8 JSON，便于下游交接与回查。"""
    path.write_text(
        json.dumps(view_to_json(view), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
