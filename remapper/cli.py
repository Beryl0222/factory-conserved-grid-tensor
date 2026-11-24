"""命令行：重放采集资料并输出各数据集视图与对账结果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .capacity import plan_capacity
from .engine import Remapper, view_to_json, write_view
from .model import DatasetView
from caseio import load_case


def _print_view(view: DatasetView) -> None:
    head = "快照" if view.closed else "预览"
    print(f"== 数据集 {view.dataset}（{head}）"
          + (f" snapshot={view.snapshot_id}" if view.snapshot_id else ""))
    print(f"已交付格 {view.delivered_cells}，等待有效资料格 {view.pending_cells}")
    for target in sorted(view.targets.values(), key=lambda t: (t.group, t.id)):
        amount = "—" if target.amount is None else f"{target.amount:.6f}"
        intensity = (
            "—" if target.intensity is None else f"{target.intensity:.6f}"
        )
        coverage = (
            "—" if target.coverage_ratio is None else f"{target.coverage_ratio:.4f}"
        )
        print(
            f"  [{target.group}] {target.id} "
          f"amount={amount}({target.amount_state.value}) "
          f"intensity={intensity}({target.intensity_state.value}) "
          f"coverage={coverage}/{target.coverage_threshold} "
          f"usable={target.covered} sources={','.join(target.sources) or '-'}"
        )
    for rec in sorted(view.reconciliations.values(), key=lambda r: r.group):
        print(
            f"  对账[{rec.group}] 有效源量={rec.effective_source_amount:.6f} "
          f"分配={rec.allocated_amount:.6f} 未覆盖={rec.uncovered_amount:.6f} "
          f"残差={rec.residual:.2e} 平衡={rec.balanced}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="栅格瓦片保量重映射")
    parser.add_argument("capture", type=Path, help="采集资料 JSON")
    parser.add_argument("--capacity", type=Path, help="容量条件 JSON")
    parser.add_argument("--out", type=Path, help="把视图 JSON 写入该目录")
    args = parser.parse_args()

    records, tensors = load_case(args.capture)
    remapper = Remapper.from_records(records)
    views = remapper.ingest(records, tensors)
    for view in sorted(views.values(), key=lambda v: v.dataset):
        _print_view(view)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for view in views.values():
            write_view(view, args.out / f"{view.dataset}.json")

    if args.capacity is not None:
        cap = json.loads(args.capacity.read_text(encoding="utf-8"))
        plan = plan_capacity(
            rows=cap["rows"],
            cols=cap["cols"],
            target_count=cap["target_count"],
            tiles=cap["tiles"],
            groups=cap["groups"],
            limit_bytes=cap["peak_scratch_bytes"],
        )
        print(
            f"容量估算 峰值={plan.peak_scratch_bytes}B "
          f"限额={plan.limit_bytes}B 可行={plan.fits}"
        )


if __name__ == "__main__":
    main()
