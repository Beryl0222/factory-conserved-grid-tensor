"""重映射引擎的端到端与规则测试。"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from caseio import load_case
from remapper import Remapper, ValueState
from remapper.capacity import plan_capacity
from remapper.engine import view_to_json
from remapper.geometry import Rect, area, assign_by_priority, intersect, subtract

DATA = Path(__file__).resolve().parents[1] / "data"


def simple_engine(
    targets,
    x=(0, 4, 10),
    y=(0, 3, 9),
    crop=(0, 10, 0, 9),
    owners=None,
):
    if owners is None:
        owners = {"s": [0, 2, 0, 2]}
    return Remapper(
        x=list(x), y=list(y), crop=list(crop),
        owners=owners, targets=targets,
        default_dataset="ds",
    )


def full_tile(valid, amount, intensity):
    return {
        "f_valid": np.asarray(valid, dtype=bool),
        "f_amount": np.asarray(amount, dtype=np.float64),
        "f_intensity": np.asarray(intensity, dtype=np.float64),
    }


class GeometryTests(unittest.TestCase):
    def test_zero_area_contact_is_not_coverage(self):
        self.assertIsNone(
            intersect(Rect(0, 4, 0, 3), Rect(4, 8, 0, 3))
        )
        self.assertIsNone(
            intersect(Rect(0, 4, 0, 3), Rect(0, 4, 3, 9))
        )

    def test_subtract_keeps_positive_area_only(self):
        pieces = subtract(Rect(0, 10, 0, 8), Rect(2, 6, 3, 5))
        self.assertTrue(all((p.x1 - p.x0) * (p.y1 - p.y0) > 0 for p in pieces))
        covered = sum((p.x1 - p.x0) * (p.y1 - p.y0) for p in pieces)
        # 两条纵条各 32，中间上下各 4 -> 共 72。
        self.assertEqual(covered, 72)

    def test_priority_then_stable_id(self):
        # 同优先级时较小 ID 先占有；不同优先级时高优先级先占有。
        region = Rect(0, 10, 0, 10)
        boxes = [Rect(0, 10, 0, 10), Rect(0, 10, 0, 10)]
        owned = assign_by_priority(region, boxes)
        self.assertEqual([pos for pos, _ in owned], [0])
        self.assertEqual(area(owned[0][1]), 100)

    def test_stable_id_tiebreak_at_engine_level(self):
        engine = simple_engine([
            {"id": "z", "group": "g", "box": [0, 10, 0, 9],
             "priority": 1, "coverage": 0.0},
            {"id": "a", "group": "g", "box": [0, 10, 0, 9],
             "priority": 1, "coverage": 0.0},
        ])
        engine.deliver("ds", "s", 1, [0, 2, 0, 2], full_tile(
            [[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]],
        ), prefix="f")
        view = engine.view("ds")
        self.assertEqual(view.target("a").denominator_area, 90)
        self.assertEqual(view.target("z").denominator_area, 0)
        self.assertEqual(
            view.target("z").amount_state, ValueState.NO_COVERAGE
        )

    def test_winner_index_upgrades_past_int8_range(self):
        engine = simple_engine([
            {"id": "g1", "group": "g", "box": [0, 10, 0, 9],
             "priority": 1, "coverage": 0.0},
        ])
        for rev in range(1, 130):
            engine.deliver("ds", "s", rev, [0, 1, 0, 1], {
                "f_valid": np.ones((1, 1), dtype=bool),
                "f_amount": np.full((1, 1), float(rev)),
                "f_intensity": np.ones((1, 1)),
            }, prefix="f")
        view = engine.view("ds")
        # 129 次同格修订后最后一次胜出，无 int8 溢出。
        self.assertEqual(view.delivered_cells, 1)
        self.assertAlmostEqual(view.target("g1").amount, 129.0)


class FixtureReplayTests(unittest.TestCase):
    def setUp(self):
        self.records, self.tensors = load_case(DATA / "capture.json")
        self.remapper = Remapper.from_records(self.records)
        self.views = self.remapper.ingest(self.records, self.tensors)

    def test_closed_snapshot_and_preview_identity(self):
        scan1 = self.views["scan1"]
        scan2 = self.views["scan2"]
        self.assertTrue(scan1.closed)
        self.assertFalse(scan1.is_preview)
        self.assertIsNotNone(scan1.snapshot_id)
        self.assertFalse(scan2.closed)
        self.assertTrue(scan2.is_preview)
        self.assertIsNone(scan2.snapshot_id)
        # 快照可按名称回查，且内容稳定。
        again = self.remapper.snapshot("scan1")
        self.assertEqual(again.snapshot_id, scan1.snapshot_id)
        self.assertEqual(view_to_json(again), view_to_json(scan1))

    def test_closed_source_is_immutable(self):
        with self.assertRaises(ValueError):
            self.remapper.deliver(
                "scan1", "scan-a", 9, [0, 2, 0, 2],
                self.tensors, prefix="a",
            )
        with self.assertRaises(ValueError):
            self.remapper.update_priority("scan1", {"t01": 9})

    def test_scan1_values_and_reconciliation(self):
        view = self.views["scan1"]
        t01, t02, t05 = view.target("t01"), view.target("t02"), view.target("t05")
        self.assertAlmostEqual(t01.amount, 149 / 15)
        self.assertAlmostEqual(t01.intensity, 43 / 23)
        self.assertEqual(t01.denominator_area, 23)
        self.assertAlmostEqual(t02.amount, 15.7)
        self.assertAlmostEqual(t02.intensity, 24 / 7)
        self.assertAlmostEqual(t05.amount, 100 / 9)
        self.assertAlmostEqual(t05.intensity, 43 / 18)
        for rec in view.reconciliations.values():
            self.assertTrue(rec.balanced)
            self.assertAlmostEqual(
                rec.allocated_amount + rec.uncovered_amount,
                rec.effective_source_amount,
            )
        # 有效源总量 34：两个组各自独立对账。
        self.assertEqual(
            {g: rec.effective_source_amount
             for g, rec in view.reconciliations.items()},
            {"locate": 34.0, "marked": 34.0},
        )

    def test_correction_invalidates_cell_and_priority_flips_ownership(self):
        view = self.views["scan2"]
        # (1,1) 失效：t02 的全部归属区域失去有效资料。
        t02 = view.target("t02")
        self.assertEqual(t02.amount_state, ValueState.PENDING)
        self.assertIsNone(t02.amount)
        self.assertFalse(t02.covered)
        # 优先级更正后 t01 先占有，分母从 25 扩大到 35。
        t01 = view.target("t01")
        self.assertEqual(t01.denominator_area, 35)
        self.assertAlmostEqual(t01.amount, 35 / 3)
        self.assertAlmostEqual(t01.coverage_ratio, 30 / 35)
        self.assertTrue(t01.covered)
        # t05 覆盖率掉到阈值之下，告警不可定量使用。
        t05 = view.target("t05")
        self.assertAlmostEqual(t05.coverage_ratio, 24 / 36)
        self.assertFalse(t05.covered)
        # 对账仍然守恒：面积损失在未覆盖余量中可见。
        for rec in view.reconciliations.values():
            self.assertTrue(rec.balanced)
        self.assertEqual(view.pending_cells, 3)

    def test_owner_filtering_rejects_non_owner_cells(self):
        # scan-b 只拥有列 2；送一块覆盖列 1-2 的瓦片，列 1 必须被忽略。
        remapper = Remapper.from_records(self.records)
        accepted = remapper.deliver(
            "scan1", "scan-b", 1, [0, 2, 1, 3],
            self.tensors, prefix="b",
        )
        self.assertEqual(accepted, 2)
        view = remapper.view("scan1")
        # 只有列 2 的两个格完成交付；其余四格未交齐且压在目标上。
        self.assertEqual(view.delivered_cells, 2)
        self.assertEqual(view.pending_cells, 4)
        # 列 1 的值不得从 scan-b 泄漏到任何目标。
        for target in view.targets.values():
            self.assertEqual(set(target.sources) - {"scan-b"}, set())

    def test_tile_outside_owner_window_is_rejected(self):
        remapper = Remapper.from_records(self.records)
        with self.assertRaises(ValueError):
            remapper.deliver(
                "scan1", "scan-b", 1, [0, 2, 0, 2],
                self.tensors, prefix="b",
            )

    def test_json_distinguishes_three_states(self):
        payload = view_to_json(self.views["scan2"])
        entries = {t["id"]: t for t in payload["targets"]}
        self.assertEqual(entries["t02"]["amount"]["state"], "pending")
        self.assertIsNone(entries["t02"]["amount"]["value"])
        # 有效覆盖但数值为零：t05 的强度恰为 0。
        self.assertEqual(entries["t05"]["intensity"]["state"], "zero")
        self.assertEqual(entries["t05"]["intensity"]["value"], 0.0)


class RemapRuleTests(unittest.TestCase):
    def test_amount_conservation_with_area_ratios(self):
        engine = simple_engine([
            {"id": "g1", "group": "g", "box": [0, 8, 0, 9],
             "priority": 1, "coverage": 0.5},
        ])
        engine.deliver("ds", "s", 1, [0, 2, 0, 2], full_tile(
            [[1, 1], [1, 1]], [[12, 30], [24, 60]], [[1, 2], [3, 4]],
        ), prefix="f")
        view = engine.view("ds")
        target = view.target("g1")
        # 格面积：12,18,24,36；目标在 x=8 处切开第二列（保留 2/3）。
        expected = 12 + 24 + 20 + 40
        self.assertAlmostEqual(target.amount, expected)
        rec = view.reconciliations["g"]
        self.assertTrue(rec.balanced)
        self.assertAlmostEqual(rec.effective_source_amount, 126.0)
        # 切在目标外的份额进入未覆盖对账（第二列各损失 1/3）。
        self.assertAlmostEqual(rec.uncovered_amount, 30.0)

    def test_intensity_is_area_weighted_mean(self):
        engine = simple_engine([
            {"id": "g1", "group": "g", "box": [0, 10, 0, 9],
             "priority": 1, "coverage": 0.01},
        ])
        engine.deliver("ds", "s", 1, [0, 2, 0, 2], full_tile(
            [[1, 1], [1, 1]], [[0, 0], [0, 0]], [[2, 8], [0, 0]],
        ), prefix="f")
        target = engine.view("ds").target("g1")
        # 左上格强度 2、右上格强度 8；按有效重叠面积（=格面积）加权。
        self.assertAlmostEqual(target.intensity, (24 + 144) / 90)
        self.assertEqual(target.intensity_state, ValueState.MEASURED)
        # 总量全零但有覆盖 -> zero 而非 no_coverage。
        self.assertEqual(target.amount_state, ValueState.ZERO)
        self.assertEqual(target.amount, 0.0)

    def test_no_coverage_vs_pending_vs_zero(self):
        engine = simple_engine([
            {"id": "inside", "group": "g", "box": [0, 10, 0, 9],
             "priority": 1, "coverage": 0.9},
            {"id": "outside", "group": "g", "box": [20, 30, 20, 30],
             "priority": 1, "coverage": 0.5},
        ])
        engine.deliver("ds", "s", 1, [0, 2, 0, 2], full_tile(
            [[1, 0], [0, 0]], [[5, 9], [9, 9]], [[1, 9], [9, 9]],
        ), prefix="f")
        view = engine.view("ds")
        inside = view.target("inside")
        outside = view.target("outside")
        # 左上格有效 -> measured；其余三格已交付但失效且压在目标上，
        # 有效覆盖率 12/90，低于阈值 -> 不可定量；失效格计入 pending。
        self.assertEqual(inside.amount_state, ValueState.MEASURED)
        self.assertAlmostEqual(inside.coverage_ratio, 12 / 90)
        self.assertFalse(inside.covered)
        self.assertEqual(inside.denominator_area, 90)
        self.assertEqual(view.pending_cells, 3)
        self.assertEqual(outside.amount_state, ValueState.NO_COVERAGE)
        self.assertEqual(outside.intensity_state, ValueState.NO_COVERAGE)
        self.assertIsNone(outside.coverage_ratio)

    def test_groups_are_independent_and_priority_uses_stable_id(self):
        engine = simple_engine([
            {"id": "a", "group": "g1", "box": [0, 10, 0, 9],
             "priority": 1, "coverage": 0.0},
            {"id": "b", "group": "g2", "box": [0, 10, 0, 9],
             "priority": 1, "coverage": 0.0},
        ])
        engine.deliver("ds", "s", 1, [0, 2, 0, 2], full_tile(
            [[1, 1], [1, 1]], [[10, 10], [10, 10]], [[1, 1], [1, 1]],
        ), prefix="f")
        view = engine.view("ds")
        # 两个互斥组各自拿到完整有效源量，互不扣减。
        for tid in ("a", "b"):
            self.assertAlmostEqual(view.target(tid).amount, 40.0)
        self.assertTrue(view.reconciliations["g1"].balanced)
        self.assertTrue(view.reconciliations["g2"].balanced)

    def test_revision_correction_then_snapshot_is_replayable(self):
        engine = simple_engine([
            {"id": "g1", "group": "g", "box": [0, 10, 0, 9],
             "priority": 1, "coverage": 0.01},
        ])
        engine.deliver("ds", "s", 1, [0, 2, 0, 2], full_tile(
            [[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]],
        ), prefix="f")
        engine.deliver("ds", "s", 2, [0, 2, 0, 2], full_tile(
            [[1, 1], [1, 0]], [[2, 2], [2, 9]], [[3, 3], [3, 9]],
        ), prefix="f")
        view = engine.view("ds")
        # 修订胜出：三格有效，每格量 2；失效格不贡献。
        self.assertAlmostEqual(view.target("g1").amount, 6.0)
        self.assertEqual(view.pending_cells, 1)
        snap = engine.close("ds")
        self.assertEqual(engine.snapshot("ds").snapshot_id, snap.snapshot_id)
        # 快照 JSON 可回查、可解析。
        payload = json.loads(json.dumps(view_to_json(snap)))
        self.assertEqual(payload["snapshot_id"], snap.snapshot_id)


class CapacityTests(unittest.TestCase):
    def test_plan_respects_scratch_limit_without_pair_matrix(self):
        cap = json.loads((DATA / "capacity.json").read_text(encoding="utf-8"))
        plan = plan_capacity(
            rows=cap["rows"], cols=cap["cols"],
            target_count=cap["target_count"], tiles=cap["tiles"],
            groups=cap["groups"], limit_bytes=cap["peak_scratch_bytes"],
        )
        self.assertTrue(plan.fits)
        # 全配对矩阵需要 cells × target_count 个 float64，远超限额。
        pair_matrix = cap["rows"] * cap["cols"] * cap["target_count"] * 8
        self.assertGreater(pair_matrix, cap["peak_scratch_bytes"])


if __name__ == "__main__":
    unittest.main()
