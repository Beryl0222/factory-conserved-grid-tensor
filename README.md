# 混合网格的保量张量重映射

两张合成目标格网给出检验部门和定位部门各自认可的坐标，双方独立对账。数据为本地合成。

- `data/capture.json` 保存采集原始内容。records 是交换记录，tensors 字段分开描述名称、类型、形状和值。整数索引从零开始，未指定闭合的区间均为左闭右开。
- `data/capacity.json` 保存制造端的容量、规模或分片条件。大尺寸是接口需考虑的使用规模。工作区计入新分配的解包、转换、归约、版本暂存及交接缓冲。调用方已有只读输入和已消费的输出不计入工作区，额外物化副本计入。

## 交换约定

`x`、`y` 给源格网轴边界，格点身份采用行、列。每块瓦片为矩形索引窗，`owners` 确定实际拥有者。字段 `amount` 是有效源格的完整格面积所含总量，`intensity` 是源格均匀强度。裁剪窗以及目标框按 x0、x1、y0、y1。区域元素左闭右开，坐标为整数微米，边界零面积接触不算覆盖。
`priority` 较大者优先；同优先级按较小稳定 ID 决胜。目标有效覆盖率的分母是目标在公共裁剪窗内、扣除同组优先占有部分后的面积。空区域不产生覆盖率。各组的有效目标量加上未被这组目标覆盖的有效源量，应等于原有效源量，裁剪窗外的量计入未覆盖。质量判断需覆盖率达到配置值，有效区域的重映射数值容差 1e-9。已经关闭的来源版本不得被修改。

## 重映射模型（`remapper` 包）

- `remapper.engine.Remapper`：持有源格网、裁剪窗、拥有者窗与目标声明，按数据集提供视图。
  - `deliver(dataset, source, rev, slice, tensors, prefix)` 合入局部瓦片；非拥有者格被忽略，同一格的后到修订覆盖先前值。
  - `update_priority(dataset, {目标: 优先级})` 更正开放数据集的归属优先级。
  - `close(dataset)` 关单并生成带 SHA-256 标识的不可变快照，`snapshot(dataset)` 可回查；未关单的 `view()` 标明预览身份。
  - `ingest(records, tensors)` 一次性回放完整交换事件流（含关单、修订、优先级更正）。
- 总量型（amount）：按有效源格与归属碎片的实际面积比例分配，互斥组内逐格扣减已占有区域后计算；`Reconciliation` 给出分配量、未覆盖份额与守恒残差。
- 强度型（intensity）：只用有效重叠面积做加权均值，同时输出有效覆盖率 `coverage_ratio` 与阈值判定 `covered`。
- 互斥组（group）之间归属互不影响，各自独立对账；同组目标重叠时按优先级、再按稳定 ID 归属。
- 三态输出（`ValueState`）显式区分：`no_coverage`（无覆盖）、`zero`（覆盖且值为零）、`pending`（等待有效资料，含已交付但失效的格）；可定量的非零值为 `measured`。
- 空间约束：不构造源格 × 目标配对矩阵，每格只与候选目标做矩形求交；数值保留在瓦片内，全网格只保留每格一字节的获胜索引和一字节的目标接触判定。`remapper.capacity.plan_capacity` 按 `capacity.json` 的规模估算峰值暂存并校验限额。

## 读取资料

使用 Python 3.11。首次创建本地环境时安装锁定依赖。只读查看显示原始字段和张量尺寸。

```sh
python -m pip install -r requirements.txt
python caseio.py data/capture.json
python -m remapper.cli data/capture.json --capacity data/capacity.json --out out/
python -m unittest discover -s tests -v
```

CLI 打印每个数据集的预览/快照身份、各目标两类张量与状态、覆盖率判定和分组对账；`--out` 另写可交接的 JSON。扩展到另一张网格与扫描瓦片时，复用同一 `Remapper` 接口传入新的轴边界、拥有者窗、目标框和瓦片张量即可。
