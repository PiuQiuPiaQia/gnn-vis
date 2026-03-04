# DLMSF Patch 有限差分方案（方案一）

状态：当前目录仅保留设计文档，不放实现代码。

## 目标

新增一条物理分析线路：
基于 PV/steering 前向算子，通过 patch 扰动得到二维敏感度图 `S_map`，再复用现有 IG 对齐指标（Spearman、Top-K IoU）完成对比。

## 方法总览

1. 基线前向计算（`J_phys`）
   - 输入：`eval_inputs[time_idx=0]` 的三维场，至少包含 `u`、`v`，最好再有 `T/theta` 与 `Z/phi`。
   - 区域：先沿用 SWE 子域（台风中心 `±20°`）。
   - 预处理：可选环境分离（Shapiro 去涡旋或风场环境分离）。
   - 平衡风 `V_bal` 计算：
     - MVP：地转近似（由高度梯度估算）。
     - 升级：接入外部 QG/PV inversion 黑盒。
   - 用 `V_bal` 计算深层引导气流（DLMSF，示例层结为 `925–300 hPa`）。
   - 定义标量目标：`J_phys = DLMSF · d_hat`。

2. Patch 图表示（节点/超节点）
   - 将 ROI 划分为空间 patch（例如 `2°×2°`）。
   - 可选构造机制解释超节点（`BH/CH/四象限`）。

3. 有限差分扰动
   - 每次运行只选一种扰动族（不混用）：
     - 机制优先：扰动环境 `q_prime`。
     - 通道优先：扰动 patch 内 `u/v`。
   - 对每个 patch `P` 施加：`x_pert = x + eps * mask_P`。
   - 用同一前向算子重算 `J_phys_pert`。
   - 得到 patch 分数：`S_P = |J_phys_pert - J_phys| / eps`。

4. 敏感度图回填
   - 将 `S_P` 回填到 patch 覆盖像素，形成二维 `S_map`。
   - 仅在可视化场景下可选轻微平滑。

5. 与现有流程对齐
   - 将 `S_map` 接到当前 SWE 使用的对齐阶段（Spearman、Top-K IoU、重叠图与曲线图）。
   - 与 SWE 并行运行；两条线路分别和 IG 对齐比较。

## 预期输出

- `S_map`：用于对齐评估的二维物理敏感度图。
- `A_k`（可选）：系统块贡献汇总表（`BH/CH/四象限`）。
- 复现实验参数：
  - `eps`、patch 尺寸、ROI 半径
  - DLMSF 层范围与半径定义
  - `d_hat` 定义方式
  - 扰动变量族（`q_prime` 或 `u/v`）

## MVP 范围约束

- 首版不实现严格的真 PV inversion。
- 除非集成必须，MVP 不新增 CLI 入口。
- 优先复用现有指标、落盘和绘图链路。
