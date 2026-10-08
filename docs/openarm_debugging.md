# OpenArm 实验排查与修复（2026-10-08）

当前分支的失败包含接入缺陷和策略迁移问题，不能统一归为模型能力不足。
修复后，真实 gateway 零样本运行在 39 步完成红块叠绿块，独立物理评分为真。
这是一次成功样本，不是成功率估计；固定动作回归与 VLM 成绩分开记录。

## 失败证据与对应修复

1. **默认下降下限使真实抓取不可达。** 原腕部 floor 为 -0.225 m，home 为
   -0.220 m，只允许下降 5 mm。指垫与桌上 45 mm 方块仅约 1 mm 高度重叠，
   对准后闭合仍抓空。相同原子动作改用 -0.250 m 可以夹住并提升超过 50 mm。
   两份配置统一使用 -0.250 m；真实 MuJoCo 接触/抓取测试取代旧的假定几何偏移测试。
2. **夹爪状态阈值不适配 OpenArm。** 抓住方块时开口约 45 mm，旧阈值 20 mm
   会向策略报告 OPEN。阈值改为 60 mm；完全打开约 142.5 mm，空抓阈值仍为 6 mm。
3. **相机与提示不匹配。** 原生腕部相机接近水平侧视，图像上下主要对应高度。
   直接套用俯视腕相机的深度规则导致走反、上下振荡及拒绝抓取。保留用户已有的
   原生图像与 FWD/BACK 执行交换，增加 OpenArm 水平侧抓提示，明确腕部与指垫的
   区别、抬升导致的透视变化，并关闭三步开环动作。高度提示改为“允许继续下降多少”，
   不再把腕部安全下限说成桌面。
4. **向中央搬运时肘部撞机身。** 原 IK 未约束 link4 与 pedestal 的碰撞。
   物理接触使腕部高度和姿态偏离目标，即使 IK 未报错。加入机器人自身避碰约束，
   orientation cost 从 1.5 提到 12，nullspace cost 从 8.5 降到 2，每 tick 求解
   10 次。固定动作可保持抓取并完成中央堆叠。扩展点集中在 backend/kinematics.py，
   外部依赖锁定实测的 openarm-control 0.4.0。
5. **未完成阶段被超时跳过。** 历史 20:28 零样本运行在第 40 步尚未执行任何
   GRASP，就被推进到 LIFT，第 80 步再进入 MOVE。runner 现在遇到阶段超时立即失败，
   不再授权后续依赖动作。
6. **缺少真正的堆叠评分。** 新增 red_on_green：红绿块几何关系、红/绿与绿/桌
   支撑接触、低速度、无机械臂接触必须同时满足。最终 DONE 仍须通过该检查。
   物体坐标和评分不进入模型 observation。未启用检查时保留硬件的原有完成语义。

旧微调模型实验另有服务端 EOS 错误：`invalid_eos_runs` 不应作为有效策略评测。
修正 EOS 后的四个本地 adapter 实验仍全部输出 MV_DOWN；该 adapter 未训练
OpenArm，修复接入并不意味着它已获得跨机器人迁移能力。此次成功验证使用零样本
VLM，没有把固定动作脚本或 simulator 真值当作策略输入。

## 验证与复现

无需模型的回归：

```bash
.venv/bin/python -m pytest tests/test_openarm_grasp.py tests/test_openarm_success.py -q
```

它覆盖原 floor 的空抓负例、修复后的真实抓取/抬升、无肘部碰撞的搬运/释放/撤离、
稳定堆叠正例，以及悬空、错位、移动、仍被夹住等评分负例。

已完成的自主运行：

- 路径：`rollouts/openarm_fix_verification/fixed_ik/Gateway-CoT/1008/task_0/21-23-40/`
- 模型：`gpt-6.1-sol`，`reasoning_effort: medium`。
- 任务：`Pick up the red cube and place it on the green cube`。
- `summary.json`：`steps: 39`、`success: true`、`environment_success: true`。
- 视频：该目录下 `rollout_success.mp4`。
- 首次中间修复运行完成抓取/抬升，但中央搬运失败；最终加入避碰和透视提示后完成任务。
- 一个最终复跑的规划请求遭遇网关 HTTP 499，随后正常重试成功；不计为机器人策略样本。

复现成功样本的配置已保存在 `rollouts/openarm_fix_verification/fixed_ik.yaml`：

```bash
.venv/bin/python scripts/run_real.py \
  --robot-config rollouts/openarm_fix_verification/fixed_ik.yaml --no-show
```

该配置沿用相同物理模型、动作、相机和插件，只把 gateway reasoning 调成 medium、
总步数设为 65，并取消 smooth 的墙钟等待；模拟物理 tick 数不变。
公共配置可直接这样运行（凭据从本地 secrets/env 读取）：

```bash
.venv/bin/python scripts/run_real.py \
  --robot-config configs/robot_openarm_zero.yaml --vlm-backend gateway --no-show
```

更换任务时同步修改 success_check；`red_on_green` 只评默认堆叠任务。改变工具姿态
或相机安装后，需重新校准 floor 与视觉约定。微调路径需另外采集 OpenArm 轨迹训练。
