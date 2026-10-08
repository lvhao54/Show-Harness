# VLABench 插花资产

本目录为 OpenArm 的 `--scene insert_flower` 保存花瓶 `vase_0` 和玫瑰 `rose_0` 的真实网格、贴图与碰撞模型，共 183 个原始文件（25,342,439 字节）。

来源均固定到具体版本：

- [VLABench 插花任务源码](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/VLABench/tasks/hierarchical_tasks/primitive/insert_flower_series.py)，代码版本 `cf588fe60c0c7282174fe979f5913170cfe69017`。
- [花瓶资产](https://huggingface.co/datasets/lerobot/vlabench-assets/tree/08d7a4479a55c8fbb89ac0fd6fcf06e53d65b4a4/obj/meshes/containers/vases/vase_0)与[玫瑰资产](https://huggingface.co/datasets/lerobot/vlabench-assets/tree/08d7a4479a55c8fbb89ac0fd6fcf06e53d65b4a4/obj/meshes/flowers/bloom_flower/rose/rose_0)，来自 `lerobot/vlabench-assets` 镜像版本 `08d7a4479a55c8fbb89ac0fd6fcf06e53d65b4a4`。

代码与资产镜像分别记录版本，不表示二者存在已确认的版本对应关系。本项目采用插花任务概念与对象资产，使用现有 OpenArm 桌面、机械臂和相机；未移植完整 VLABench benchmark、房间、Franka 机器人或专家策略，也未新增 `dm_control` 依赖。

原始 `vase_0/vase.xml`、`rose_0/rose_0.xml` 及其引用的资产按字节保留。各目录下的 `openarm.xml` 是派生版本：花瓶的 `pos`、`size`、mesh `scale` 乘以 0.65，玫瑰乘以 0.8；碰撞模型与辅助 site 设为透明，碰撞仍启用。`../openarm2_insert_flower.xml` 将花瓶固定，并创建 `flower_1`、`flower_2`、`flower_3` 三个自由根 body；每个根 body 刚性连接花头与花杆。

[ASSET_MANIFEST.json](ASSET_MANIFEST.json) 的 `files[].path` 相对此目录，可逐项按 `bytes` 和 SHA256 校验原始文件；将 `download_base_url` 与 `source_path` 拼接可取得固定版本的来源文件。派生文件及改造记录在 `derived_files` 中，不使用原始文件的哈希。完整来源记录见 [SOURCE.json](SOURCE.json)。

VLABench **代码**采用 MIT，许可证保留于 [VLABench-CODE-LICENSE.txt](VLABench-CODE-LICENSE.txt)。记录版本的资产镜像未提供单独的资产许可证；该代码许可证不能单独证明网格和贴图的授权范围。
