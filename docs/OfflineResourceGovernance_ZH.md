# DJ 离线资源治理设计文档

## 1. 当前情况

### 1.1 联网行为主要发生在什么时候

这些资源大多数不是在 `uv pip install -e .` 时下载，而是在 DJ 运行 recipe 时下载。当前代码里比较明确的联网入口包括：

- [model_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/model_utils.py)
  - `check_model()`：通过 `wget.download(...)` 下载 `MODEL_LINKS` / `BACKUP_MODEL_LINKS`
  - 多个 `prepare_*`：通过 `AutoProcessor.from_pretrained(...)`、`AutoConfig.from_pretrained(...)`、`GenerationConfig.from_pretrained(...)`、`model_class.from_pretrained(...)` 触发 HuggingFace 联网
  - 部分模型准备逻辑：通过 `subprocess.run(["wget", ...])` 下载文件
  - `prepare_nltk_model()`、`prepare_nltk_pos_tagger()`：内部直接 `nltk.download(...)`
  - 多处 `LazyLoader.check_packages(...)`：运行时缺包会触发安装
- [asset_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/asset_utils.py)
  - `load_words_asset()`：通过 `requests.get(ASSET_LINKS[...])` 下载词表
- [nltk_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/nltk_utils.py)
  - `ensure_nltk_resource()`：资源缺失时调用 `nltk.download(...)`
- [lazy_loader.py](/Users/dludora/Code/data-juicer/data_juicer/utils/lazy_loader.py)
  - `LazyLoader.check_packages(...)`：缺包时自动安装 Python 依赖
- OP 内部散点入口
  - [phrase_grounding_recall_filter.py](/Users/dludora/Code/data-juicer/data_juicer/ops/filter/phrase_grounding_recall_filter.py)：直接 `nltk.download(...)`
  - 多个多模态 OP：通过 `LazyLoader.check_packages(...)` 触发 pip / git 依赖安装
  - 部分 OP / 相关准备逻辑还会直接拉取 repo 或模型文件，后续需要单独收口
- 其他工具入口
  - [analysis/collector.py](/Users/dludora/Code/data-juicer/data_juicer/analysis/collector.py)：`AutoTokenizer.from_pretrained(...)`
  - [qc_utils.py](/Users/dludora/Code/data-juicer/data_juicer/tools/quality_classifier/qc_utils.py)：`wget.download(...)`

### 1.2 现在的问题是什么

当前的核心问题有 4 个：

1. 下载入口分散

- `model_utils.py`
- `asset_utils.py`
- `nltk_utils.py`
- `lazy_loader.py`
- 多个 OP 内部直接下载

2. 没有统一控制面

- 目前只有 `DATA_JUICER_ASSETS_CACHE`
- `DATA_JUICER_MODELS_CACHE`
- `DATA_JUICER_EXTERNAL_MODELS_HOME`
- 这些只能控制缓存位置，不能控制下载源和联网策略

3. 默认来源不统一

- 一部分资源走 DJ 自己的 OSS：`MODEL_LINKS`、`ASSET_LINKS`
- 一部分直接走第三方：HuggingFace、GitHub、NLTK、pip

4. 有些联网逻辑写死在 OP 内部

- 后面很难统一切到内网镜像
- 也很难让专有云统一治理

## 3. 现有入口概览

为了便于实现，这里只保留对改造有用的入口分组：

### 3.1 文件型资源

主要包括：

- 模型文件
- 静态词表

主要入口：

- [model_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/model_utils.py)
- [asset_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/asset_utils.py)

当前默认来源：

- `MODEL_LINKS`
- `BACKUP_MODEL_LINKS`
- `ASSET_LINKS`

### 3.2 仓式资源

主要包括：

- HuggingFace `from_pretrained(...)`
- OP 内部 `git clone` 的 repo

相关入口：

- [model_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/model_utils.py)
- 若干 mapper / filter OP

### 3.3 数据包与运行时依赖

主要包括：

- NLTK 数据包
- 运行时自动安装的 Python 包

相关入口：

- [nltk_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/nltk_utils.py)
- [lazy_loader.py](/Users/dludora/Code/data-juicer/data_juicer/utils/lazy_loader.py)

## 4. 设计原则

这次改造按下面几个原则收敛：

1. 默认行为不变

- 用户不配新入口时，继续走当前逻辑

2. 先收控制权，再收实现

- 第一阶段先统一“是否允许联网、从哪里找资源”
- 不要求一开始就重写所有下载逻辑

3. 先环境变量，后主配置

- 专有云、容器、K8s 更适合环境变量注入
- `config.py` 放到第二阶段

4. 先治理集中入口，再治理 OP 内散点逻辑

- 先处理 `model_utils.py`、`asset_utils.py`、`nltk_utils.py`、`lazy_loader.py`
- 再逐步处理 OP 内 `git clone` / `wget` / `nltk.download(...)`

## 5. 实现方案

### 5.1 先增加一个统一管理文件

第一步先新增一个统一管理模块：

- `data_juicer/utils/resource_policy_utils.py`

第一阶段它只负责两件事：

- 统一读取资源治理相关配置
- 统一决定资源来源和联网策略

第一阶段不要让各模块自己解析环境变量。

### 5.2 第一阶段先暴露环境变量

第一阶段只提供环境变量入口，不引入新的 YAML 配置。

建议支持的变量：

- `DJ_RESOURCE_OFFLINE_MODE`
- `DJ_RESOURCE_ALLOW_PUBLIC_FALLBACK`
- `DJ_RESOURCE_LOCAL_CACHE_ROOTS`
- `DJ_MODEL_BASE_URL`
- `DJ_ASSET_BASE_URL`
- `DJ_HF_ENDPOINT`
- `DJ_HF_HOME`
- `DJ_HF_LOCAL_FILES_ONLY`
- `DJ_NLTK_DATA_DIR`
- `DJ_NLTK_ALLOW_DOWNLOAD`
- `DJ_PACKAGE_AUTO_INSTALL`

第一阶段优先级固定为：

- 显式参数 > Env > Default

这里的 Default 指当前实现本身：

- `MODEL_LINKS`
- `BACKUP_MODEL_LINKS`
- `ASSET_LINKS`
- HuggingFace 默认行为
- NLTK 默认行为
- pip 默认行为

### 5.3 第二阶段再暴露主配置

第二阶段再考虑在主配置中增加一个简单配置块，例如：

```yaml
resource:
  offline_mode: false
  allow_public_fallback: true
  local_cache_roots: []
  model_base_url: null
  asset_base_url: null
  hf_endpoint: null
  hf_home: null
  hf_local_files_only: null
  nltk_data_dir: null
  nltk_allow_download: true
  package_auto_install: true
```

这里先保持扁平，不做太深的嵌套。

第二阶段优先级再扩成：

- 显式参数 > Config > Env > Default

### 5.4 统一管理文件第一阶段职责

`resource_policy_utils.py` 第一阶段建议只做这些事：

- 读取环境变量并生成标准化 policy
- 解析模型文件来源
- 解析静态词表来源
- 配置 HF 运行环境
- 配置 NLTK 运行环境
- 统一判断是否允许公网 fallback
- 统一判断是否允许自动安装包

建议接口先收敛为：

- `get_resource_policy()`
- `resolve_model_source()`
- `resolve_asset_source()`
- `configure_hf_env()`
- `configure_nltk_env()`
- `should_allow_public_fallback()`
- `should_auto_install_package()`

第一阶段不追求大而全，不要求一开始就把 repo / HF / package 都抽象成统一对象。

### 5.5 第一阶段来源解析规则

`model` 建议查找顺序：

1. 显式本地路径
2. `DATA_JUICER_MODELS_CACHE`
3. `DATA_JUICER_EXTERNAL_MODELS_HOME`
4. `DJ_RESOURCE_LOCAL_CACHE_ROOTS`
5. `DJ_MODEL_BASE_URL`
6. `MODEL_LINKS`
7. `BACKUP_MODEL_LINKS`

`asset` 建议查找顺序：

1. `DATA_JUICER_ASSETS_CACHE`
2. `DJ_RESOURCE_LOCAL_CACHE_ROOTS`
3. `DJ_ASSET_BASE_URL`
4. `ASSET_LINKS`

对于 HF / NLTK / pip，第一阶段不改原始调用方式，只改环境和开关控制。

### 5.6 第一阶段错误和日志要求

至少要统一三类信息：

- 当前资源是什么
- 试过哪些来源
- 当前策略为什么允许或禁止继续 fallback

最小要求：

- `offline_mode=true` 时，报错要明确说明是策略禁止公网访问
- 镜像没命中时，要能看出是否继续回退到了默认公网源
- 自动装包被禁止时，要直接报缺失包名

## 6. 分阶段 Todo

### Todo 1. 收一个统一管理文件

新增：

- `data_juicer/utils/resource_policy_utils.py`

负责：

- 统一读取资源治理相关设置
- 不让各业务模块自己解析环境变量
- 提供来源解析和运行环境注入能力

交付物：

- `resource_policy_utils.py`
- 基础 policy 读取与解析测试

### Todo 2. 暴露第一阶段环境变量入口

先只做环境变量，不改 `config.py`。

要做的事：

- 增加 `DJ_*` 变量解析
- 统一布尔值和列表值解析
- 统一默认值回退逻辑
- 固定第一阶段优先级：显式参数 > Env > Default

第一阶段重点变量：

- `DJ_RESOURCE_OFFLINE_MODE`
- `DJ_RESOURCE_ALLOW_PUBLIC_FALLBACK`
- `DJ_RESOURCE_LOCAL_CACHE_ROOTS`
- `DJ_MODEL_BASE_URL`
- `DJ_ASSET_BASE_URL`
- `DJ_HF_ENDPOINT`
- `DJ_HF_HOME`
- `DJ_HF_LOCAL_FILES_ONLY`
- `DJ_NLTK_DATA_DIR`
- `DJ_NLTK_ALLOW_DOWNLOAD`
- `DJ_PACKAGE_AUTO_INSTALL`

交付物：

- 环境变量说明
- 解析逻辑
- 解析单测

### Todo 3. 接入第一批集中入口

第一批先接 4 个文件：

- [model_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/model_utils.py)
- [asset_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/asset_utils.py)
- [nltk_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/nltk_utils.py)
- [lazy_loader.py](/Users/dludora/Code/data-juicer/data_juicer/utils/lazy_loader.py)

具体改动：

- `model_utils.py`
  - `check_model()` 接 `resolve_model_source()`
  - HF 加载前统一调用 `configure_hf_env()`
  - NLTK 相关准备逻辑前统一调用 `configure_nltk_env()`
- `asset_utils.py`
  - `load_words_asset()` 接 `resolve_asset_source()`
- `nltk_utils.py`
  - `ensure_nltk_resource()` 受 `DJ_NLTK_ALLOW_DOWNLOAD` 和 `offline_mode` 控制
- `lazy_loader.py`
  - 自动安装受 `DJ_PACKAGE_AUTO_INSTALL` 控制

交付物：

- 第一批接入代码
- 对应行为测试

### Todo 4. 处理第二批散点入口

第二批再处理写死在 OP 里的联网逻辑。

优先文件：

- [phrase_grounding_recall_filter.py](/Users/dludora/Code/data-juicer/data_juicer/ops/filter/phrase_grounding_recall_filter.py)
- [vggt_mapper.py](/Users/dludora/Code/data-juicer/data_juicer/ops/mapper/vggt_mapper.py)
- [video_hand_reconstruction_mapper.py](/Users/dludora/Code/data-juicer/data_juicer/ops/mapper/video_hand_reconstruction_mapper.py)
- [video_hand_reconstruction_hawor_mapper.py](/Users/dludora/Code/data-juicer/data_juicer/ops/mapper/video_hand_reconstruction_hawor_mapper.py)
- [video_camera_pose_mapper.py](/Users/dludora/Code/data-juicer/data_juicer/ops/mapper/video_camera_pose_mapper.py)
- [video_depth_estimation_mapper.py](/Users/dludora/Code/data-juicer/data_juicer/ops/mapper/video_depth_estimation_mapper.py)
- [image_sam_3d_body_mapper.py](/Users/dludora/Code/data-juicer/data_juicer/ops/mapper/image_sam_3d_body_mapper.py)
- [pii_llm_suspect_mapper.py](/Users/dludora/Code/data-juicer/data_juicer/ops/mapper/pii_llm_suspect_mapper.py)

目标：

- OP 不再自己决定是否联网
- OP 不再自己决定走哪个源
- 逐步改成通过统一管理模块拿资源策略

### Todo 5. 暴露第二阶段主配置入口

等第一阶段稳定后，再在主配置中增加 `resource` 配置块。

这里重点不是新增能力，而是把第一阶段的环境变量能力再暴露到配置层。

要求：

- 配置项保持简单、扁平
- 不引入复杂层级
- 明确优先级：显式参数 > Config > Env > Default

### Todo 6. 盘点与验证

补一个只读工具或脚本，用于盘点 recipe 可能触发的资源下载。

目标：

- 帮助专有云提前准备资源
- 帮助排查哪些下载还没被统一入口接住

## 7. 验收口径

第一阶段至少要满足这些行为：

1. 不配置任何新入口时，行为不变
2. 配置 `DJ_MODEL_BASE_URL` 后，模型优先走镜像
3. 配置 `DJ_ASSET_BASE_URL` 后，词表优先走镜像
4. `DJ_RESOURCE_OFFLINE_MODE=true` 时，不允许公网 fallback
5. `DJ_NLTK_ALLOW_DOWNLOAD=false` 时，不允许在线下载 NLTK 数据
6. `DJ_PACKAGE_AUTO_INSTALL=false` 时，缺包直接失败
7. HF 相关加载能受 `DJ_HF_ENDPOINT` / `DJ_HF_HOME` / `DJ_HF_LOCAL_FILES_ONLY` 控制

## 8. 备注

从当前代码能确认的是：

- DJ 官方确实维护了自己的 OSS 入口：`MODEL_LINKS`、`ASSET_LINKS`
- 走这两个入口的资源，默认会优先从 DJ OSS 下载
- 但 DJ 并不是所有资源都先走 DJ OSS，仍有大量第三方来源

如果后面要回答“DJ 官方 OSS 总规模有多大、是否可以整体迁移”，还需要结合实际 OSS 清单和 license 审核，不是仅靠代码仓就能给出完整结论。
