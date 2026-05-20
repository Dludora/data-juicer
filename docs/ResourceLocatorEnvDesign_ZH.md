# Resource Locator 与环境变量接入设计

## 1. 目标

这份文档只讨论两件事：

- `resource_policy_utils.py` 第一阶段应该怎么写
- 环境变量怎么按 DJ 现有实现风格暴露出去

这里不讨论 `config.py`，也不讨论第二阶段的统一大抽象。目标是先把第一阶段做成一个能落地的工程方案。

## 2. 设计约束

第一阶段要尽量贴近 DJ 现有实现，避免为了“抽象漂亮”把调用链一次性改得太大。

当前代码的几个现状：

- cache 目录已经通过环境变量暴露：
  - `DATA_JUICER_CACHE_HOME`
  - `DATA_JUICER_ASSETS_CACHE`
  - `DATA_JUICER_MODELS_CACHE`
  - `DATA_JUICER_EXTERNAL_MODELS_HOME`
- cache 变量是在 [cache_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/cache_utils.py) 中通过 `os.getenv(...)` 直接读取的
- 其他环境变量读取风格也比较直接：
  - `os.getenv(...)`
  - `os.environ.get(...)`
- 当前下载逻辑主要分散在：
  - [model_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/model_utils.py)
  - [asset_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/asset_utils.py)
  - [nltk_utils.py](/Users/dludora/Code/data-juicer/data_juicer/utils/nltk_utils.py)
  - [lazy_loader.py](/Users/dludora/Code/data-juicer/data_juicer/utils/lazy_loader.py)

因此第一阶段不建议：

- 先改 `config.py`
- 引入很复杂的配置对象体系
- 一口气把 repo / hf / package / asset / model 都统一成一个大资源模型

## 3. 文件布局建议

第一阶段新增一个管理文件：

- `data_juicer/utils/resource_policy_utils.py`

它的职责是：

- 统一读取资源治理相关环境变量
- 统一做默认值回退
- 给现有模块提供“当前该走哪个源、是否允许联网、是否允许 fallback”的判断

第一阶段不负责：

- 重写全部下载逻辑
- 接管 repo clone 的完整流程
- 实现第二阶段的 config 层

## 4. 环境变量设计

### 4.1 第一阶段新增变量

建议新增这些环境变量：

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

### 4.2 与 DJ 现有变量的关系

这些新变量不是替代现有 cache 变量，而是补“下载源和联网策略”。

现有变量继续保留：

- `DATA_JUICER_ASSETS_CACHE`
- `DATA_JUICER_MODELS_CACHE`
- `DATA_JUICER_EXTERNAL_MODELS_HOME`

职责划分：

- 现有变量：控制本地缓存目录、补充本地查找根
- 新变量：控制镜像地址、是否离线、是否允许公网 fallback、pip/HF/NLTK 行为

### 4.3 解析风格

建议完全沿用 DJ 现有环境变量读取风格：

- 使用 `os.getenv(...)`
- 或使用 `os.environ.get(...)`
- 在 `resource_policy_utils.py` 内集中解析

不要在第一阶段把解析逻辑散落到：

- `model_utils.py`
- `asset_utils.py`
- `nltk_utils.py`
- `lazy_loader.py`

### 4.4 解析规则

建议统一规则如下：

- 布尔值支持：
  - `1/0`
  - `true/false`
  - `yes/no`
  - `on/off`
- 多值列表统一使用 `os.pathsep` 分隔
- 空字符串视为未配置
- 未配置时回退当前默认行为

例如：

- `DJ_RESOURCE_LOCAL_CACHE_ROOTS=/mnt/dj-cache:/mnt/share/models`

## 5. `resource_policy_utils.py` 第一阶段接口

第一阶段建议只暴露下面这些接口：

- `get_resource_policy()`
- `resolve_model_source()`
- `resolve_asset_source()`
- `configure_hf_env()`
- `configure_nltk_env()`
- `should_allow_public_fallback()`
- `should_auto_install_package()`

不建议第一阶段一上来就提供：

- `resolve_repo(...)`
- `resolve_hf_model(...)`
- `ensure_package(...)`

因为这些会拉高接入成本，也会让第一阶段跟第二批 OP 散点治理绑死。

## 6. `resource_policy_utils.py` 建议实现

### 6.1 `get_resource_policy()`

职责：

- 读取所有 `DJ_*` 变量
- 完成类型转换
- 补默认值
- 返回一个标准化 dict

建议返回结构：

```python
{
  "offline_mode": False,
  "allow_public_fallback": True,
  "local_cache_roots": [],
  "model_base_url": None,
  "asset_base_url": None,
  "hf_endpoint": None,
  "hf_home": None,
  "hf_local_files_only": None,
  "nltk_data_dir": None,
  "nltk_allow_download": True,
  "package_auto_install": True,
}
```

说明：

- 第一阶段直接返回 `dict` 就够了
- 先不要为了这个额外建 dataclass
- 后续如果第二阶段要接 config，再考虑是否升级成统一配置对象

### 6.2 `resolve_model_source(model_name, force=False)`

职责：

- 决定 `model_utils.check_model()` 下一步应该去哪里找/下模型

建议查找顺序：

1. 如果 `model_name` 本身就是存在的本地路径，直接返回本地路径
2. 查 `DATA_JUICER_MODELS_CACHE`
3. 查 `DATA_JUICER_EXTERNAL_MODELS_HOME`
4. 查 `DJ_RESOURCE_LOCAL_CACHE_ROOTS`
5. 如果配置了 `DJ_MODEL_BASE_URL`，拼镜像 URL
6. 回退 `MODEL_LINKS`
7. 最后回退 `BACKUP_MODEL_LINKS`

建议返回结构：

```python
{
  "kind": "local_path" | "remote_url",
  "value": "...",
  "source": "explicit_path" | "cache" | "external_root" | "local_cache_root" | "mirror" | "default_public",
}
```

这样可以最大程度复用当前 `check_model()` 的下载逻辑，不需要第一阶段重写下载器。

### 6.3 `resolve_asset_source(asset_type)`

职责：

- 决定 `asset_utils.load_words_asset()` 应该去哪里找词表

建议查找顺序：

1. 查 `DATA_JUICER_ASSETS_CACHE`
2. 查 `DJ_RESOURCE_LOCAL_CACHE_ROOTS`
3. 如果配置了 `DJ_ASSET_BASE_URL`，按约定拼 URL
4. 回退 `ASSET_LINKS`

第一阶段拼接规则建议固定为：

- `flagged_words` -> `{base_url.rstrip('/')}/flagged_words.json`
- `stopwords` -> `{base_url.rstrip('/')}/stopwords.json`

### 6.4 `configure_hf_env()`

职责：

- 在调用 `from_pretrained(...)` 之前，统一设置 HF 相关环境

建议处理：

- 若 `DJ_HF_ENDPOINT` 有值，设置 `HF_ENDPOINT`
- 若 `DJ_HF_HOME` 有值，设置 `HF_HOME`
- 若 `DJ_HF_LOCAL_FILES_ONLY=true`，设置：
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
- 若 `DJ_RESOURCE_OFFLINE_MODE=true` 且用户没有显式关闭本地限定，也可以考虑同样设置离线标记

第一阶段先不改 `from_pretrained(...)` 的调用签名，先通过环境变量控制行为。

### 6.5 `configure_nltk_env()`

职责：

- 在 `nltk_utils.py` 和 `model_utils.py` 的 NLTK 逻辑前，统一配置 data dir 和下载开关

建议处理：

- 若 `DJ_NLTK_DATA_DIR` 有值，插入 `nltk.data.path`
- 将 `offline_mode` 和 `DJ_NLTK_ALLOW_DOWNLOAD` 的组合判断统一收口

建议额外提供一个内部判断函数，例如：

- `is_nltk_download_allowed()`

规则：

- `offline_mode=true` -> 不允许
- `DJ_NLTK_ALLOW_DOWNLOAD=false` -> 不允许
- 其他情况 -> 允许

### 6.6 `should_allow_public_fallback()`

职责：

- 统一判断是否允许从镜像/本地失败后回退当前默认公网源

规则：

- `offline_mode=true` 时，必须返回 `False`
- 否则由 `DJ_RESOURCE_ALLOW_PUBLIC_FALLBACK` 决定

### 6.7 `should_auto_install_package()`

职责：

- 统一判断 `LazyLoader.check_packages(...)` 是否还允许自动安装

规则：

- 直接受 `DJ_PACKAGE_AUTO_INSTALL` 控制
- 如果 `offline_mode=true`，也建议强制关闭自动安装

## 7. 如何接入 DJ 现有实现

### 7.1 接入 `model_utils.py`

要改的点：

- `check_model()`
  - 先调用 `resolve_model_source()`
  - 若返回 `local_path`，直接返回
  - 若返回 `remote_url`，继续沿用当前 `wget.download(...)`
- 所有 HF `from_pretrained(...)` 前统一调用 `configure_hf_env()`
- 所有 NLTK 相关准备逻辑前统一调用 `configure_nltk_env()`

这样能最大程度复用 DJ 当前实现。

### 7.2 接入 `asset_utils.py`

要改的点：

- `load_words_asset()`
  - 本地找不到时，不再直接使用 `ASSET_LINKS[...]`
  - 改成先问 `resolve_asset_source()`
  - 若结果是远端 URL，再继续沿用当前 `requests.get(...)`

### 7.3 接入 `nltk_utils.py`

要改的点：

- `ensure_nltk_resource()` 一进入就先调用 `configure_nltk_env()`
- 在所有 `nltk.download(...)` 之前统一判断是否允许下载
- 若不允许下载，则直接返回失败或抛出清晰错误，不再继续联网尝试

### 7.4 接入 `lazy_loader.py`

要改的点：

- `LazyLoader.check_packages(...)` 安装前先判断 `should_auto_install_package()`

第一阶段不需要重写 `LazyLoader` 的整体安装实现。

## 8. 第一阶段实现顺序

建议顺序：

1. 新增 `resource_policy_utils.py`
2. 在其中补环境变量解析和策略判断
3. 接入 `model_utils.py`
4. 接入 `asset_utils.py`
5. 接入 `nltk_utils.py`
6. 接入 `lazy_loader.py`

这样做的好处是：

- 先把控制面立住
- 再逐个接入集中入口
- 不需要一开始就处理所有 OP 内散点逻辑

## 9. 第一阶段验收口径

至少验证这些行为：

1. 不设置任何 `DJ_*` 变量时，行为不变
2. 设置 `DJ_MODEL_BASE_URL` 后，模型优先走镜像
3. 设置 `DJ_ASSET_BASE_URL` 后，词表优先走镜像
4. `DJ_RESOURCE_OFFLINE_MODE=true` 时，不允许公网 fallback
5. `DJ_NLTK_ALLOW_DOWNLOAD=false` 时，不允许在线下载 NLTK
6. `DJ_PACKAGE_AUTO_INSTALL=false` 时，缺包直接失败
7. `DJ_HF_ENDPOINT` / `DJ_HF_HOME` / `DJ_HF_LOCAL_FILES_ONLY` 能影响 HF 加载行为
