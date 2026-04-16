# smiles_captioning_mapper

Generates molecule descriptions from SMILES strings with an API model.

使用 API 模型根据 SMILES 生成分子文本描述。

This operator is designed for chemistry datasets that contain fields such as `SMILES` and
`description`. It uses Data-Juicer's OpenAI-compatible API wrapper to turn a SMILES string
into a concise natural-language caption, and supports custom prompts, retry logic, and
custom output field names.

该算子适用于包含 `SMILES`、`description` 等字段的化学数据集。它复用了 Data-Juicer
统一的 OpenAI-compatible API 封装，将 SMILES 转成简洁的自然语言描述，并支持自定义
prompt、重试次数和输出字段名。

Type 算子类型: **mapper**

Tags 标签: text, cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | `str` | `'gpt-4o'` | API model name. |
| `smiles_key` | `str` | `'SMILES'` | Input field name that stores the source SMILES string. |
| `caption_key` | `str` | `'description'` | Output field name used to store the generated molecular description. |
| `api_endpoint` | `Optional[str]` | `None` | URL endpoint for the API. |
| `response_path` | `Optional[str]` | `None` | Path to extract content from the API response. |
| `system_prompt` | `Optional[str]` | `None` | System prompt for the captioning task. |
| `input_template` | `Optional[str]` | `None` | Template for building the user input. It must contain the placeholder `{smiles}`. |
| `overwrite` | `bool` | `False` | Whether to overwrite an existing non-empty caption. |
| `try_num` | `PositiveInt` | `3` | Number of retries when the API call fails or returns an empty caption. |
| `model_params` | `Optional[Dict]` | `None` | Parameters for initializing the API model. |
| `sampling_params` | `Optional[Dict]` | `None` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/smiles_captioning_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_smiles_captioning_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
