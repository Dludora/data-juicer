# smiles_augmentation_mapper

Generates randomized but equivalent SMILES strings for molecular augmentation.

使用随机但等价的 SMILES 写法对分子样本进行增强。

This operator parses the input SMILES with RDKit and generates alternative SMILES
serializations by randomizing the traversal order on the same molecular graph. It can
either expand one sample into multiple augmented samples or write the generated SMILES
into another field while preserving the original one.

该算子使用 RDKit 解析输入的 SMILES，并通过随机化同一分子图的遍历顺序生成不同但等价的
SMILES 表达。它既可以把一条样本扩增成多条样本，也可以在保留原始字段的同时把增强结果写入另一列。

Type 算子类型: **mapper**

Tags 标签: text, cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `aug_num` | `PositiveInt` | `1` | Number of augmented SMILES strings to generate for each input sample. |
| `keep_original_sample` | `bool` | `True` | Whether to keep the original sample in the output dataset. |
| `smiles_key` | `str` | `'SMILES'` | Input field name that stores the source SMILES string. |
| `output_key` | `Optional[str]` | `None` | Output field name used to store the randomized SMILES string. If `None`, overwrite the source field. |
| `isomeric_smiles` | `bool` | `True` | Whether to preserve stereochemistry and isotopic information when generating randomized SMILES. |
| `max_try` | `Optional[PositiveInt]` | `None` | Maximum number of randomization attempts for each sample. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/smiles_augmentation_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_smiles_augmentation_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
