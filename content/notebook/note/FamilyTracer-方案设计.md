# 家族模型追踪方案：基于连续相似度度量的模型血缘检测

## 背景

随着开源大语言模型（LLM）的普及和微调技术的 democratization，模型盗窃和未授权修改问题日益严重。现有的模型指纹方法多为二分类（判断是否为特定模型），缺乏对模型"家族血缘"关系的连续度量能力。

**核心问题**：如何量化一个 suspect model 与原始模型（ancestor）之间的"血缘关系"程度？

---

## 核心创新：从论文方法迁移

借鉴论文《Measuring Human Involvement in AI-Generated Text》的核心思想：

| 论文方法 | 家族模型追踪对应 |
|---------|----------------|
| **人类参与度** | **模型血缘度**（Ancestry Degree） |
| **Prompt vs Generated Text** | **Ancestor Model vs Suspect Model** |
| **BERTScore相似度** | **模型行为相似度** |
| **连续数据集CAS-CS** | **连续血缘数据集** |
| **双头RoBERTa** | **双头模型血缘检测器** |

---

## 方案设计

### Step 1: 探针数据集注入（Probe Injection）

类比论文中的 Prompt，我们设计**探针数据集**来标记模型行为：

**探针数据集构建**：
```
Probe Set P = {p₁, p₂, ..., pₙ}
- 覆盖不同能力维度（推理、知识、生成、代码）
- 包含边界案例和对抗性样本
- 随机采样生成不同"血缘度"的训练数据
```

**血缘度定义**（类比Human Involvement）：
- **100%血缘**：原始模型（直接继承）
- **0%血缘**：无关模型（完全独立）
- **中间值**：经过不同程度微调/修改的模型

---

### Step 2: 行为相似度度量（Behavioral Similarity Score）

类比BERTScore，设计**ModelBehaviorScore**：

#### 2.1 行为向量提取

对于每个探针样本，从模型中提取**行为特征向量**：

```python
# 对于探针样本 x，从模型 M 中提取行为向量
behavior_vector(M, x) = [
    - logits分布 (p(xᵢ|x<ᵢ))
    - 隐藏层表示 (hidden states)
    - 注意力模式 (attention patterns)
    - 生成困惑度 (perplexity)
    - 语义嵌入 (semantic embeddings)
]
```

#### 2.2 ModelBehaviorScore 计算

使用对比学习框架计算两个模型行为的相似度：

```
Reference: Ancestor Model (A)
Candidate: Suspect Model (S)

Precision (利用率): 探针在S中的行为有多少与A匹配
Recall (血缘度): 探针在A中的行为有多少保留在S中  
F1 (整体相似度): 平衡指标
```

**归一化公式**（类比论文公式1）：
$$
y_{ancestry} = \frac{\text{ModelBehaviorScore}(A, S) - \min}{\max - \min}
$$

---

### Step 3: 连续血缘数据集构建（Family Tree Dataset）

类比论文的CAS-CS数据集，构建**FT-LLM**（Family Tree LLM）数据集：

#### 3.1 数据集生成策略

```
原始模型 Ancestor (LLaMA-2-7B)
    ↓ 100%血缘
    ├── 直接复制模型
    ↓ 80%血缘  
    ├── 轻微微调（1 epoch, 小学习率）
    ↓ 50%血缘
    ├── 中度微调（5 epoch, 领域数据）
    ↓ 20%血缘
    ├── 深度微调（20 epoch, 大学习率）
    ↓ 0%血缘
    └── 无关模型（Mistral, Falcon等）
```

#### 3.2 血缘度标注方法

- **100%**：完全相同的模型参数
- **80%**：轻微微调（<100 steps）
- **50%**：中度微调（LoRA适配）
- **20%**：深度微调（全参数）
- **0%**：不同架构/预训练的模型

---

### Step 4: 双头血缘检测器（Dual-Head Ancestry Detector）

类比论文的双头RoBERTa模型，设计：

#### 4.1 架构设计

```
                    Suspect Model Behavior
                           ↓
              Shared Transformer Encoder
                 (RoBERTa/BERT-based)
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   Regression Head   Token/Probe     Architecture
   (血缘度估计)      Classification   Verification
        ↓             (探针分类)          Head
   Ancestry Score        ↓                  ↓
   (0-1连续值)    哪些探针被"记住"    架构相似度
```

#### 4.2 回归头（Regression Head）

估计 suspect model 与 ancestor 的**血缘度**：
- 输出：0-1 之间的连续值
- 训练目标：MSE损失
- 应用：判断模型是"直系亲属"还是"远房亲戚"

#### 4.3 探针分类头（Probe Classification Head）

识别哪些**探针样本**被 suspect model "记住"（继承）：
- 输入：探针样本的行为特征
- 输出：每个探针是否保留（0/1）
- 应用：可解释性——展示哪些能力被继承/丢失

#### 4.4 架构验证头（Architecture Verification Head）

验证 suspect model 的架构是否与 ancestor 兼容：
- 检测模型结构相似度
- 防止架构欺骗攻击

---

### Step 5: 评估指标

类比论文的评估方法：

#### 5.1 连续评估

| 指标 | 说明 |
|-----|------|
| **MSE** | 血缘度估计的均方误差 |
| **ACC (±0.15)** | 血缘度误差在0.15以内的准确率 |
| **Spearman ρ** | 与真实血缘度的相关性 |

#### 5.2 二分类评估（设定BST阈值）

| 阈值 | 应用场景 |
|-----|---------|
| **BST=0.5** | 判断是否属于家族（亲子鉴定） |
| **BST=0.2** | 判断是否经过大幅修改（版权侵权） |
| **BST=0.8** | 判断是否几乎未变（盗窃检测） |

---

## 应用场景

### 场景1：模型盗窃检测
```
Suspect Model: 某商用模型
Ancestor: LLaMA-2-7B
检测结果：血缘度=0.95
结论：高度怀疑为LLaMA-2的直接复制，违反许可证
```

### 场景2：合法微调认证
```
Suspect Model: 某领域适配模型
Ancestor: LLaMA-2-7B
检测结果：血缘度=0.45
结论：经过充分微调，可认定为衍生作品，合法
```

### 场景3：家族树重建
```
通过连续血缘度估计，重建模型演化树：
LLaMA-2-7B (根节点)
    ├── Model A (0.85) → Model C (0.72)
    ├── Model B (0.60) → Model D (0.35)
    └── Model E (0.90)
```

---

## 技术优势对比

| 特性 | 传统指纹 | EverTracer | 本方案（FamilyTracer） |
|-----|---------|-----------|---------------------|
| **检测粒度** | 二分类 | 二分类 | ✅ 连续血缘度 |
| **血缘追踪** | ❌ 无法 | ❌ 无法 | ✅ 家族树重建 |
| **可解释性** | 低 | 中 | ✅ 探针级分析 |
| **对抗鲁棒性** | 弱 | 强 | ✅ 行为级检测 |
| **多代追踪** | ❌ 不能 | ❌ 不能 | ✅ 跨代追踪 |

---

## 实验设计建议

### 数据集构建
1. **基础模型**：LLaMA-2, Mistral, Falcon, Qwen
2. **变体生成**：
   - 不同微调步数（100, 500, 1000, 5000 steps）
   - 不同微调方法（LoRA, QLoRA, Full Fine-tune）
   - 不同领域数据（医学、法律、代码）
3. **混合模型**：模型合并（Model Merging）产物

### 基线对比
- **传统指纹**：IF, HashChain, ProFlingo
- **EverTracer**：记忆化检测方法
- **本方案**：FamilyTracer

---

## 结论

本方案将论文中"人类参与度检测"的核心思想迁移到"模型血缘追踪"领域：

1. **探针数据集** → 替代 Prompt，标记模型行为
2. **ModelBehaviorScore** → 替代 BERTScore，度量模型相似度
3. **连续血缘度** → 替代 Human Involvement，量化家族关系
4. **双头检测器** → 同时估计血缘度和识别继承的探针

这种方法能够实现：
- ✅ **细粒度**：连续的血缘度估计（0-1）
- ✅ **可解释**：展示哪些能力被继承/丢失
- ✅ **家族树**：重建模型演化历史
- ✅ **抗攻击**：基于行为而非参数，抗剪枝/合并
