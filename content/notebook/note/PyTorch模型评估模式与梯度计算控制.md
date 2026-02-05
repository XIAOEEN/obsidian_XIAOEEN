## PyTorch中 self.model.eval() 与 with torch.no_grad(): 的区别

## 概述

在PyTorch中进行模型推理或验证时，我们通常会看到两种模式组合使用：
- `model.eval()` - 将模型设置为评估模式
- `with torch.no_grad():` - 禁用梯度计算

虽然它们经常一起使用，但实际上控制着完全不同的机制。

---

## 1. model.eval() - 控制模型的行为模式

### 作用

- 将模型从 **训练模式** (training mode) 切换到 **评估模式** (evaluation mode)
- 主要影响包含 **训练/评估行为差异** 的层

### 受影响的层

| 层类型       | 训练模式 (train)   | 评估模式 (eval)  | 示例                     |
| --------- | -------------- | ------------ | ---------------------- |
| Dropout   | 随机丢弃神经元 (按概率p) | 保留所有神经元      | `nn.Dropout(p=0.5)`    |
| BatchNorm | 使用当前batch统计量   | 使用移动平均统计量    | `nn.BatchNorm1d/2d/3d` |
| LayerNorm | 使用当前batch统计量   | 使用当前batch统计量 | 通常不受影响                 |

---
#### About BatchNorm and LayerNorm：归一化对象的差异

| 对比维度               | BatchNorm      | LayerNorm       |
| ------------------ | -------------- | --------------- |
| **归一化对象**          | **所有样本的同一个通道** | **同一样本所有特征**    |
| **计算维度**           | 通道维度           | 特征维度            |
| **是否依赖batch size** | ✅ 是            | ❌ 否             |
| **是否需要eval模式**     | ✅ 必须切换         | ❌ 无需切换          |
| **典型应用**           | CNN图像分类        | Transformer、RNN |

**形象理解**：
```python
# BatchNorm (按通道统计)
Input: [样本1, 样本2, 样本3, ..., 样本N]
       每个通道 → 计算 N 个样本的分布

# LayerNorm (按样本统计)
Input: [样本1, 样本2, 样本3, ..., 样本N]
       每个样本 → 计算自己所有特征的分布
```

---

### 代码示例

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.dropout(x)
        x = self.bn(x)
        return self.fc(x)

model = SimpleModel()
input_data = torch.randn(32, 64)

# 训练模式下
model.train()
output_train = model(input_data)  # Dropout会随机丢弃神经元，BN用当前batch统计

# 评估模式下
model.eval()
output_eval = model(input_data)   # Dropout不丢弃神经元，BN用移动平均统计
# 这里保存了整个计算图，如果需要的话可以进行反向传播
# 这里展示意外情况下进行梯度计算反向传播，然后参数更新
loss = criterion(output, target) # 评估损失
loss.backword()
optimzer.step() 
```


---

## 2. with torch.no_grad(): - 控制梯度计算

### 作用
- **禁用梯度计算** (gradient computation)
- 构建一个上下文环境，在该环境中的张量操作**不会记录计算图**
- **不影响模型的行为**，只影响PyTorch的自动求导机制

### 原理

```python
# 有梯度计算
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()  # 正常反向传播
print(x.grad)  # 输出: tensor(4.)

# 无梯度计算（用no_grad）
with torch.no_grad():
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2
    # y.backward()  # 这会报错，因为没有计算图
    print(x.grad)  # 输出: None
```

这里对比torch.no_grad() V.S. eval()的底层原理：
```python
# model.eval() 的本质
def eval(self):
	self.traing = False 
	for module in self.modules():
		module.traing = self.traing
	# 只影响self.traing属性，不影响requires_grad
	
# torch.no_grad() 的本质
class no_grad:
	def __enter__(self):
		# 禁用autograd引擎
		self.prev = torch.is_grad_enbled()  # 在禁用之前，存一下当前的grad状态，方便退出的时候恢复
		torch.set_gred_enabled(False) # 真正的禁用梯度计算
	def __exit__(self, exc_type, exc_val, exc_tb):
		torch.set_grad_enable(self, prev)
		
补充解释一下__enter__()和__exit__()函数的调用时机：
with torch.no_grad():
	# 这里触发__enter__()
	output = model(input_data)
	# 这里触发__exit__()
```
### 性能优势

使用 `torch.no_grad()` 可以：

1. **减少内存占用**
   - 不存储中间激活值（用于反向传播的梯度计算）
   - 内存使用通常可减少 50% 以上

2. **提高计算速度**
   - 跳过梯度计算的额外开销
   - 前向传播速度提升约 20-40%

### 代码示例

```python
import torch
import torch.nn as nn

model = nn.Linear(100, 10)
input_data = torch.randn(32, 100)

# 有梯度计算（会占用更多内存和计算资源）
output1 = model(input_data)
# 这里保存了整个计算图，如果需要的话可以进行反向传播

# 无梯度计算（更高效）
with torch.no_grad():
    output2 = model(input_data)
    # 计算图不保存，内存占用少，速度快
    loss
```

---

## 3. 对比总结

### 本质区别

| 特性           | model.eval()           | with torch.no_grad(): |
| ------------ | ---------------------- | --------------------- |
| **作用对象**     | **模型层的行为,不影响梯度更新**     | **禁用梯度更新，不保存计算图**     |
| **主要影响**     | Dropout、BatchNorm等层的行为 | 内存占用、计算速度             |
| **是否必需**     | 在推理时必须调用               | 可选，但强烈推荐              |
| **是否需要反向传播** | 无关                     | 明确禁用                  |

### 推理时的典型实践

```python
# 正确做法：组合使用
model.eval()  # 设置模型为评估模式

with torch.no_grad():  # 禁用梯度计算
    for data, target in test_loader:
        output = model(data)
        predictions = output.argmax(dim=1)
        # ... 计算指标 ...

# 不推荐：只使用model.eval()
model.eval()
for data, target in test_loader:
    output = model(data)  # 仍会计算梯度（浪费资源）
    predictions = output.argmax(dim=1)

# 不推荐：只使用no_grad
with torch.no_grad():
    model.train()  # 错误！Dropout仍在工作，BN用当前统计
    for data, target in test_loader:
        output = model(data)
```


### torch.inference_mode() (PyTorch 1.9+)

PyTorch 1.9+ 引入了比 `torch.no_grad()` 更优的选择：

```python
# 相比于 with torch.no_grad():
# - 禁用更多底层检查，速度更快
# - 不能进行任何需要梯度的操作（更安全）

with torch.inference_mode():
    output = model(data)
    # output.requires_grad 会返回 False
    # 更安全，性能更好
```

**性能对比**：
```python
import time

# torch.no_grad()
start = time.time()
with torch.no_grad():
    for _ in range(1000):
        output = model(input_data)
no_grad_time = time.time() - start

# torch.inference_mode()
start = time.time()
with torch.inference_mode():
    for _ in range(1000):
        output = model(input_data)
inference_time = time.time() - start

print(f"no_grad time: {no_grad_time:.3f}s")
print(f"inference_mode time: {inference_time:.3f}s")
print(f"Speedup: {(no_grad_time/inference_time - 1)*100:.1f}%")
```

**推荐在现代PyTorch版本中使用 `torch.inference_mode()`**。

---

## 参考资料

1. PyTorch官方文档: [Train vs Eval Mode](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.eval)
2. PyTorch官方文档: [No Grad](https://pytorch.org/docs/stable/generated/torch.no_grad.html)
3. PyTorch官方文档: [Inference Mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html)
4. PyTorch论坛讨论: [Why eval() and no_grad()](https://discuss.pytorch.org/t/what-does-model-eval-do/8901)
5. PyTorch性能最佳实践: [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

*最后更新: 2026-01-23*
