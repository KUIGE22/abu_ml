# API 文档

本文档汇总了在 Jupyter 笔记本中发现的公开函数与类。公开项是名称不以下划线开头的顶级定义。

说明：这些 API 定义在笔记本中。要使用它们，请先在笔记本中执行定义单元，或将其重构为 Python 模块。

## ipython/1-初识机器学习.ipynb

### cross_entropy

- 类型: 函数
- 定义单元格: 22
- 签名: `cross_entropy(y, p)`

说明:

> 交叉熵

### score

- 类型: 函数
- 定义单元格: 15
- 签名: `score(x, w, b)`

说明:

> 线性函数

### sigmoid

- 类型: 函数
- 定义单元格: 16
- 签名: `sigmoid(s)`

说明:

> sigmoid函数

### softmax

- 类型: 函数
- 定义单元格: 17
- 签名: `softmax(s)`

说明:

> softmax函数

示例:

```python
plt.plot(x, softmax(scores).T, linewidth=2)
```

### step

- 类型: 函数
- 定义单元格: 29
- 签名: `step(x, d_yx)`

示例:

```python
    x = step(x, d_yx)
```


## ipython/2-机器学习进阶.ipynb

### entropy

- 类型: 函数
- 定义单元格: 84
- 签名: `entropy(P)`

说明:

> 根据每个样本出现的概率，计算信息量，输入P是数据集上每个数值统计的频率（概率）向量

### gbdt_model

- 类型: 函数
- 定义单元格: 108
- 签名: `gbdt_model(x_train, x_test, y_train, y_test)`

说明:

> 返回训练好的GBDT模型及分数

### loss_func

- 类型: 函数
- 定义单元格: 28
- 签名: `loss_func(X, W, b, y)`

### lr_model

- 类型: 函数
- 定义单元格: 108
- 签名: `lr_model(x_train, x_test, y_train, y_test)`

说明:

> 返回训练好的逻辑分类模型及分数

### mean_absolute_error

- 类型: 函数
- 定义单元格: 63
- 签名: `mean_absolute_error(y, y_pred)`

### mean_squared_error

- 类型: 函数
- 定义单元格: 65
- 签名: `mean_squared_error(y, y_pred)`

### median_absolute_error

- 类型: 函数
- 定义单元格: 64
- 签名: `median_absolute_error(y, y_pred)`

### r2_score

- 类型: 函数
- 定义单元格: 66
- 签名: `r2_score(y, y_pred)`

示例:

```python
r2_score(y_test, y_pred)
```

### rf_model

- 类型: 函数
- 定义单元格: 108
- 签名: `rf_model(x_train, x_test, y_train, y_test)`

说明:

> 返回训练好的随机森林模型及分数

### set_cabin_type

- 类型: 函数
- 定义单元格: 12
- 签名: `set_cabin_type(p_df)`

示例:

```python
data_train_fix1 = set_cabin_type(data_train_fix1)
```

### set_cabin_type

- 类型: 函数
- 定义单元格: 79
- 签名: `set_cabin_type(p_df)`

示例:

```python
data_train_fix1 = set_cabin_type(data_train_fix1)
```

### set_cabin_type

- 类型: 函数
- 定义单元格: 86
- 签名: `set_cabin_type(p_df)`

示例:

```python
data_train = set_cabin_type(data_train)
```

### set_cabin_type

- 类型: 函数
- 定义单元格: 94
- 签名: `set_cabin_type(p_df)`

### set_missing_ages

- 类型: 函数
- 定义单元格: 8
- 签名: `set_missing_ages(p_df)`

示例:

```python
data_train = set_missing_ages(data_train)
```

### set_missing_ages

- 类型: 函数
- 定义单元格: 77
- 签名: `set_missing_ages(p_df)`

说明:

> 均值特征填充

示例:

```python
data_train = set_missing_ages(data_train)
```

### set_missing_ages

- 类型: 函数
- 定义单元格: 86
- 签名: `set_missing_ages(p_df)`

示例:

```python
data_train = set_missing_ages(data_train)
```

### set_missing_ages

- 类型: 函数
- 定义单元格: 94
- 签名: `set_missing_ages(p_df)`

### set_missing_ages2

- 类型: 函数
- 定义单元格: 78
- 签名: `set_missing_ages2(p_df)`

说明:

> 回归模型预测特征填充

### stack_models

- 类型: 函数
- 定义单元格: 108
- 签名: `stack_models(x_train, x_test, y_train, y_test)`

说明:

> 返回融合后的模型及分数

### three_kfolder

- 类型: 函数
- 定义单元格: 113
- 签名: `three_kfolder(data, n_folds=5, shuffle=True, ratios=[4, 1, 2])`

说明:

> 按ratios数组随机(shuffle)三分割数据集，返回：traing_set, stacking_set, testing_set

### train_val

- 类型: 函数
- 定义单元格: 79
- 签名: `train_val(data)`

说明:

> 封装所有处理训练步骤

示例:

```python
train_val(data_train_fix1)
```


## ipython/5-深层学习模型.ipynb

### AuditoryBrain

- 类型: 类
- 定义单元格: 5
- 签名: `AuditoryBrain`

说明:

> 听觉皮层

方法:

- `process(self, x)`
  - 处理信号

### AuditoryBrain

- 类型: 类
- 定义单元格: 7
- 签名: `AuditoryBrain`

说明:

> 听觉皮层

方法:

- `process(self, x)`
  - 处理信号

### Brain

- 类型: 类
- 定义单元格: 5
- 签名: `Brain`

说明:

> 脑皮层

方法:

- `process(self, x)`
  - 根据不同的传入信号，传递给不同的皮层组织处理

示例:

```python
brain = Brain()
```

### Neuron

- 类型: 类
- 定义单元格: 3
- 签名: `Neuron(object)`

说明:

> 神经元

方法:

- `spike(self, x)`
  - 神经元激活函数。输入某种类型的刺激信号，有可能激活神经元响应刺激

示例:

```python
        self.neurons = [Neuron('视觉信号') for i in range(num)]
```

### Neuron

- 类型: 类
- 定义单元格: 11
- 签名: `Neuron(object)`

说明:

> 神经元

方法:

- `spike(self, x)`
  - 输入某种类型的刺激信号，有可能激活神经元响应刺激

### SignalInput

- 类型: 类
- 定义单元格: 3
- 签名: `SignalInput(object)`

说明:

> 输入信号

示例:

```python
x_see = SignalInput('视觉信号', '一只猫在卖萌!')
```

### VisualBrain

- 类型: 类
- 定义单元格: 5
- 签名: `VisualBrain`

说明:

> 视觉皮层

方法:

- `process(self, x)`
  - 处理信号

### VisualBrain

- 类型: 类
- 定义单元格: 7
- 签名: `VisualBrain`

说明:

> 视觉皮层

方法:

- `process(self, x)`
  - 处理信号

### relu

- 类型: 函数
- 定义单元格: 19
- 签名: `relu(x)`


## ipython/6-学习空间特征.ipynb

### compare_imgs

- 类型: 函数
- 定义单元格: 16
- 签名: `compare_imgs(imgs, titles=[])`

说明:

> 对比图片

示例:

```python
compare_imgs([origin, cov], ['original', 'cov_mean'])
```

### ensure_dir

- 类型: 函数
- 定义单元格: 29
- 签名: `ensure_dir(dir_path)`

示例:

```python
ensure_dir(target + 'weights')
```

### show_img

- 类型: 函数
- 定义单元格: 11
- 签名: `show_img(img)`

说明:

> 展示图片

示例:

```python
show_img(cov)
```


## ipython/8-处理时间序列.ipynb

### cos_similar

- 类型: 函数
- 定义单元格: 6
- 签名: `cos_similar(v1, v2)`

说明:

> 用余弦向量判断相似程度

示例:

```python
print(cos_similar(puppy_vec, dog_vec))
```

### euc_distance

- 类型: 函数
- 定义单元格: 6
- 签名: `euc_distance(v1, v2)`

说明:

> 用欧氏距离判断相似距离

示例:

```python
print(euc_distance(puppy_vec, dog_vec))
```

### process_text

- 类型: 函数
- 定义单元格: 3
- 签名: `process_text(text)`

说明:

> 将标点符号替换成空格

### text2vec

- 类型: 函数
- 定义单元格: 3
- 签名: `text2vec(text)`

说明:

> 将文本转换成向量

示例:

```python
text2vec(text)
```


## lecture/泰坦尼克号(上)——LR.ipynb

### set_cabin_type

- 类型: 函数
- 定义单元格: 26
- 签名: `set_cabin_type(p_df)`

### set_missing_ages

- 类型: 函数
- 定义单元格: 16
- 签名: `set_missing_ages(p_df)`

说明:

> 均值特征填充

### set_missing_ages2

- 类型: 函数
- 定义单元格: 18
- 签名: `set_missing_ages2(p_df)`

说明:

> 回归模型预测特征填充


## lecture/泰坦尼克号(下)——决策树和集成学习.ipynb

### entropy

- 类型: 函数
- 定义单元格: 3
- 签名: `entropy(P)`

说明:

> 根据每个样本出现的概率，计算信息量，输入P是数据集上每个数值统计的频率（概率）向量

示例:

```python
H = entropy(p)
```

