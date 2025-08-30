# API Documentation

This document summarizes public functions and classes discovered in the Jupyter notebooks. Public items are top-level names that do not start with an underscore.

Note: These APIs live in notebooks. To use them, execute the defining cell in a notebook, or refactor them into a Python module.

## ipython/1-初识机器学习.ipynb

### cross_entropy

- Kind: Function
- Defined in cell: 22
- Signature: `cross_entropy(y, p)`

Description:

> 交叉熵

### score

- Kind: Function
- Defined in cell: 15
- Signature: `score(x, w, b)`

Description:

> 线性函数

### sigmoid

- Kind: Function
- Defined in cell: 16
- Signature: `sigmoid(s)`

Description:

> sigmoid函数

### softmax

- Kind: Function
- Defined in cell: 17
- Signature: `softmax(s)`

Description:

> softmax函数

Example:

```python
plt.plot(x, softmax(scores).T, linewidth=2)
```

### step

- Kind: Function
- Defined in cell: 29
- Signature: `step(x, d_yx)`

Example:

```python
    x = step(x, d_yx)
```


## ipython/2-机器学习进阶.ipynb

### entropy

- Kind: Function
- Defined in cell: 84
- Signature: `entropy(P)`

Description:

> 根据每个样本出现的概率，计算信息量，输入P是数据集上每个数值统计的频率（概率）向量

### gbdt_model

- Kind: Function
- Defined in cell: 108
- Signature: `gbdt_model(x_train, x_test, y_train, y_test)`

Description:

> 返回训练好的GBDT模型及分数

### loss_func

- Kind: Function
- Defined in cell: 28
- Signature: `loss_func(X, W, b, y)`

### lr_model

- Kind: Function
- Defined in cell: 108
- Signature: `lr_model(x_train, x_test, y_train, y_test)`

Description:

> 返回训练好的逻辑分类模型及分数

### mean_absolute_error

- Kind: Function
- Defined in cell: 63
- Signature: `mean_absolute_error(y, y_pred)`

### mean_squared_error

- Kind: Function
- Defined in cell: 65
- Signature: `mean_squared_error(y, y_pred)`

### median_absolute_error

- Kind: Function
- Defined in cell: 64
- Signature: `median_absolute_error(y, y_pred)`

### r2_score

- Kind: Function
- Defined in cell: 66
- Signature: `r2_score(y, y_pred)`

Example:

```python
r2_score(y_test, y_pred)
```

### rf_model

- Kind: Function
- Defined in cell: 108
- Signature: `rf_model(x_train, x_test, y_train, y_test)`

Description:

> 返回训练好的随机森林模型及分数

### set_cabin_type

- Kind: Function
- Defined in cell: 12
- Signature: `set_cabin_type(p_df)`

Example:

```python
data_train_fix1 = set_cabin_type(data_train_fix1)
```

### set_cabin_type

- Kind: Function
- Defined in cell: 79
- Signature: `set_cabin_type(p_df)`

Example:

```python
data_train_fix1 = set_cabin_type(data_train_fix1)
```

### set_cabin_type

- Kind: Function
- Defined in cell: 86
- Signature: `set_cabin_type(p_df)`

Example:

```python
data_train = set_cabin_type(data_train)
```

### set_cabin_type

- Kind: Function
- Defined in cell: 94
- Signature: `set_cabin_type(p_df)`

### set_missing_ages

- Kind: Function
- Defined in cell: 8
- Signature: `set_missing_ages(p_df)`

Example:

```python
data_train = set_missing_ages(data_train)
```

### set_missing_ages

- Kind: Function
- Defined in cell: 77
- Signature: `set_missing_ages(p_df)`

Description:

> 均值特征填充

Example:

```python
data_train = set_missing_ages(data_train)
```

### set_missing_ages

- Kind: Function
- Defined in cell: 86
- Signature: `set_missing_ages(p_df)`

Example:

```python
data_train = set_missing_ages(data_train)
```

### set_missing_ages

- Kind: Function
- Defined in cell: 94
- Signature: `set_missing_ages(p_df)`

### set_missing_ages2

- Kind: Function
- Defined in cell: 78
- Signature: `set_missing_ages2(p_df)`

Description:

> 回归模型预测特征填充

### stack_models

- Kind: Function
- Defined in cell: 108
- Signature: `stack_models(x_train, x_test, y_train, y_test)`

Description:

> 返回融合后的模型及分数

### three_kfolder

- Kind: Function
- Defined in cell: 113
- Signature: `three_kfolder(data, n_folds=5, shuffle=True, ratios=[4, 1, 2])`

Description:

> 按ratios数组随机(shuffle)三分割数据集，返回：traing_set, stacking_set, testing_set

### train_val

- Kind: Function
- Defined in cell: 79
- Signature: `train_val(data)`

Description:

> 封装所有处理训练步骤

Example:

```python
train_val(data_train_fix1)
```


## ipython/5-深层学习模型.ipynb

### AuditoryBrain

- Kind: Class
- Defined in cell: 5
- Signature: `AuditoryBrain`

Description:

> 听觉皮层

Methods:

- `process(self, x)`
  - 处理信号

### AuditoryBrain

- Kind: Class
- Defined in cell: 7
- Signature: `AuditoryBrain`

Description:

> 听觉皮层

Methods:

- `process(self, x)`
  - 处理信号

### Brain

- Kind: Class
- Defined in cell: 5
- Signature: `Brain`

Description:

> 脑皮层

Methods:

- `process(self, x)`
  - 根据不同的传入信号，传递给不同的皮层组织处理

Example:

```python
brain = Brain()
```

### Neuron

- Kind: Class
- Defined in cell: 3
- Signature: `Neuron(object)`

Description:

> 神经元

Methods:

- `spike(self, x)`
  - 神经元激活函数。输入某种类型的刺激信号，有可能激活神经元响应刺激

Example:

```python
        self.neurons = [Neuron('视觉信号') for i in range(num)]
```

### Neuron

- Kind: Class
- Defined in cell: 11
- Signature: `Neuron(object)`

Description:

> 神经元

Methods:

- `spike(self, x)`
  - 输入某种类型的刺激信号，有可能激活神经元响应刺激

### SignalInput

- Kind: Class
- Defined in cell: 3
- Signature: `SignalInput(object)`

Description:

> 输入信号

Example:

```python
x_see = SignalInput('视觉信号', '一只猫在卖萌!')
```

### VisualBrain

- Kind: Class
- Defined in cell: 5
- Signature: `VisualBrain`

Description:

> 视觉皮层

Methods:

- `process(self, x)`
  - 处理信号

### VisualBrain

- Kind: Class
- Defined in cell: 7
- Signature: `VisualBrain`

Description:

> 视觉皮层

Methods:

- `process(self, x)`
  - 处理信号

### relu

- Kind: Function
- Defined in cell: 19
- Signature: `relu(x)`


## ipython/6-学习空间特征.ipynb

### compare_imgs

- Kind: Function
- Defined in cell: 16
- Signature: `compare_imgs(imgs, titles=[])`

Description:

> 对比图片

Example:

```python
compare_imgs([origin, cov], ['original', 'cov_mean'])
```

### ensure_dir

- Kind: Function
- Defined in cell: 29
- Signature: `ensure_dir(dir_path)`

Example:

```python
ensure_dir(target + 'weights')
```

### show_img

- Kind: Function
- Defined in cell: 11
- Signature: `show_img(img)`

Description:

> 展示图片

Example:

```python
show_img(cov)
```


## ipython/8-处理时间序列.ipynb

### cos_similar

- Kind: Function
- Defined in cell: 6
- Signature: `cos_similar(v1, v2)`

Description:

> 用余弦向量判断相似程度

Example:

```python
print(cos_similar(puppy_vec, dog_vec))
```

### euc_distance

- Kind: Function
- Defined in cell: 6
- Signature: `euc_distance(v1, v2)`

Description:

> 用欧氏距离判断相似距离

Example:

```python
print(euc_distance(puppy_vec, dog_vec))
```

### process_text

- Kind: Function
- Defined in cell: 3
- Signature: `process_text(text)`

Description:

> 将标点符号替换成空格

### text2vec

- Kind: Function
- Defined in cell: 3
- Signature: `text2vec(text)`

Description:

> 将文本转换成向量

Example:

```python
text2vec(text)
```


## lecture/泰坦尼克号(上)——LR.ipynb

### set_cabin_type

- Kind: Function
- Defined in cell: 26
- Signature: `set_cabin_type(p_df)`

### set_missing_ages

- Kind: Function
- Defined in cell: 16
- Signature: `set_missing_ages(p_df)`

Description:

> 均值特征填充

### set_missing_ages2

- Kind: Function
- Defined in cell: 18
- Signature: `set_missing_ages2(p_df)`

Description:

> 回归模型预测特征填充


## lecture/泰坦尼克号(下)——决策树和集成学习.ipynb

### entropy

- Kind: Function
- Defined in cell: 3
- Signature: `entropy(P)`

Description:

> 根据每个样本出现的概率，计算信息量，输入P是数据集上每个数值统计的频率（概率）向量

Example:

```python
H = entropy(p)
```

