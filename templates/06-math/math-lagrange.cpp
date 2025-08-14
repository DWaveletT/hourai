/**
## 定理

给定 $n$ 个横坐标不同的点 $(x_i, y_i)$，可以唯一确定一个 $n - 1$ 阶多项式如下：
$$
f(x) = \sum_{i=1}^n \frac{\prod_{j\neq i} (x-x_j)}{\prod_{j\neq i}(x_i-x_j)} \cdot y_i
$$
**/
