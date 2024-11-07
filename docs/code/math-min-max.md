## 定理

$$\begin{aligned}
\max_{i\in S} \{x_i\} &= \sum_{T\subseteq S}(-1)^{|T| - 1}\min_{j\in T}\{x_j\} \\
\min_{i\in S} \{x_i\} &= \sum_{T\subseteq S}(-1)^{|T| - 1}\max_{j\in T}\{x_j\} \\
\end{aligned}$$

期望意义下上式依然成立。

另外设 $\max^k$ 表示第 $k$ 大的元素，可以推广为如下式子：

$$
\max_{i\in S}^k \{x_i\} = \sum_{T\subseteq S}(-1)^{|T| - k}\binom{|T - 1|}{k - 1} \min_{j\in T}\{x_j\}
$$

此外在数论上可以得到：

$$
\operatorname*{lcm}_{i\in S} \{x_i\} = \prod_{T\subseteq S} \left(\gcd_{j\in T}\{x_j\}\right)^{(-1)^{|T| - 1}}
$$

## 应用

对于计算“$n$ 个属性都出现的期望时间”问题，设第 $i$ 个属性第一次出现的时间是 $t_i$，所求即为 $\max(t_i)$，使用 min-max 容斥转为计算 $\min(t_i)$。

比如 $n$ 个独立物品，每次抽中物品 $i$ 的概率是 $p_i$，问期望抽多少次抽中所有物品。那么就可以计算 $\min_S$ 表示第一次抽中物品集合 $S$ 内物品的时间，可以得到：

$$\max_{U}=\sum_{S\subseteq U}(-1)^{|S| - 1}\min_S = \sum_{S\subseteq U}(-1)^{|S| - 1}\cdot \frac{1}{\sum _{x\in S}p_x}$$

```cpp

```
