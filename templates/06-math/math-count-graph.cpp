/**
## LGV 定理叙述

设 $G$ 是一张有向无环图，边带权，每个点的度数有限。给定起点集合 $A=\{a_1,a_2, \cdots,a_n\}$，终点集合 $B = \{b_1, b_2, \cdots,b_n\}$。

- 一段路径 $p:v_0\to^{w_1} v_1\to^{w_2} v_2\to \cdots \to^{w_k} v_k$ 的边权被定义为 $\omega (p) = \prod w_i$。
- 一对顶点 $(a, b)$ 的权值定义为 $e(a, b) = \sum_{p:a\to b}\omega (p)$。

设矩阵 $M$ 如下：
$$
M = \begin{pmatrix}
e(a_1, b_1) & e(a_1, b_2) & \cdots & e(a_1, b_n) \\
e(a_2, b_1) & e(a_2, b_2) & \cdots & e(a_2, b_n) \\
\vdots & \vdots & \ddots & \vdots \\
e(a_n, b_1) & e(a_n, b_2) & \cdots & e(a_n, b_n) \\
\end{pmatrix}
$$ 从 $A$ 到 $B$ 得到一个**不相交**的路径组 $p=(p_1, p_2, \cdots,p_n)$，其中从 $a_i$ 到达 $b_{\pi_i}$，$\pi$ 是一个排列。定义 $\sigma(\pi)$ 是 $\pi$ 逆序对的数量。

给出 LGV 的叙述如下：
$$
\det(M) = \sum_{p:A\to B} (-1)^{\sigma (\pi)} \prod_{i=1}^n \omega(p_i)
$$

可以将边权视作边的重数，那么 $e(a, b)$ 就可以视为从 $a$ 到 $b$ 的不同路径方案数。

## 矩阵树定理

对于无向图，

- 定义度数矩阵 $D_{i, j} = [i=j]\deg(i)$；
- 定义邻接矩阵 $E_{i, j} = E_{j, i}$ 是从 $i$ 到 $j$ 的边数个数；
- 定义拉普拉斯矩阵 $L = D - E$。

对于无向图的矩阵树定理叙述如下：
$$t(G) = \det(L_i) = \frac{1}{n}\lambda_1\lambda_2\cdots \lambda_{n-1}$$

其中 $L_i$ 是将 $L$ 删去第 $i$ 行和第 $i$ 列得到的子式。

对于有向图，类似于无向图定义入度矩阵、出度矩阵、邻接矩阵 $D^{\mathrm{in}}, D^{\mathrm{out}}, E$，同时定义拉普拉斯矩阵 $L^{\mathrm{in}} = D^{\mathrm{in}} - E,L^{\mathrm{out}} - E$。
$$\begin{aligned}
t^{\mathrm{leaf}}(G, k) &= \det(L^{\mathrm{in}}_k) \\
t^{\mathrm{root}}(G, k) &= \det(L^{\mathrm{out}}_k) \\
\end{aligned}$$

其中 $t^{\mathrm{leaf}}(G, k)$ 表示以 $k$ 为根的叶向树，$t^{\mathrm{root}}(G, k)$ 表示以 $k$ 为根的根向树。

## BEST 定理

对于一个有向欧拉图 $G$，记点 $i$ 的出度为 $\mathrm{out}_ i$，同时 $G$ 的根向生成树个数为 $T$。$T$ 可以任意选取根。则 $G$ 的本质不同的欧拉回路个数为：
$$T \prod_{i}(\mathrm{out}_i - 1)!$$

## 图连通方案数

$n$ 个点的图，$k$ 个连通块，第 $i$ 个大小为 $s_i$，添加 $k-1$ 条边使得它们连通的方案数：
$$
Ans=n^{k-2}\cdot\prod_{i=1}^{k}s_i
$$

**/
