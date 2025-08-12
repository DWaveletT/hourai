/**
## 形式

考虑一个经典的 dp 转移方程如下：

$$f_i = \max_{j < i}\{f(j) + w(j, i)\}$$

我们将式子拆成三个部分：只跟 $i$ 有关或者与 $i,j$ 均不相关的部分 $a(i)$，只跟 $j$ 有关的部分 $b(j)$，跟 $i,j$ 均有关的部分 $c(i,j)$：

$$f_{i} = a(i) + \max_{j<i} \{b(j)+c(i,j)\}$$

斜率优化可被用来解决这样一个情形：$c(i,j)=ic_j$。此时 $b(j)+c(i,j)$ 可视作关于 $j$ 的一次函数。如果 $c_j$ 随着 $j$ 的增大而单调，那么可用单调栈维护；否则可以考虑 CDQ 分治或者在凸包上二分。在凸包上可以使用二分查询最高/最低点。

## 例题

玩具装箱。原始转移方程为：

$$f_i = \max_{j< i}\{f_j + (s_i-s_j-L')^2\}$$

其中 $s_i = i+\sum_{j\le i}c_i, L'=L+1$。将其分类得到：

$$
\begin{aligned}
f_i &= \max_{j<i}\{f_j+s_i^2+s_j^2+L'^2-2s_is_j+2s_jL'-2s_iL' \} \\
&= (s_i^2 -2s_iL'+ L'^2) + \max_{j<i}\{(f_j+s_j^2+2s_jL') -2s_is_j \}
\end{aligned}
$$

在原始的玩具装箱中，$s_j$ 单调增加，也就是斜率单调增加。因此可以直接使用单调栈维护凸包。同时 $s_i$ 也单调增加，因此可以用指针维护。
**/
#include "../header.cpp"
int n, L, p, e, C[MAXN], Q[MAXN];
f80 S[MAXN], F[MAXN];
f80 gtx(int x){ return S[x]; }
f80 gty(int x){ return F[x] + S[x] * S[x]; }
f80 gtw(int x){ return -2.0 * (L - S[x]); }
f80 gtk(int x,int y){ return (gty(y) - gty(x)) / (gtx(y) - gtx(x)); }
int main(){ 
  cin >> n >> L;
  for(int i = 1;i <= n;++ i){
    cin >> C[i];
    S[i] = S[i - 1] + C[i];
  }
  for(int i = 1;i <= n;++ i){
    S[i] += i;
  }
  e = p = 1, L ++, Q[p] = 0;
  for(int i = 1;i <= n;++ i){
    while(e < p && gtk(Q[e], Q[e + 1]) < gtw(i))
      ++ e;
    int j = Q[e];
    F[i] = F[j] + pow(S[i] - S[j] - L, 2);
    while(1 < p && gtk(Q[p - 1], Q[p]) > gtk(Q[p], i))
      e -= (e == p), -- p;
    Q[++ p] = i;
  }
  printf("%.0Lf\n", F[n]);
  return 0;
}