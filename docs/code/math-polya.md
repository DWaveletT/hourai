## Burnside 引理

记所有染色方案的集合为 $X$，其中单个染色方案为 $x$。一种**对称操作** $g\in X$ 作用于染色方案 $x\in X$ 上可以得到另外一种染色 $x'$。

将所有对称操作作为集合 $G$，那么 $Gx = \{gx \mid g\in G\}$ 是**与 $x$ 本质相同的染色方案的集合**，形式化地称为 $x$ 的轨道。统计本质不同染色方案数，就是**统计不同轨道个数**。

Burnside 引理说明如下：

$$
|X / G| = \frac{1}{|G|} \sum_{g\in G}|X^g|
$$

其中 $X^g$ 表示在 $g\in G$ 的作用下，**不动点**的集合。不动点被定义为 $x = gx$ 的 $x$。

## Polya 定理

对于通常的染色问题，$X$ 可以看作一个长度为 $n$ 的序列，每个元素是 $1$ 到 $m$ 的整数。可以将 $n$ 看作面数、$m$ 看作颜色数。Polya 定理叙述如下：

$$
|X / G| = \frac{1}{|G|} \sum_{g\in G}\sum_{g\in G} m^{c(g)}
$$

其中 $c(g)$ 表示对一个序列做轮换操作 $g$ 可以**分解成多少个置换环**。

然而，增加了限制（比如要求某种颜色必须要多少个），就无法直接应用 Polya 定理，需要利用 Burnside 引理进行具体问题具体分析。

## 应用

给定 $n$ 个点 $n$ 条边的环，现在有 $n$ 种颜色，给每个顶点染色，询问有多少种本质不同的染色方案。

显然 $X$ 是全体元素在 $1$ 到 $n$ 之间长度为 $n$ 的序列，$G$ 是所有可能的单次旋转方案，共有 $n$ 种，第 $i$ 种方案会把 $1$ 置换到 $i$。于是：

$$
\begin{aligned}
\mathrm{ans} &= \frac{1}{|G|} \sum_{i=1}^n m^{c(g_i)} \\
&= \frac{1}{n} \sum_{i=1}^{n} n^{\gcd(i,n)} \\
&= \frac{1}{n} \sum_{d\mid n}^n n^{d} \sum_{i=1}^n [\gcd(i,n) = d] \\
&= \frac{1}{n} \sum_{d\mid n}^n n^{d} \varphi(n/d) \\
\end{aligned}
$$

```cpp
#include<bits/stdc++.h>
using namespace std;
const int MOD = 1e9 + 7;
int power(int a, int b){
    int r = 1;
    while(b){
        if(b & 1) r = 1ll * r * a % MOD;
        b >>= 1,  a = 1ll * a * a % MOD;
    }
    return r;
}
vector <tuple<int, int> > P;
void solve(int step, int n, int d, int f, int &ans){
    if(step == P.size()){
        ans = (ans + 1ll * power(n, n / d) * f) % MOD;
    } else {
        auto [w, c] = P[step];
        int dd = 1, ff = 1;
        for(int i = 0;i <= c;++ i){
            solve(step + 1, n, d * dd, f * ff, ans);
            ff = ff * (w - (i == 0));
            dd = dd * w;
        }
    }
}
int main(){
    int T;
    cin >> T;
    while(T --){
        int n, t;
        cin >> n;
        t = n;
        for(int i = 2;i * i <= n;++ i) if(n % i == 0){
            int w = i, c = 0;
            while(t % i == 0){
                t /= i, c ++;
            }
            P.push_back({ w, c });
        }
        if(t != 1){
            P.push_back({ t, 1 });
        }
        int ans = 0;
        solve(0, n, 1, 1, ans);
        ans = 1ll * ans * power(n, MOD - 2) % MOD;
        cout << ans << endl;
        P.clear();
    }
    return 0;
}
```
