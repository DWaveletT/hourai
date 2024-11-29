## 定理

给定 $n$ 个横坐标不同的点 $(x_i, y_i)$，可以唯一确定一个 $n - 1$ 阶多项式如下：

$$
f(x) = \sum_{i=1}^n \frac{\prod_{j\neq i} (x-x_j)}{\prod_{j\neq i}(x_i-x_j)} \cdot y_i
$$

下面代码先求出了多项式再计算 $f(k)$，也可以直接带入计算。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int MAXN = 2e3 + 3;
const int MOD  = 998244353;
int X[MAXN], Y[MAXN], F[MAXN], G[MAXN], H[MAXN], A[MAXN];
int power(int a, int b){
    int r = 1;
    while(b){
        if(b & 1) r = 1ll * r * a % MOD;
        b >>= 1,  a = 1ll * a * a % MOD;
    }
    return r;
}
int main(){
    int n, k;
    cin >> n >> k;
    for(int i = 1;i <= n;++ i){
        cin >> X[i] >> Y[i];
    }
    F[0] = 1;
    for(int i = 1;i <= n;++ i){     // 计算 prod(x - x_i)
        for(int j = 0;j <= n;++ j){
            G[j] = ((j == 0 ? 0 : F[j - 1]) - 1ll * F[j] * X[i] % MOD + MOD) % MOD;
        }
        for(int j = 0;j <= n;++ j){
            F[j] = G[j];
        }
    }
    for(int i = 1;i <= n;++ i){
        for(int j = 0;j <= n;++ j){
            G[j] = F[j];
        }
        for(int j = n;j >= 0;-- j){ // 计算 prod(x - x_j) / (x - x_i)
            H[j] = G[j + 1];
            G[j] = (G[j] + 1ll * H[j] * X[i]) % MOD;
        }
        int w = 1;                  // 计算 inv(prod(x_i - x_j))
        for(int j = 1;j <= n;++ j) if(j != i)
            w = 1ll * w * (X[i] - X[j] + MOD) % MOD;
        w = 1ll * power(w, MOD - 2) * Y[i] % MOD;
        for(int j = 0;j <= n;++ j)
            A[j] = (A[j] + 1ll * w * H[j]) % MOD;
    }
    int t = 1, ans = 0;
    for(int i = 0;i <= n - 1;++ i){
        ans = (ans + 1ll * A[i] * t) % MOD;
        t = 1ll * t * k % MOD;
    }
    cout << ans << endl;
}
```
