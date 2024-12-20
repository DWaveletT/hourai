## 用法

对于积性函数 $f$，找到易求前缀和的积性函数 $g, h$ 使得 $h = f*g$，根据递推式计算 $S(n) = \sum_{i=1}^n f(i)$：

$$
S(n) = H(n) - \sum_{d = 1}^n g(d) \times S(\left\lfloor \frac{n}{d}\right\rfloor)
$$

## 例题

- 对于 $f = \varphi$，寻找 $g = 1, h = \mathrm{id}$；
- 对于 $f = \mu$，寻找 $g = 1, h = \varepsilon$。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int MAXN = 1e7 + 3;
const int H = 1e7;
int P[MAXN], p; bool V[MAXN];
long long ph[MAXN], sph[MAXN];
long long mu[MAXN], smu[MAXN];
long long tp[MAXN];
long long solve_ph(long long N){
    for(int d = N / H;d >= 1;-- d){
        long long n = N / d;
        long long wh = 1ll * n * (n + 1) / 2;
        tp[d] = wh;
        for(long long l = 2, r;l <= n;l = r + 1){
            r = n / (n / l);
            long long wg = r - l + 1;
            long long ws = n / l <= H ? sph[n / l] : tp[N / (n / l)];
            tp[d] -= wg * ws;
        }
    }
    return N <= H ? sph[N] : tp[1];
}
long long solve_mu(long long N){
    for(int d = N / H;d >= 1;-- d){
        long long n = N / d;
        long long wh = 1;
        tp[d] = wh;
        for(long long l = 2, r;l <= n;l = r + 1){
            r = n / (n / l);
            long long wg = r - l + 1;
            long long ws = n / l <= H ? smu[n / l] : tp[N / (n / l)];
            tp[d] -= wg * ws;
        }
    }
    return N <= H ? smu[N] : tp[1];
}
int main(){
    ios :: sync_with_stdio(false);
    cin.tie(nullptr);
    ph[1] = 1;
    mu[1] = 1;
    for(int i = 2;i <= H;++ i){
        if(!V[i]){
            P[++ p] = i;
            ph[i] = i - 1;
            mu[i] = -1;
        }
        for(int j = 1;j <= p && P[j] <= H / i;++ j){
            int &p = P[j];
            V[i * p] = true;
            if(i % p == 0){
                ph[i * p] = ph[i] * p;
                mu[i * p] = 0;
                break;
            } else {
                ph[i * p] = ph[i] * (p - 1);
                mu[i * p] = -mu[i];
            }
        }
    }
    for(int i = 1;i <= H;++ i){
        sph[i] = sph[i - 1] + ph[i];
        smu[i] = smu[i - 1] + mu[i];
    }
    int T;
    cin >> T;
    while(T --> 0){
        int n;
        cin >> n;
        cout << solve_ph(n) << " " << solve_mu(n) << "\n";
    }
    return 0;
}
```
