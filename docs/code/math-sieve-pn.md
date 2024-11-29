## 用法
对于积性函数 $f(x)$，寻找积性函数 $g(x)$ 使得 $g(p) = f(p)$，且 $g$ 易求前缀和 $G$。

令 $h = f * g^{-1}$，可以证明只有 PN 处 $h$ 的函数值非 $0$，PN 指每个素因子幂次都不小于 $2$ 的数。同时可以证明 $n$ 以内的 PN 只有 $\mathcal O(\sqrt n)$ 个，且可以暴力枚举质因子幂次得到所有 PN。

可利用下面公式计算 $h(p^c)$：

$$
h(p^c) = f(p^c) - \sum_{i = 1}^c g(p^i) \times h(p^{c - i})
$$

## 例题

> 定义积性函数 $f(x)$ 满足 $f(p^k) = p^k(p^k - 1)$，计算 $\sum f(i)$。

取 $g(p) = \mathrm{id}(p)\varphi(p) = f(p)$，根据 $g * \mathrm{id} = \mathrm{id}_2$ 利用杜教筛求解。$h(p^c)$ 的值利用递推式进行计算。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int MAXN = 1e7 + 3;
const int MAXM = 1e5 + 3;
const int H = 1e7;
const int MOD = 1e9 + 7;
const int DIV2 = 500000004;
const int DIV6 = 166666668;
int P[MAXN], p; bool V[MAXN];
int g[MAXN], le[MAXN], ge[MAXN];
int s1(long long n){    // 1^1 + 2^1 + ... + n^1
    n %= MOD;
    return 1ll * n * (n + 1) % MOD * DIV2 % MOD;
}
int s2(long long n){    // 1^2 + 2^2 + ... + n^2
    n %= MOD;
    return 1ll * n * (n + 1) % MOD * (2 * n + 1) % MOD * DIV6 % MOD;
}
int sg(long long n, long long N){
    return n <= H ? le[n] : ge[N / n];
}
int sieve_du(long long N){
    for(int d = N / H;d >= 1;-- d){
        long long n = N / d;
        int wh = s2(n);
        for(long long l = 2, r;l <= n;l = r + 1){
            r = n / (n / l);
            int wg = (s1(r) - s1(l - 1) + MOD) % MOD;
            int ws = sg(n / l, N);
            ge[d] = (ge[d] + 1ll * wg * ws) % MOD;
        }
        ge[d] = (wh - ge[d] + MOD) % MOD;
    }
    return N <= H ? le[N] : ge[1];
}
vector <int> hc[MAXM], gc[MAXM];
int ANS;
void sieve_pn(int last, long long x, int h, long long N){
    ANS = (ANS + 1ll * h * sg(N / x, N)) % MOD;
    for(long long i = last + 1;x <= N / P[i] / P[i];++ i){
        int c = 2;
        for(long long t = x * P[i] * P[i];t <= N;t *= P[i], c ++){
            int hh = 1ll * h * hc[i][c] % MOD;
            sieve_pn(i, t, hh, N);
        }
    }
}
int main(){
    ios :: sync_with_stdio(false);
    cin.tie(nullptr);
    g[1] = 1;
    for(int i = 2;i <= H;++ i){
        if(!V[i]){
            P[++ p] = i, g[i] = 1ll * i * (i - 1) % MOD;
        }
        for(int j = 1;j <= p && P[j] <= H / i;++ j){
            int &p = P[j];
            V[i * p] = true;
            if(i % p == 0){
                g[i * p] = 1ll * g[i] * p % MOD * p % MOD;
                break;
            } else {
                g[i * p] = 1ll * g[i] * p % MOD * (p - 1) % MOD;
            }
        }
    }
    for(int i = 1;i <= H;++ i){
        le[i] = (le[i - 1] + g[i]) % MOD;
    }
    long long N;
    cin >> N;
    for(int i = 1;i <= p && 1ll * P[i] * P[i] <= N;i ++){
        int &p = P[i];
        hc[i].push_back(1);
        gc[i].push_back(1);
        for(long long c = 1, t = p;t <= N;t = t * p, ++ c){
            if(c == 1){
                gc[i].push_back(1ll * p * (p - 1) % MOD);
            } else {
                gc[i].push_back(1ll * gc[i].back() * p % MOD * p % MOD);
            }
            int w = 1ll * (t % MOD) * ((t - 1) % MOD) % MOD;
            int s = 0;
            for(int j = 1;j <= c;++ j){
                s = (s + 1ll * gc[i][j] * hc[i][c - j]) % MOD;
            }
            hc[i].push_back((w - s + MOD) % MOD);
        }
    }
    sieve_du(N);
    sieve_pn(0, 1, 1, N);
    cout << ANS << "\n";
    return 0;
}
```
