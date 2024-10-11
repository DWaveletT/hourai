```cpp
#include<bits/stdc++.h>
using namespace std;

// Pn Sieve:
// Find g that g(p) = f(p) and easy to calc G(n)
// Let h = f / g, h(p^c) = f(p^c) - sum(g(p^i) * h(p^{c - i}), i = 1, 2, ..., c))
// Dfs Pn numbers(c_i >= 2)
// Ans = sum(h(x) * G(n / x), x is Pn number)

// Code for [template] Min_25:
// > f(p^k) = p^k(p^k - 1)
// > f(a*b) = f(a) * (b), gcd(a, b) = 1
// f(p) = p(p - 1) = id(p) * phi(p) = g(n)
// g * id = id_2 (P.S. id_2(n) = n^2), so we can du sieve it.

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
