```cpp
#include <bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MOD = 571373;

namespace Seg{
    #define lc(t) (t << 1)
    #define rc(t) (t << 1 | 1)
    const int SIZ = 4e5 + 3;
    int S[SIZ];
    int Tmul[SIZ], Tadd[SIZ];

    void pushup(int t, int a, int b){
        S[t] = (S[lc(t)] + S[rc(t)]) % MOD;
    }
    void pushdown(int t, int a, int b){
        if(Tmul[t] != 1){
            int l = lc(t), r = rc(t);
            int c = a + b >> 1;
            Tmul[l] = 1ll * Tmul[l] * Tmul[t] % MOD;
            Tadd[l] = 1ll * Tadd[l] * Tmul[t] % MOD;
            S[l] = 1ll * S[l] * Tmul[t] % MOD;
            Tmul[r] = 1ll * Tmul[r] * Tmul[t] % MOD;
            Tadd[r] = 1ll * Tadd[r] * Tmul[t] % MOD;
            S[r] = 1ll * S[r] * Tmul[t] % MOD;
            Tmul[t] = 1;
        }
        if(Tadd[t] != 0){
            int l = lc(t), r = rc(t);
            int c = a + b >> 1;
            S[l] = (S[l] + 1ll * (c - a + 1) * Tadd[t]) % MOD;
            Tadd[l] = (Tadd[l] + Tadd[t]) % MOD;
            S[r] = (S[r] + 1ll * (b - c    ) * Tadd[t]) % MOD;
            Tadd[r] = (Tadd[r] + Tadd[t]) % MOD;
            Tadd[t] = 0;
        }
    }

    void modify1(int t, int a, int b, int l, int r, int w){
        if(l <= a && b <= r){
            S[t] = 1ll * S[t] * w % MOD;
            
            Tmul[t] = 1ll * Tmul[t] * w % MOD;
            Tadd[t] = 1ll * Tadd[t] * w % MOD;
        } else {
            int c = a + b >> 1;
            pushdown(t, a, b);
            if(l <= c) modify1(lc(t), a, c, l, r, w);
            if(r >  c) modify1(rc(t), c + 1, b, l, r, w);
            pushup(t, a, b);
        }
    }

    void modify2(int t, int a, int b, int l, int r, int w){
        if(l <= a && b <= r){
            S[t] = (S[t] + 1ll * (b - a + 1) * w) % MOD;
            Tadd[t] = (Tadd[t] + w) % MOD;
        } else {
            int c = a + b >> 1;
            pushdown(t, a, b);
            if(l <= c) modify2(lc(t), a, c, l, r, w);
            if(r >  c) modify2(rc(t), c + 1, b, l, r, w);
            pushup(t, a, b);
        }
    }

    int query(int t, int a, int b, int l, int r){
        if(l <= a && b <= r){
            return S[t];
        } else {
            int c = a + b >> 1;
            int ans = 0;
            pushdown(t, a, b);
            if(l <= c) ans += query(lc(t), a, c, l, r);
            if(r >  c) ans += query(rc(t), c + 1, b, l, r);
            ans %= MOD;
            
            pushup(t, a, b);
            return ans;
        }
    }
}

// ===== TEST =====

int qread(){
    int w = 1, c, ret;
    while((c = getchar()) >  '9' || c <  '0') w = (c == '-' ? -1 : 1); ret = c - '0';
    while((c = getchar()) >= '0' && c <= '9') ret = ret * 10 + c - '0';
    return ret * w;
}

int main(){
    int n = qread(), m = qread(); qread();
    for(int i = 1;i <= n;++ i){
        int a = qread();
        Seg :: modify2(1, 1, n, i, i, a);
    }
    for(int i = 1;i <= m;++ i){
        int op = qread();
        if(op == 1){
            int l = qread(), r = qread();
            int w = qread();
            Seg :: modify1(1, 1, n, l, r, w);
        } else
        if(op == 2){
            int l = qread(), r = qread();
            int w = qread();
            Seg :: modify2(1, 1, n, l, r, w);
        } else {
            int l = qread(), r = qread();
            printf("%d\n", Seg :: query(1, 1, n, l, r));
        }
    }
    return 0;
}
```
