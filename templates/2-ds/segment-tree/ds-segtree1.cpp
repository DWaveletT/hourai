#include <bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const i64 MOD = 1e18;

// 区间加、区间查询数值和
namespace Seg1{
    #define lc(t) (t << 1)
    #define rc(t) (t << 1 | 1)
    const int SIZ = 4e5 + 3;
    i64 S[SIZ], T[SIZ];

    void pushup(int t, int a, int b){
        S[t] = (S[lc(t)] + S[rc(t)]) % MOD;
    }
    void pushdown(int t, int a, int b){
        if(T[t]){
            int l = lc(t), r = rc(t);
            int c = a + b >> 1;
            S[l] = (S[l] + 1ll * (c - a + 1) * T[t]) % MOD;
            S[r] = (S[r] + 1ll * (b -     c) * T[t]) % MOD;
            T[l] = (T[l] + T[t]) % MOD;
            T[r] = (T[r] + T[t]) % MOD;
            T[t] = 0;
        }
    }

    void modify(int t, int a, int b, int l, int r, int w){
        if(l <= a && b <= r){
            T[t] = (T[t] + w) % MOD;
            S[t] = (S[t] + 1ll * (b - a + 1) * w) % MOD;
        } else {
            int c = a + b >> 1;
            pushdown(t, a, b);
            if(l <= c) modify(lc(t), a, c, l, r, w);
            if(r >  c) modify(rc(t), c + 1, b, l, r, w);
            pushup(t, a, b);
        }
    }

    i64 query(int t, int a, int b, int l, int r){
        if(l <= a && b <= r){
            return S[t];
        } else {
            int c = a + b >> 1;
            i64 ans = 0;
            pushdown(t, a, b);
            if(l <= c) ans += query(lc(t), a, c, l, r);
            if(r >  c) ans += query(rc(t), c + 1, b, l, r);
            pushup(t, a, b);
            return ans;
        }
    }
}

// 区间加、区间查询最大 & 最小值
namespace Seg2{
    #define lc(t) (t << 1)
    #define rc(t) (t << 1 | 1)
    const int SIZ = 4e5 + 3;
    i64 M[SIZ], N[SIZ], T[SIZ];

    void pushup(int t, int a, int b){
        M[t] = max(M[lc(t)], M[rc(t)]);
        N[t] = min(N[lc(t)], N[rc(t)]);
    }
    void pushdown(int t, int a, int b){
        if(T[t]){
            int l = lc(t), r = rc(t);
            int c = a + b >> 1;
            M[l] += T[t], N[l] += T[t];
            M[r] += T[t], N[r] += T[t];
            T[l] = (T[l] + T[t]);
            T[r] = (T[r] + T[t]);
            T[t] = 0;
        }
    }

    void modify(int t, int a, int b, int l, int r, int w){
        if(l <= a && b <= r){
            T[t] = T[t] + w;
            M[t] += w;
            N[t] += w;
        } else {
            int c = a + b >> 1;
            pushdown(t, a, b);
            if(l <= c) modify(lc(t), a, c, l, r, w);
            if(r >  c) modify(rc(t), c + 1, b, l, r, w);
            pushup(t, a, b);
        }
    }

    int query1(int t, int a, int b, int l, int r){
        if(l <= a && b <= r){
            return M[t];
        } else {
            int c = a + b >> 1;
            int ans = -INF;
            pushdown(t, a, b);
            if(l <= c) ans = max(ans, query1(lc(t), a, c, l, r));
            if(r >  c) ans = max(ans, query1(rc(t), c + 1, b, l, r));
            pushup(t, a, b);
            return ans;
        }
    }
    int query2(int t, int a, int b, int l, int r){
        if(l <= a && b <= r){
            return N[t];
        } else {
            int c = a + b >> 1;
            int ans = INF;
            pushdown(t, a, b);
            if(l <= c) ans = min(ans, query2(lc(t), a, c, l, r));
            if(r >  c) ans = min(ans, query2(rc(t), c + 1, b, l, r));
            pushup(t, a, b);
            return ans;
        }
    }
}

// ===== TEST =====

int qread();

int main(){
    int n = qread(), m = qread();
    for(int i = 1;i <= n;++ i){
        int a = qread();
        Seg1 :: modify(1, 1, n, i, i, a);
    }
    for(int i = 1;i <= m;++ i){
        int op = qread();
        if(op == 1){
            int l = qread(), r = qread();
            int w = qread();
            Seg1 :: modify(1, 1, n, l, r, w);
        } else{
            int l = qread(), r = qread();
            printf("%lld\n", Seg1 :: query(1, 1, n, l, r));
        }
    }
    return 0;
}