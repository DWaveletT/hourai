#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXN= 1e5 + 3;

int MOD;

int n, m, root;
int A[MAXN];
int qread();
vector <int> E[MAXN];
int S[MAXN], G[MAXN], D[MAXN], F[MAXN];
void dfs1(int u, int f){
    S[u] = 1, G[u] = 0, D[u] = D[f] + 1, F[u] = f;
    for(auto &v : E[u]) if(v != f){
        dfs1(v, u);
        S[u] += S[v];
        if(S[v] > S[G[u]])
            G[u] = v;
    }
}
int B[MAXN];
int P[MAXN], Q[MAXN], T[MAXN], L[MAXN], R[MAXN], cnt;
void dfs2(int u, int f){
    P[++ cnt] = u, B[cnt] = A[u], Q[u] = cnt;
    L[u] = cnt;
    if(u != G[f]) T[u] = u;
        else      T[u] = T[f];
    if(G[u]) dfs2(G[u], u);
    for(auto &v : E[u]) if(v != f && v != G[u]){
        dfs2(v, u);
    }
    R[u] = cnt;
}

namespace Seg{
    #define lc(t) (t << 1)
    #define rc(t) (t << 1 | 1)
    const int SIZ = 4e5 + 3;
    i64 S[SIZ], T[SIZ];
    void pushup(int t, int a, int b){
        S[t] = (S[lc(t)] + S[rc(t)]) % MOD;
    }
    void pushdown(int t, int a, int b){
        if(T[t]){
            int c = a + b >> 1;
            T[lc(t)] = (T[lc(t)] + T[t]) % MOD;
            T[rc(t)] = (T[rc(t)] + T[t]) % MOD;
            S[lc(t)] = (S[lc(t)] + 1ull * (c - a + 1) * T[t]) % MOD;
            S[rc(t)] = (S[rc(t)] + 1ull * (b - c    ) * T[t]) % MOD;
            T[t] = 0;
        }
    }
    void modify(int t, int a, int b, int l, int r, int w){
        if(l <= a && b <= r){
            S[t] = (S[t] + 1ll * w * (b - a + 1)) % MOD;
            T[t] = (T[t] + w) % MOD;
        } else {
            int c = a + b >> 1;
            pushdown(t, a, b);
            if(l <= c) modify(lc(t), a, c, l, r, w);
            if(r >  c) modify(rc(t), c + 1, b, l, r, w);
            pushup(t, a, b);
        }
    }
    i64 query(int t, int a, int b, int l, int r){
        if(l <= a && b <= r)
            return S[t];
        int c = a + b >> 1;
        i64 ans = 0;
        pushdown(t, a, b);
        if(l <= c) ans = (ans + query(lc(t), a, c, l, r)) % MOD;
        if(r >  c) ans = (ans + query(rc(t), c + 1, b, l, r)) % MOD;
        return ans;
    }
    void build(int t, int a, int b){
        if(a == b){
            S[t] = B[a] % MOD;
        } else {
            int c = a + b >> 1;
            build(lc(t), a, c);
            build(rc(t), c + 1, b);
            pushup(t, a, b);
        }
    }
}
int main(){
    n = qread(), m = qread(), root = qread(), MOD = qread();
    for(int i = 1;i <= n;++ i)
        A[i] = qread();
    for(int i = 2;i <= n;++ i){
        int u = qread(), v = qread();
        E[u].push_back(v);
        E[v].push_back(u);
    }
    dfs1(root, 0);
    dfs2(root, 0);
    Seg :: build(1, 1, n);
    for(int i = 1;i <= m;++ i){
        int op = qread();
        if(op == 1){
            int u = qread(), v = qread(), k = qread();
            while(T[u] != T[v]){
                if(D[T[u]] < D[T[v]])
                    swap(u, v);
                Seg :: modify(1, 1, n, Q[T[u]], Q[u], k);
                u = F[T[u]];
            }
            if(D[u] < D[v]) swap(u, v);
            Seg :: modify(1, 1, n, Q[v], Q[u], k);
        } else if(op == 2){
            int u = qread(), v = qread();
            i64 ans = 0;
            while(T[u] != T[v]){
                if(D[T[u]] < D[T[v]])
                    swap(u, v);
                ans = (ans + Seg :: query(1, 1, n, Q[T[u]], Q[u])) % MOD;
                u = F[T[u]];
            }
            if(D[u] < D[v]) swap(u, v);
            ans = (ans + Seg :: query(1, 1, n, Q[v], Q[u])) % MOD;
            printf("%lld\n", ans);
        } else if(op == 3){
            int x = qread(), w = qread();
            Seg :: modify(1, 1, n, L[x], R[x], w);
        } else {
            int x = qread();
            printf("%lld\n", Seg :: query(1, 1, n, L[x], R[x]));
        }
    }
    return 0;
}