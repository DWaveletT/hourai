## 例题

给定 $n$ 个点组成的树，点有点权 $v_i$。$m$ 个操作，分为两种：

- `0 x k` 查询距离 $x$ 不超过 $k$ 的所有点的点权之和；
- `0 x y` 将点 $x$ 的点权修改为 $y$。

```cpp
#include<bits/stdc++.h>
#define endl "\n"
using namespace std;

const int MAXN = 1e5 + 3;
vector<int> E[MAXN];

namespace LCA{
    const int SIZ = 1e5 + 3;
    int D[SIZ], F[SIZ];
    int P[SIZ], Q[SIZ], o;
    void dfs(int u, int f){
        P[u] = ++ o;
        Q[o] = u;
        F[u] = f;
        D[u] = D[f] + 1;
        for(auto &v : E[u]) if(v != f){
            dfs(v, u);
        }
    }
    const int MAXH = 18 + 3;
    int h = 18;
    int ST[SIZ][MAXH];
    int cmp(int a, int b){
        return D[a] < D[b] ? a : b;
    }
    int T[SIZ], n;
    void init(int _n){
        n = _n;
        dfs(1, 0);
        for(int i = 1;i <= n;++ i)
            ST[i][0] = Q[i];
        for(int i = 2;i <= n;++ i)
            T[i] = T[i >> 1] + 1;
        for(int i = 1;i <= h;++ i){
            for(int j = 1;j <= n;++ j) if(j + (1 << i - 1) <= n){
                ST[j][i] = cmp(ST[j][i - 1], ST[j + (1 << i - 1)][i - 1]);
            }
        }
    }
    int lca(int a, int b){
        if(a == b)
            return a;
        int l = P[a];
        int r = P[b];
        if(l > r)
            swap(l, r);
        ++ l;
        int d = T[r - l + 1];
        return F[cmp(ST[l][d], ST[r - (1 << d) + 1][d])];
    }
    int dis(int a, int b){
        return D[a] + D[b] - 2 * D[lca(a, b)];
    }
}

namespace BIT{
    void modify(int D[], int n, int p, int w){
        ++ p;
        while(p <= n)
            D[p] += w, p += p & -p;
    }
    int query(int D[], int n, int p){
        if(p < 0) return 0;
        p = min(n, p + 1);
        int r = 0;
        while(p >  0)
            r += D[p], p -= p & -p;
        return r;
    }
}

namespace PTree{
    const int SIZ = 1e5 + 3;
    bool V[SIZ];
    int  S[SIZ], L[SIZ];
    vector<int> EE[MAXN];
    int *D1[MAXN];
    int *D2[MAXN];

    void dfs1(int s, int &g, int u, int f){
        S[u] = 1;
        int maxsize = 0;
        for(auto &v : E[u]) if(v != f && !V[v]){
            dfs1(s, g, v, u);
            if(S[v] > maxsize)
                maxsize = S[v];
            S[u] += S[v];
        }
        maxsize = max(maxsize, s - S[u]);
        if(maxsize <= s / 2)
            g = u;
    }

    int n;
    void build(int s, int &g, int u, int f){
        dfs1(s, g, u, f);
        V[g] = true, L[g] = s;
        for(auto &u : E[g]) if(!V[u]){
            int h = 0;
            if(S[u] < S[g]) build(S[u], h, u, 0);
            else            build(s - S[g], h, u, 0);
            EE[g].push_back(h);
            EE[h].push_back(g);
        }
    }
    int F[SIZ];
    void dfs2(int u, int f){
        F[u] = f;
        for(auto &v : EE[u]) if(v != f){
            dfs2(v, u);
        }
    }
    void build(int _n){
        n = _n;
        int s = n, g = 0;
        dfs1(s, g, 1, 0);
        V[g] = true, L[g] = s;
        for(auto &u : E[g]){
            int h = 0;
            if(S[u] < S[g]) build(S[u], h, u, 0);
            else            build(s - S[g], h, u, 0);
            EE[g].push_back(h);
            EE[h].push_back(g);
        }
        dfs2(g, 0);
        for(int i = 1;i <= n;++ i){
            L[i] += 2;
            D1[i] = new int[L[i] + 3];
            D2[i] = new int[L[i] + 3];
            for(int j = 0;j < L[i] + 3;++ j)
                D1[i][j] = D2[i][j] = 0;
        }
    }
    void modify(int x, int w){
        int u = x;
        while(1){
            BIT :: modify(D1[x], L[x], LCA :: dis(u, x), w);
            int y = F[x];
            if(y != 0){
                int e = LCA :: dis(x, y);
                BIT :: modify(D2[x], L[x], LCA :: dis(u, y), w);
                x = y;
            } else break;
        }
    }
    int query(int x, int d){
        int ans = 0, u = x;
        while(1){
            ans += BIT :: query(D1[x], L[x], d - LCA :: dis(u, x));
            int y = F[x];
            if(y != 0){
                int e = LCA :: dis(x, y);
                ans -= BIT :: query(D2[x], L[x], d - LCA :: dis(u, y));
                x = y;
            } else break;
        }
        return ans;
    }
}

int W[MAXN];

int main(){
    ios :: sync_with_stdio(false);
    int n, m;
    cin >> n >> m;
    for(int i = 1;i <= n;++ i){
        cin >> W[i];
    }
    for(int i = 2;i <= n;++ i){
        int u, v;
        cin >> u >> v;
        E[u].push_back(v);
        E[v].push_back(u);
    }
    LCA :: init(n);

    PTree :: build(n);

    for(int i = 1;i <= n;++ i)
        PTree :: modify(i, W[i]);

    int lastans = 0;
    for(int i = 1;i <= m;++ i){
        int op; cin >> op;
        if(op == 0){
            int x, d;
            cin >> x >> d;
            x ^= lastans;
            d ^= lastans;
            cout << (lastans = PTree :: query(x, d)) << endl;
        } else {
            int x, w;
            cin >> x >> w;
            x ^= lastans;
            w ^= lastans;
            PTree :: modify(x, -W[x]    );
            PTree :: modify(x,  W[x] = w);
        }
    }
    return 0;
}
```
