#include<bits/stdc++.h>
using namespace std;

const int MAXN = 5e5 + 3;
vector<int> E[MAXN];

namespace LCA1{
    const int SIZ = 5e5 + 3;
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
    void init(int _n, int root){
        n = _n;
        dfs(root, 0);
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

namespace LCA2{
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
    int T[MAXN];
    void dfs2(int u, int f){
        if(u != G[f]) T[u] = u;
            else      T[u] = T[f];
        if(G[u]) dfs2(G[u], u);
        for(auto &v : E[u]) if(v != f && v != G[u]){
            dfs2(v, u);
        }
    }
    int lca(int u, int v){
        while(T[u] != T[v]){
            if(D[T[u]] > D[T[v]]) u = F[T[u]]; else v = F[T[v]];
        }
        return D[u] < D[v] ? u : v;
    }
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);

    int n, m, s;
    cin >> n >> m >> s;
    for(int i = 2;i <= n;++ i){
        int u, v;
        cin >> u >> v;
        E[u].push_back(v);
        E[v].push_back(u);
    }
    LCA2 :: dfs1(s, 0);
    LCA2 :: dfs2(s, 0);
    for(int i = 1;i <= m;++ i){
        int a, b;
        cin >> a >> b;
        cout << LCA2 :: lca(a, b) << "\n";
    }
    return 0;
}