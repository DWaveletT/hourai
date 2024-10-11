```cpp
#include <bits/stdc++.h>
using namespace std;

int qread(){
    int w = 1, c, ret;
    while((c = getchar()) >  '9' || c <  '0') w = (c == '-' ? -1 : 1); ret = c - '0';
    while((c = getchar()) >= '0' && c <= '9') ret = ret * 10 + c - '0';
    return ret * w;
}

namespace Dinic{
    const long long INF = 1e18;
    const int SIZ = 1e5 + 3;
    int n, m;
    int H[SIZ], V[SIZ], N[SIZ], F[SIZ], t = 1;
    int add(int u, int v, int f){
        V[++ t] = v, N[t] = H[u], F[t] = f, H[u] = t;
        V[++ t] = u, N[t] = H[v], F[t] = 0, H[v] = t;
        n = max(n, u);
        n = max(n, v);
        return t - 1;
    }
    void clear(){
        for(int i = 1;i <= n;++ i)
            H[i] = 0;
        n = m = 0, t = 1;
    }
    int D[SIZ];
    bool bfs(int s, int t){
        queue <int> Q;
        for(int i = 1;i <= n;++ i)
            D[i] = 0;
        Q.push(s), D[s] = 1;
        while(!Q.empty()){
            int u = Q.front(); Q.pop();
            for(int i = H[u];i;i = N[i]){
                const int &v = V[i];
                const int &f = F[i];
                if(f != 0 && !D[v]){
                    D[v] = D[u] + 1;
                    Q.push(v);
                }
            }
        }
        return D[t] != 0;
    }
    int C[SIZ];
    long long dfs(int s, int t, int u, long long maxf){
        if(u == t)
            return maxf;
        long long totf = 0;
        for(int &i = C[u];i;i = N[i]){
            const int &v = V[i];
            const int &f = F[i];
            if(D[v] == D[u] + 1){
                long long resf = dfs(s, t, v, min(maxf, 1ll * f));
                totf += resf;
                maxf -= resf;
                F[i    ] -= resf;
                F[i ^ 1] += resf;
                if(maxf == 0)
                    return totf;
            }
        }
        return totf;
    }
    long long dinic(int s, int t){
        long long ans = 0;
        while(bfs(s, t)){
            memcpy(C, H, sizeof(H));
            ans += dfs(s, t, s, INF);
        }
        return ans;
    }
}

namespace GHTree{
    const int MAXN =  500 + 5;
    const int MAXM = 1500 + 5;
    const int INF  = 1e9;
    int n, m, U[MAXM], V[MAXM], W[MAXM], A[MAXM], B[MAXM];
    void add(int u, int v, int w){
        ++ m;
        U[m] = u;
        V[m] = v;
        W[m] = w;
        A[m] = Dinic :: add(u, v, w);
        B[m] = Dinic :: add(v, u, w);
        n = max(n, u);
        n = max(n, v);
    }
    vector <pair<int, int> > E[MAXN];
    void build(vector <int> N){
        int s = N.front();
        int t = N.back();
        if(s == t) return;
        for(int i = 1;i <= m;++ i){
            int a = A[i]; Dinic :: F[a] = W[i], Dinic :: F[a ^ 1] = 0;
            int b = B[i]; Dinic :: F[b] = W[i], Dinic :: F[b ^ 1] = 0;
        }
        int w = Dinic :: dinic(s, t);
        E[s].push_back(make_pair(t, w));
        E[t].push_back(make_pair(s, w));
        
        vector <int> P;
        vector <int> Q;
        for(auto &u : N){
            if(Dinic :: D[u] != 0)
                P.push_back(u);
            else
                Q.push_back(u);
        }
        build(P), build(Q);
    }
    int D[MAXN];
    int cut(int s, int t){
        queue <int> Q; Q.push(s);
        for(int i = 1;i <= n;++ i)
            D[i] = -1;
        D[s] = INF;
        while(!Q.empty()){
            int u = Q.front(); Q.pop();
            for(auto &e : E[u]){
                int v = e.first;
                int w = e.second;
                if(D[v] == -1){
                    D[v] = min(D[u], w);
                    Q.push(v);
                }
            }
        }
        return D[t];
    }
}

```
