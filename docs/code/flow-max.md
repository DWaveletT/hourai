```cpp
#include <bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

namespace Dinic{
    const i64 INF = 1e18;
    const int SIZ = 5e5 + 3;
    int n, m;
    int H[SIZ], V[SIZ], N[SIZ], F[SIZ], t = 1;
    void add(int u, int v, int f){
        V[++ t] = v, N[t] = H[u], F[t] = f, H[u] = t;
        V[++ t] = u, N[t] = H[v], F[t] = 0, H[v] = t;
        n = max(n, u);
        n = max(n, v);
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
    i64 dfs(int s, int t, int u, i64 maxf){
        if(u == t)
            return maxf;
        i64 totf = 0;
        for(int &i = C[u];i;i = N[i]){
            const int &v = V[i];
            const int &f = F[i];
            if(D[v] == D[u] + 1){
                i64 resf = dfs(s, t, v, min(maxf, 1ll * f));
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
    i64 dinic(int s, int t){
        i64 ans = 0;
        while(bfs(s, t)){
            memcpy(C, H, sizeof(H));
            ans += dfs(s, t, s, INF);
        }
        return ans;
    }
}

// ===== TEST =====

int qread();

int main(){
    int n = qread(), m = qread(), s = qread(), t = qread();
    for(int i = 1;i <= m;++ i){
        int u = qread(), v = qread(), f = qread();
        Dinic :: add(u, v, f);
    }
    printf("%lld\n", Dinic :: dinic(s, t));
    return 0;
}
```
