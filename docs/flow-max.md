```cpp
#include "../header.cpp"
namespace Dinic{
  const i64 INF = 1e18;
  const int SIZ = 5e5 + 3;
  int n;
  int H[MAXN], V[MAXM], N[MAXM], F[MAXM], t = 1;
  void add(int u, int v, int f){
    V[++ t] = v, N[t] = H[u], F[t] = f, H[u] = t;
    V[++ t] = u, N[t] = H[v], F[t] = 0, H[v] = t;
    n = max(n, u);
    n = max(n, v);
  }
  void clear(){
    for(int i = 1;i <= n;++ i)
      H[i] = 0;
    n = 0, t = 1;
  }
  i64 D[MAXN];
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
  int C[MAXN];
  i64 dfs(int s, int t, int u, i64 maxf){
    if(u == t)
      return maxf;
    i64 totf = 0;
    for(int &i = C[u];i;i = N[i]){
      const int &v = V[i];
      const int &f = F[i];
      if(f && D[v] == D[u] + 1){
        i64 f = dfs(s, t, v, min(1ll * f, maxf));
        F[i] -= f, F[i ^ 1] += f, totf += f, maxf -= f;
        if(maxf == 0)
          return totf;
      }
    }
    return totf;
  }
  i64 dinic(int s, int t){
    i64 ans = 0;
    while(bfs(s, t)){
      memcpy(C, H, sizeof(int) * (n + 3));
      ans += dfs(s, t, s, INFL);
    }
    return ans;
  }
}

```
