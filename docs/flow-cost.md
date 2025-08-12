```cpp
#include "../header.cpp"
namespace MCMF{
  int H[MAXN], V[MAXM], N[MAXM], W[MAXM], F[MAXM], o = 1, n;
  void add(int u, int v, int f, int c){
    V[++ o] = v, N[o] = H[u], H[u] = o, F[o] = f, W[o] =  c;
    V[++ o] = u, N[o] = H[v], H[v] = o, F[o] = 0, W[o] = -c;
    n = max(n, u);
    n = max(n, v);
  }
  void clear(){
    for(int i = 1;i <= n;++ i)
      H[i] = 0;
    n = 0, o = 1;
  }
  bool I[MAXN];
  i64 D[MAXN];
  bool spfa(int s, int t){
    queue <int> Q;
    Q.push(s), I[s] = true;
    for(int i = 1;i <= n;++ i)
      D[i] = INFL;
    D[s] = 0;
    while(!Q.empty()){
      int u = Q.front(); Q.pop(), I[u] = false;
      for(int i = H[u];i;i = N[i]){
        const int &v = V[i];
        const int &f = F[i];
        const int &w = W[i];
        if(f && D[u] + w < D[v]){
          D[v] = D[u] + w;
          if(!I[v]) Q.push(v), I[v] = true;
        }
      }
    }
    return D[t] != INFL;
  }
  int C[MAXN]; bool T[MAXN];
  pair<i64, i64> dfs(int s, int t, int u, i64 maxf){
    if(u == t)
      return make_pair(maxf, 0);
    i64 totf = 0;
    i64 totc = 0;
    T[u] = true;
    for(int &i = C[u];i;i = N[i]){
      const int &v = V[i];
      const int &f = F[i];
      const int &w = W[i];
      if(f && D[v] == D[u] + w && !T[v]){
        auto p = dfs(s, t, v, min(1ll * F[i], maxf));
        i64 f = p.first;
        i64 c = p.second;
        F[i  ] -= f;
        F[i ^ 1] += f;
        totf += f;
        totc += 1ll * f * W[i] + c;
        maxf -= f;
        if(maxf == 0){
          T[u] = false;
          return make_pair(totf, totc);
        }
      }
    }
    T[u] = false;
    return make_pair(totf, totc);
  }
  pair<i64, i64> mcmf(int s, int t){
    i64 ans1 = 0;
    i64 ans2 = 0;
    pair<i64, i64> r;
    while(spfa(s, t)){
      memcpy(C, H, sizeof(int) * (n + 3));
      r = dfs(s, t, s, INFL);
      ans1 += r.first;
      ans2 += r.second;
    }
    return make_pair(ans1, ans2);
  }
}
int qread();
int main(){
  int n = qread(), m = qread(), s = qread(), t = qread();
  for(int i = 1;i <= m;++ i){
    int u = qread(), v = qread(), f = qread(), c = qread();
    MCMF :: add(u, v, f, c);
  }
  pair<long long, long long> ans = MCMF :: mcmf(s, t);
  printf("%lld %lld\n", ans.first, ans.second);
  return 0;
}
```
