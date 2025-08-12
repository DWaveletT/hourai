```cpp
#include "../header.cpp"
int n, m, root, MOD, A[MAXN];
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
    else        T[u] = T[f];
  if(G[u]) dfs2(G[u], u);
  for(auto &v : E[u]) if(v != f && v != G[u]){
    dfs2(v, u);
  }
  R[u] = cnt;
}
namespace Seg{
  const int SIZ = 4e5 + 3;
  i64 S[SIZ], T[SIZ];
  void pushup(int t, int a, int b);
  void pushdown(int t, int a, int b);
  void modify(int t, int a, int b, int l, int r, int w);
  i64 query(int t, int a, int b, int l, int r);
  void build(int t, int a, int b);
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
```
