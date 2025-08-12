```cpp
#include "../header.cpp"
namespace LeftHeap{
  const int SIZ = 1e5 + 3;
  int W[SIZ], D[SIZ], L[SIZ], R[SIZ], F[SIZ], s;
  bool E[SIZ];
  int merge(int u, int v){
    if(u == 0 || v == 0)
      return u | v;
    if(W[u] > W[v] || (W[u] == W[v] && u > v))
      swap(u, v);
    int &lc = L[u];
    int &rc = R[u];
    rc = merge(rc, v);
    if(D[lc] < D[rc])
      swap(lc, rc);
    D[u] = min(D[lc], D[rc]) + 1;
    if(lc != 0) F[lc] = u;
    if(rc != 0) F[rc] = u;
    return u;
  }
  void pop(int &root){
    int root0 = merge(L[root], R[root]);
    F[root0] = root0;
    F[root ] = root0;
    E[root ] = true;
    root = root0;
  }
  int top(int &root){
    return W[root];
  }
  int getfa(int u){
    return u == F[u] ? u : F[u] = getfa(F[u]);
  }
  int newnode(int w){
    ++ s;
    W[s] = w;
    F[s] = s;
    D[s] = 1;
    return s;
  }
}

```
