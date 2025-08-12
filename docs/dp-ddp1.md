## 例题

给定一棵 $n$ 个点的树，点有点权，求最大独立集。$m$ 次修改，每次把 $x$ 的权值修改成 $y$。

```cpp
#include "../header.cpp"
int W[MAXN];
struct Mat{ int M[2][2]; };
struct Vec{ int V[2];  };
Mat operator *(const Mat &a, const Mat &b){
  Mat c;
  c.M[0][0] = max(a.M[0][0] + b.M[0][0], a.M[0][1] + b.M[1][0]);
  c.M[0][1] = max(a.M[0][0] + b.M[0][1], a.M[0][1] + b.M[1][1]);
  c.M[1][0] = max(a.M[1][0] + b.M[0][0], a.M[1][1] + b.M[1][0]);
  c.M[1][1] = max(a.M[1][0] + b.M[0][1], a.M[1][1] + b.M[1][1]);
  return c;
}
Vec operator *(const Mat &a, const Vec &v){
  Vec r;
  r.V[0] = max(a.M[0][0] + v.V[0], a.M[0][1] + v.V[1]);
  r.V[1] = max(a.M[1][0] + v.V[0], a.M[1][1] + v.V[1]);
  return r;
}
namespace Gra{
  vector<int> E[MAXN];
  int G[MAXN], S[MAXN], D[MAXN], T[MAXN], F[MAXN];
  int X[MAXN], Y[MAXN];
  int H[MAXN][2];
  int K[MAXN][2];
  struct Mat M[MAXN];
  void dfs1(int u, int f){
    S[u] = 1;
    F[u] = f;
    for(auto &v : E[u]) if(v != f){
      dfs1(v, u);
      S[u] += S[v];
      if(S[v] > S[G[u]]) G[u] = v;
    }
  }
  int o;
  void dfs2(int u, int f){
    if(u == G[f])
      X[u] = X[f];
    else
      X[u] = u;
    H[u][0] = H[u][1] = 0;
    K[u][0] = K[u][1] = 0;
    const int &g = G[u];
    D[u] = ++ o;
    T[o] = u;
    if(g){
      dfs2(g, u);
      Y[u] = Y[g];
      K[u][0] += max(K[g][0], K[g][1]);
      K[u][1] += K[g][0];
    } else {
      Y[u] = u;
    }
    for(auto &v : E[u]) if(v != f && v != g){
      dfs2(v, u);
      H[u][0] += max(K[v][0], K[v][1]);
      H[u][1] += K[v][0];
    }
    M[u].M[0][0] = H[u][0];
    M[u].M[0][1] = H[u][0];
    M[u].M[1][0] = H[u][1] + W[u];
    M[u].M[1][1] = -INF;
    K[u][0] += H[u][0];
    K[u][1] += H[u][1] + W[u];
  }
}
namespace Seg{
  const int SIZ = 4e5 + 3;
  struct Mat M[SIZ];
  #define lc(t) (t << 1)
  #define rc(t) (t << 1 | 1)
  void pushup(int t, int a, int b){
    M[t] = M[lc(t)] * M[rc(t)];
  }
  void build(int t, int a, int b){
    if(a == b){
      M[t] = Gra :: M[Gra :: T[a]];
    } else {
      int c = a + b >> 1;
      build(lc(t), a, c);
      build(rc(t), c + 1, b);
      pushup(t, a, b);
    }
  }
  void modify(int t, int a, int b, int p, const Mat &w){
    if(a == b){
      M[t] = w;
    } else {
      int c = a + b >> 1;
      if(p <= c) modify(lc(t), a, c, p, w);
        else   modify(rc(t), c + 1, b, p, w);
      pushup(t, a, b);
    }
  }
  Mat query(int t, int a, int b, int l, int r){
    if(l <= a && b <= r){
      return M[t];
    } else {
      int c = a + b >> 1;
      if(r <= c) return query(lc(t), a, c  , l, r); else 
      if(l >  c) return query(rc(t), c + 1, b, l, r); else 
        return query(lc(t), a, c  , l, r) *
             query(rc(t), c + 1, b, l, r);
    }
  }
}
int qread();
int main(){
  int n = qread(), m = qread();
  up(1, n, i)
    W[i] = qread();
  up(2, n, i){
    int u = qread(), v = qread();
    Gra :: E[u].push_back(v);
    Gra :: E[v].push_back(u);
  }
  Gra :: dfs1(1, 0);
  Gra :: dfs2(1, 0);
  Seg :: build(1, 1, n);
  Vec v0;
  v0.V[0] = v0.V[1] = 0;
  up(1, m, i){
    using namespace Gra;
    int x = qread(), y = qread();
    W[x] = y;
    int u = x;
    while(u != 0){
      const int &v = X[u];
      const int &f = F[v];
      M[u].M[0][0] = H[u][0];
      M[u].M[0][1] = H[u][0];
      M[u].M[1][0] = H[u][1] + W[u];
      M[u].M[1][1] = -INF;
      const Vec p = Seg :: query(1, 1, n, D[v], D[Y[u]]) * v0;
      Seg :: modify(1, 1, n, D[u], M[u]);
      const Vec q = Seg :: query(1, 1, n, D[v], D[Y[u]]) * v0;
      if(f != 0){
        H[f][0] = H[f][0] - max(p.V[0], p.V[1]) + max(q.V[0], q.V[1]);
        H[f][1] = H[f][1] - p.V[0] + q.V[0];
      }
      u = f;
    }
    Vec v1 = Seg :: query(1, 1, n, D[1], D[Y[1]]) * v0;
    printf("%d\n", max(v1.V[0], v1.V[1]));
  }
  return 0;
}
```
