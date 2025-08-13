/**
## 例题

给定 $n$ 个点组成的树，点有点权 $v_i$。$m$ 个操作，分为两种：

- `0 x k` 查询距离 $x$ 不超过 $k$ 的所有点的点权之和；
- `0 x y` 将点 $x$ 的点权修改为 $y$。
**/
#include "../header.cpp"
vector<int> E[MAXN];
namespace LCA{
  const int MAXH = 18 + 3;
  int D[MAXN], F[MAXN];
  int P[MAXN], Q[MAXN], o, h = 18;
  void dfs(int u, int f){
    ++ o;
    P[u] = o, Q[o] = u;
    F[u] = f, D[u] = D[f] + 1;
    for(auto &v : E[u]) if(v != f)
      dfs(v, u);
  }
  int ST[MAXN][MAXH];
  int cmp(int a, int b){
    return D[a] < D[b] ? a : b;
  }
  int T[MAXN], n;
  void init(int _n);  // 初始化 ST 表
  int lca(int a, int b){
    if(a == b) return a;
    int l = P[a], r = P[b];
    if(l > r) swap(l, r);
    ++ l;
    int d = T[r - l + 1];
    return F[cmp(ST[l][d], ST[r - (1 << d) + 1][d])];
  }
  int dis(int a, int b);
}

namespace BIT{
  void add(int D[], int n, int p, int w){
    ++ p;
    while(p <= n) D[p] += w, p += p & -p;
  }
  int pre(int D[], int n, int p){
    if(p < 0) return 0;
    p = min(n, p + 1);
    int r = 0;
    while(p >  0) r += D[p], p -= p & -p;
    return r;
  }
}

namespace PTree{
  vector<int> EE[MAXN];
  bool V[MAXN];
  int S[MAXN], L[MAXN], *D1[MAXN], *D2[MAXN];
  using LCA :: dis, BIT :: add, BIT :: pre;
  void dfs1(int s, int &g, int u, int f){
    S[u] = 1;
    int maxsize = 0;
    for(auto &v : E[u]) if(v != f && !V[v]){
      dfs1(s, g, v, u);
      maxsize = max(maxsize, S[v]);
      S[u] += S[v];
    }
    maxsize = max(maxsize, s - S[u]);
    if(maxsize <= s / 2) g = u;
  }

  int n;
  void build(int s, int &g, int u, int f){
    dfs1(s, g, u, f);
    V[g] = true, L[g] = s;
    for(auto &u : E[g]) if(!V[u]){
      int h = 0;
      if(S[u] < S[g]) build(S[u], h, u, 0);
      else        build(s - S[g], h, u, 0);
      EE[g].push_back(h);
      EE[h].push_back(g);
    }
  }
  int F[MAXN];
  void dfs2(int u, int f){
    F[u] = f;
    for(auto &v : EE[u]) if(v != f){
      dfs2(v, u);
    }
  }
  void build(int _n){   // 建树（需初始化 LCA）
    n = _n;
    int s = n, g = 0;
    dfs1(s, g, 1, 0);
    V[g] = true, L[g] = s;
    for(auto &u : E[g]){
      int h = 0;
      if(S[u] < S[g]) build(S[u], h, u, 0);
      else build(s - S[g], h, u, 0);
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
  void modify(int x, int w){  // 修改点权
    int u = x;
    while(1){
      add(D1[x], L[x], dis(u, x), w);
      int y = F[x];
      if(y != 0){
        int e = LCA :: dis(x, y);
        add(D2[x], L[x], dis(u, y), w);
        x = y;
      } else break;
    }
  }
  int query(int x, int d){
    int ans = 0, u = x;
    while(1){
      ans += pre(D1[x], L[x], d - dis(u, x));
      int y = F[x];
      if(y != 0){
        int e = dis(x, y);
        ans-= pre(D2[x], L[x], d - dis(u, y));
        x = y;
      } else break;
    }
    return ans;
  }
}
