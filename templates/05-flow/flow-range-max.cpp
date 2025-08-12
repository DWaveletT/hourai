/**
## 用法

- `add(u, v, l, r, c)`：连一条容量在 $[l, r]$ 的从 $u$ 到 $v$ 的边；
- `solve()`：检查是否存在无源汇可行流；
- `solve(s, t)`：计算有源汇最大流。
**/
#define add add0
#include "flow-max.cpp"
#undef add
namespace Dinic{
  int G[MAXN];
  void add(int u, int v, int l, int r){
    G[v] += l;
    G[u] -= l;
    add0(u, v, r - l);
  }
  void clear(){
    for(int i = 1;i <= t;++ i){
      N[i] = F[i] = V[i] = 0;
    }
    for(int i = 1;i <= n;++ i){
      H[i] = G[i] = C[i] = 0;
    }
    t = 1, n = 0;
  }
  bool solve(){
    int s = ++ n;
    int t = ++ n;
    i64 sum = 0;
    for(int i = 1;i <= n - 2;++ i){
      if(G[i] < 0)
        add0(i, t, -G[i]);
      else
        add0(s, i,  G[i]), sum += G[i];
    }
    auto res = dinic(s, t);
    if(res != sum)
      return true;
    return false;
  }
  i64 solve(int s0, int t0){
    add0(t0, s0, INF);
    int s = ++ n;
    int t = ++ n;
    i64 sum = 0;
    for(int i = 1;i <= n - 2;++ i){
      if(G[i] < 0)
        add0(i, t, -G[i]);
      else
        add0(s, i,  G[i]), sum += G[i];
    }
    auto res = dinic(s, t);
    if(res != sum)
      return -1;
    return dinic(s0, t0);
  }
}
