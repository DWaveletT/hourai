/**
## 用法

- `add(u, v, l, r, c)`：连一条容量在 $[l, r]$ 的从 $u$ 到 $v$ 的费用为 $c$ 的边；
- `solve()`：计算无源汇最小费用可行流；
- `solve(s, t)`：计算有源汇最小费用最大流。
**/
#define add add0
#include "flow-cost.cpp"
#undef add
namespace MCMF{
  i64 cost0;
  int G[MAXN];
  void add(int u, int v, int l, int r, int c){
    G[v] += l;
    G[u] -= l;
    cost0 += 1ll * l * c;
    add0(u, v, r - l, c);
  }
  i64 solve(){
    int s = ++ n;
    int t = ++ n;
    i64 sum = 0;
    for(int i = 1;i <= n - 2;++ i){
      if(G[i] < 0)
        add0(i, t, -G[i], 0);
      else
        add0(s, i,  G[i], 0), sum += G[i];
    }
    auto res = mcmf(s, t);
    if(res.first != sum)
      return -1;
    return res.second + cost0;
  }
  i64 solve(int s0, int t0){
    add0(t0, s0, INF, 0);
    int s = ++ n;
    int t = ++ n;
    i64 sum = 0;
    for(int i = 1;i <= n - 2;++ i){
      if(G[i] < 0)
        add0(i, t, -G[i], 0);
      else
        add0(s, i,  G[i], 0), sum += G[i];
    }
    auto res = mcmf(s, t);
    if(res.first != sum)
      return -1;
    return res.second + cost0;
  }
}