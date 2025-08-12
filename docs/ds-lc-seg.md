```cpp
#include "../../header.cpp"
struct Line{ int id; double k, b; Line() = default;};
namespace LCSeg{
  const int SIZ = 2e5 + 3;
  struct Line T[SIZ];
  #define lc(t) (t << 1)
  #define rc(t) (t << 1 | 1)
  bool cmp(int p, Line x, Line y){
    double w1 = x.k * p + x.b;
    double w2 = y.k * p + y.b;
    double d = w1 - w2;
    if(fabs(d) < 1e-8) return x.id > y.id;
    return d < 0;
  }
  void merge(int t, int a, int b, Line x, Line y){
    int c = a + b >> 1;
    if(cmp(c, x, y)) swap(x, y);
    if(cmp(a, y, x)){
      T[t] = x; if(a != b) merge(rc(t), c + 1, b, T[rc(t)], y);
    } else {
      T[t] = x; if(a != b) merge(lc(t), a,   c, T[lc(t)], y);
    }
  }
  // 插入线段 (l, f(l)) -- (r, f(r))
  void modify(int t, int a, int b, int l, int r, Line x){
    if(l <= a && b <= r) merge(t, a, b, T[t], x);
    else {
      int c = a + b >> 1;
      if(l <= c) modify(lc(t), a,   c, l, r, x);
      if(r >  c) modify(rc(t), c + 1, b, l, r, x);
    }
  }
  // 查询 x = p 位置最高的线段（有多条取编号最小）
  void query(int t, int a, int b, int p, Line &x){
    if(cmp(p, x, T[t])) x = T[t];
    if(a != b){
      int c = a + b >> 1;
      if(p <= c) query(lc(t), a,   c, p, x);
      if(p >  c) query(rc(t), c + 1, b, p, x);
    }
  }
}
```
