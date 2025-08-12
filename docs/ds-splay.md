```cpp
#include "../../header.cpp"
namespace Splay{
  const int SIZ = 1e6 + 1e5 + 3;
  int F[SIZ], C[SIZ], S[SIZ], X[SIZ][2], size;
  bool T[SIZ];
  bool is_root(int x){ return   F[x]   == 0;}
  bool is_rson(int x){ return X[F[x]][1] == x;}
  void push_down(int x){
    if(!T[x]) return;
    int lc = X[x][0], rc = X[x][1];
    if(lc) T[lc] ^= 1, swap(X[lc][0], X[lc][1]);
    if(rc) T[rc] ^= 1, swap(X[rc][0], X[rc][1]);
    T[x] = 0;
  }
  void pushup(int x){
    S[x] = C[x] + S[X[x][0]] + S[X[x][1]];
  }
  void rotate(int x){
    int y = F[x], z = F[y];
    bool f = is_rson(x);
    bool g = is_rson(y);
    int &t = X[x][!f];
    if(z){ X[z][g] = x; }
    if(t){ F[t]  = y; }
    X[y][f] = t, t = y;
    F[y] = x, pushup(y);
    F[x] = z, pushup(x);
  }
  void splay(int &r, int x, int g = 0){
    for(int f;f = F[x], f != g;rotate(x))
      if(F[f] != g) rotate(is_rson(x) == is_rson(f) ? f : x);
    if(is_root(x)) r = x;
  }
  int  get_kth(int &r, int w){
    int x = r, o = x;
    for(;x;){
      push_down(x);
      if(w <= S[X[x][0]]) o = x, x = X[x][0]; else {
        w -= S[X[x][0]];
        if(C[x] && w <= C[x]){o = x; break;}
        w -= C[x], o = x, x = X[x][1];
      } 
    }
    splay(r, o); return o;
  }
  int  build(int l, int r){
    if(l == r){
      C[l] = S[l] = 1; return l;
    }
    int c = l + r >> 1, a = 0, b = 0;
    if(l <= c - 1) a = build(l, c - 1), F[a] = c, X[c][0] = a;
    if(c + 1 <= r) b = build(c + 1, r), F[b] = c, X[c][1] = b;
    C[c] = 1, pushup(c); return c;
  }
  void output(int n, int &r){
    push_down(r);
    if(X[r][0]) output(n, X[r][0]);
    if(r != 1 && r != n + 2) printf("%d ", r - 1);
    if(X[r][1]) output(n, X[r][1]);
  }
}
```
