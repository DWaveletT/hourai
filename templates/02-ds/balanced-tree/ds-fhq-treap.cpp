#include "../../header.cpp"
mt19937_64 MT(114514);
namespace Treap{
  const int SIZ = 1e6 + 1e5 + 3;
  int F[SIZ], C[SIZ], S[SIZ], W[SIZ], X[SIZ][2], sz;
  u64 H[SIZ];
  int  newnode(int w){
    W[++ sz] = w, C[sz] = S[sz] = 1; H[sz] = MT();
    return sz;
  }
  void pushup(int x){
    S[x] = C[x] + S[X[x][0]] + S[X[x][1]];
  }
  pair<int, int> split(int u, int x){
    if(u == 0)
      return make_pair(0, 0);
    if(W[u] > x){
      auto [a, b] = split(X[u][0], x);
      X[u][0] = b, pushup(u);
      return make_pair(a, u);
    } else {
      auto [a, b] = split(X[u][1], x);
      X[u][1] = a, pushup(u);
      return make_pair(u, b);
    }
  }
  int merge(int a, int b){
    if(a == 0 || b == 0)
      return a | b;
    if(H[a] < H[b]){
      X[a][1] = merge(X[a][1], b), pushup(a);
      return a;
    } else {
      X[b][0] = merge(a, X[b][0]), pushup(b);
      return b;
    }
  }
  void insert(int &root, int w){
    auto [p, q] = split(root, w);
    auto [a, b] = split(p, w - 1);
    if(b != 0){
      ++ S[b], ++ C[b];
    } else b = newnode(w);
    p = merge(a, b), root = merge(p, q);
  }
  void erase(int &root, int w){
    auto [p, q] = split(root, w);
    auto [a, b] = split(p, w - 1);
    -- C[b], -- S[b];
    p  = C[b] == 0 ? a : merge(a, b);
    root = merge(p, q);
  }
  int  find_rank(int &root, int w){
    int x = root, o = x, a = 0;
    for(;x;){
      if(w <  W[x])
        o = x, x = X[x][0];
      else {
        a += S[X[x][0]];
        if(w == W[x]){ o = x; break; }
        a += C[x];
        o = x, x = X[x][1];
      }
    }
    return a + 1;
  }
  int  find_kth(int &root, int w){
    int x = root, o = x, a = 0;
    for(;x;){
      if(w <= S[X[x][0]])
        o = x, x = X[x][0];
      else {
        w -= S[X[x][0]];
        if(w <= C[x]){ o = x; break; }
        w -= C[x];
        o = x, x = X[x][1];
      } 
    }
    return W[x];
  }
  int  find_pre(int &root, int w){
    return find_kth(root, find_rank(root, w) - 1);
  }
  int  find_suc(int &root, int w){
    return find_kth(root, find_rank(root, w + 1));
  }
}