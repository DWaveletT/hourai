```cpp
#include "../header.cpp"
namespace LinkCutTree{
  const int SIZ = 1e5 + 3;
  int F[SIZ], C[SIZ], S[SIZ], W[SIZ], A[SIZ], X[SIZ][2], size;
  bool T[SIZ];
  bool is_root(int x){ return X[F[x]][0] != x && X[F[x]][1] != x;}
  bool is_rson(int x){ return X[F[x]][1] == x;}
  int  new_node(int w){
    ++ size;
    W[size] = w, C[size] = S[size] = 1;
    A[size] = w, F[size] = 0;
    X[size][0] = X[size][1] = 0;
    return size;
  }
  void push_up(int x){
    S[x] = C[x] + S[X[x][0]] + S[X[x][1]];
    A[x] = W[x] ^ A[X[x][0]] ^ A[X[x][1]];
  }
  void push_down(int x){
    if(!T[x]) return;
    int lc = X[x][0], rc = X[x][1];
    if(lc) T[lc] ^= 1, swap(X[lc][0], X[lc][1]);
    if(rc) T[rc] ^= 1, swap(X[rc][0], X[rc][1]);
    T[x] = false;
  }
  void update(int x){
    if(!is_root(x)) update(F[x]); push_down(x);
  }
  void rotate(int x){
    int y = F[x], z = F[y];
    bool f = is_rson(x);
    bool g = is_rson(y);
    if(is_root(y)){
      F[x] = z, F[y] = x;
      X[y][ f] = X[x][!f], F[X[x][!f]] = y;
      X[x][!f] = y;
    } else {
      F[x] = z, F[y] = x;
      X[z][ g] = x;
      X[y][ f] = X[x][!f], F[X[x][!f]] = y;
      X[x][!f] = y;
    }
    push_up(y), push_up(x);
  }
  void splay(int x){
    update(x);
    for(int f = F[x];f = F[x], !is_root(x);rotate(x))
      if(!is_root(f)) rotate(is_rson(x) == is_rson(f) ? f : x);
  }
  int  access(int x){
    int p;
    for(p = 0;x;p = x, x = F[x]){
      splay(x), X[x][1] = p, push_up(x);
    }
    return p;
  }
  void make_root(int x){
    x = access(x);
    T[x] ^= 1, swap(X[x][0], X[x][1]);
  }
  int find_root(int x){
    access(x), splay(x), push_down(x);
    while(X[x][0]) x = X[x][0], push_down(x);
    splay(x);
    return x;
  }
  void link(int x, int y){
    make_root(x), splay(x), F[x] = y;
  }
  void cut(int x, int p){
    make_root(x), access(p), splay(p), X[p][0] = F[x] = 0;
  }
  void modify(int x, int w){
    splay(x), W[x] = w, push_up(x);
  }
}
const int MAXN = 1e5 + 3;
map<pair<int, int>, bool> M;
int n, m;
int main(){
  cin >> n >> m;
  for(int i = 1;i <= n;++ i){
    int a; cin >> a;
    LinkCutTree :: new_node(a);
  }
  for(int i = 1;i <= m;++ i){
    int o; cin >> o;
    if(o == 0){
      int u, v; cin >> u >> v;
      LinkCutTree :: make_root(u);
      int p = LinkCutTree :: access(v);
      printf("%d\n", LinkCutTree :: A[p]);
    } else if(o == 1){
      int u, v; cin >> u >> v;
      int a = LinkCutTree :: find_root(u);
      int b = LinkCutTree :: find_root(v);
      if(a != b){
        LinkCutTree :: link(u, v);
        M[make_pair(min(u, v), max(u, v))] = true;
      }
    } else if(o == 2){
      int u, v; cin >> u >> v;
      if(M.count(make_pair(min(u, v), max(u, v)))){
        M.erase(make_pair(min(u, v), max(u, v)));
        LinkCutTree :: cut(u, v);
      }
    } else {
      int u, w; cin >> u >> w;
      LinkCutTree :: modify(u, w);
    }
  }
  return 0;
}
```
