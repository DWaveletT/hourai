/**
## 例题

给定一个仙人掌，多组询问 $u, v$ 之间最短路长度。
**/
#include "../header.cpp"
const int MAXD=  18 + 3;
struct edge{int u, v, w;};
vector <edge> V1[MAXN];
vector <edge> V2[MAXN];
vector <int> H[MAXN];
int n, D[MAXN], W[MAXN], F[MAXD][MAXN];
int o, X[MAXN], L[MAXN];
bool E[MAXN];
void dfs1(int u, int f){
  D[u] = D[f] + 1, F[0][u] = f;
  for(auto &e : V1[u]) if(e.v != f){
    if(D[e.v] && D[e.v] < D[u]){
      int a = e.u;
      int b = e.v;
      int c = ++ o, t = c + n;
      H[c].push_back(a);
      L[c] = W[a] - W[b] + e.w;
      while(a != b)
        E[a] = true, a = F[0][a], H[c].push_back(a);
      for(auto &x : H[c]){
        int w = min(W[x] - W[b], L[c] - W[x] + W[b]);
        V2[x].push_back(edge{x, t, w});
        V2[t].push_back(edge{t, x, w});
      }
    } else if(!D[e.v]){
      W[e.v] = W[u] + e.w, dfs1(e.v, u);
    }
  }
  for(auto &e : V1[u]) if(D[e.v] > D[u]){
    if(!E[e.v]){
      V2[e.u].push_back({e.u, e.v, e.w});
      V2[e.v].push_back({e.v, e.u, e.w});
    }
  }
}
int d = 18;
void dfs2(int u, int f){
  D[u] = D[f] + 1, F[0][u] = f;
  up(1, d, i) F[i][u] = F[i - 1][F[i - 1][u]];
  for(auto &e : V2[u]) if(e.v != f){
    X[e.v] = X[e.u] + e.w;
    dfs2(e.v, u);
  }
}
int lca(int u, int v){
  if(D[u] < D[v]) swap(u, v);
  dn(d, 0, i) if(D[F[i][u]] >= D[v]) u = F[i][u];
  if(u == v) return u;
  dn(d, 0, i) if(F[i][u] != F[i][v]) u = F[i][u], v = F[i][v];
  return F[0][u];
}
int jump(int u, int v){
  dn(d, 0, i) if(D[F[i][v]] >  D[u]) v = F[i][v];
  return v;
}
int dis(int x, int y){
  int t = lca(x, y);
  if(t > n){
    int u = jump(t, x);
    int v = jump(t, y);
    int w = abs(W[u] - W[v]);
    int l = min(w, L[t - n] - w);
    return X[x] - X[u] + X[y] - X[v] + l;
  } else {
    return X[x] + X[y] - 2 * X[t];
  }
}
int m, q;
int qread();
int main(){
  n = qread(), m = qread(), q = qread();
  up(1, m, i){
    int u = qread(), v = qread(), w = qread();
    V1[u].push_back(edge{u, v, w});
    V1[v].push_back(edge{v, u, w});
  }
  dfs1(1, 0);
  dfs2(1, 0);
  up(1, q, i){
    int u = qread(), v = qread();
    printf("%d\n", dis(u, v));
  }
  return 0;
}