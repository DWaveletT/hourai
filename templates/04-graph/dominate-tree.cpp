// From Alex_Wei
#include "../header.cpp"

int n, m, dn, F[MAXN], ind[MAXN], dfn[MAXN];
int sdom[MAXN], idom[MAXN], sz[MAXN];
vector<int> E[MAXN], rE[MAXN], buc[MAXN];
void dfs(int u, int f) {
  ind[dfn[u] = ++dn] = u, F[u] = f;
  for(auto &v: E[u]) if(!dfn[v]) dfs(v, u);
}
struct dsu {
  // M 维护 sdom 最小的点的编号
  int F[MAXN], M[MAXN];
  int find(int x) {
    if(F[x] == x) return F[x];
    int f = F[x];
    F[x] = find(f);
    if(sdom[M[f]] < sdom[M[x]]) M[x] = M[f];
    return F[x];
  }
  int get(int x) { return find(x), M[x]; }
} tr;
int main() {
  cin >> n >> m;
  for(int i = 1; i <= m; i++) {
    int u, v; cin >> u >> v;
    E[u].push_back(v), rE[v].push_back(u);
  }
  dfs(1, 0), sdom[0] = n + 1;
  for(int i = 1; i <= n; i++) tr.F[i] = i;
  for(int i = n; i; -- i){
    int u = ind[i];
    for(auto &v: buc[i]) idom[v] = tr.get(v);
    if(i == 1) break;
    sdom[u] = i;
    for(auto &v: rE[u]) {
      sdom[u] = min(sdom[u], dfn[v] <= i ? dfn[v] : sdom[tr.get(v)]);
    }
    tr.M[u] = u, tr.F[u] = F[u];
    buc[sdom[u]].push_back(u);
  }
  for(int i = 2; i <= n; i++) {
    int u = ind[i];
    if(sdom[idom[u]] != sdom[u])
        idom[u] = idom[idom[u]];
    else idom[u] = sdom[u];
  }
  for(int i = n;i;i --){
    sz[i] += 1;
    if(i > 1) sz[ind[idom[i]]] += sz[i];
  }
  for(int i = 1;i <= n;++ i){
    cout << sz[i] << " \n"[i == n];
  }
  return 0;
}
