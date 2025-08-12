#include "../header.cpp"
vector<pair<int, int> > E[MAXN];
namespace LCA{
  const int SIZ = 5e5 + 3;
  int D[SIZ], H[SIZ], F[SIZ], P[SIZ], Q[SIZ], o;
  void dfs(int u, int f){
    P[u] = ++ o, Q[o] = u, F[u] = f, D[u] = D[f] + 1;
    for(auto &[v, w] : E[u]) if(v != f){
      H[v] = H[u] + w, dfs(v, u);
    }
  }
  const int MAXH = 18 + 3;
  int h = 18;
  int ST[SIZ][MAXH];
  int cmp(int a, int b){
    return D[a] < D[b] ? a : b;
  }
  int T[SIZ], n;
  void init(int _n, int root);
  int lca(int a, int b);
  int dis(int a, int b);
}
bool cmp(int a, int b){
  return LCA :: P[a] < LCA :: P[b];
}
bool I[MAXN];
vector <int> E1[MAXN], V1;
void solve(vector <int> &V){
  using LCA :: lca; using LCA :: D;
  stack <int> S;
  sort(V.begin(), V.end(), cmp);
  S.push(1);
  int v, l;
  for(auto &u : V) I[u] = true;
  for(auto &u : V) if(u != 1){
    int f = lca(u, S.top());
    l = -1;
    while(D[v = S.top()] > D[f]){
      if(l != -1)
        E1[v].push_back(l);
      V1.push_back(l = v), S.pop();
    }
    if(l != -1)
      E1[f].push_back(l);
    if(f != S.top()) S.push(f);
    S.push(u);
  }
  l = -1;
  while(!S.empty()){
    v = S.top();
    if(l != -1) E1[v].push_back(l);
    V1.push_back(l = v), S.pop();
  }
  // dfs(1, 0); // SOLVE HERE !!!
  for(auto &u : V1)
    E1[u].clear(), I[u] = false;
  V1.clear();
}
