#include "../header.cpp"
vector <vector<int>> A;
vector <pair<int, int>> V[MAXN];
stack  <int> S;
int D[MAXN], L[MAXN], o; bool I[MAXN];
void dfs(int u, int l){
  D[u] = L[u] = ++ o; I[u] = true, S.push(u);
  int s = 0;
  for(auto &[v, g] : V[u]) if(g != l) {
    if(D[v]){
      if(I[v])    L[u] = min(L[u], D[v]);
    } else {
      dfs(v, g), L[u] = min(L[u], L[v]), ++ s;
    }
  }
  if(D[u] == L[u]){
    vector <int> T;
    while(S.top() != u){
      int v = S.top(); S.pop();
      T.push_back(v), I[v] = false;
    }
    T.push_back(u), S.pop(), I[u] = false;
    A.push_back(T);
  }
}
