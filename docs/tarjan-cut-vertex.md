```cpp
#include "../header.cpp"
vector<int> V[MAXN];
int n, m, o, D[MAXN], L[MAXN];
bool F[MAXN], C[MAXN];
void dfs(int u, int g){
  L[u] = D[u] = ++ o, F[u] = true; int s = 0;
  for(auto &v : V[u]){
    if(!F[v]){
      dfs(v, g), ++ s;
      L[u] = min(L[u], L[v]);
      if(u != g && L[v] >= D[u]) C[u] = true;
    } else {
      L[u] = min(L[u], D[v]);
    }
  }
  if(u == g && s > 1) C[u] = true;
}
int main(){
  cin >> n >> m;
  for(int i = 1;i <= m;++ i){
    int u, v;
    cin >> u >> v;
    V[u].push_back(v);
    V[v].push_back(u);
  }
  for(int i = 1;i <= n;++ i)
    if(!F[i]) dfs(i, i);
  vector <int> ANS;
  for(int i = 1;i <= n;++ i)
    if(C[i]) ANS.push_back(i);
  cout << ANS.size() << endl;
  for(auto &u : ANS)
    cout << u << " ";
  return 0;
}
```
