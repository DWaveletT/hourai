#include "../header.cpp"
namespace SAM{
  const int SIZ = 2e6 + 3;
  int M[SIZ][MAXM];
  int L[SIZ], F[SIZ], S[SIZ];
  int last = 0, s = 0, h = 25;
  void init(){
    F[0] = -1, last = s = 0;
  }
  void extend(char c){
    int cur = ++ s, e = c - 'a';
    L[cur] = L[last] + 1;
    S[cur] = 1;
    int p = last;
    while(p != -1 && !M[p][e])
      M[p][e] = cur, p = F[p];
    if(p == -1){
      F[cur] = 0;
    } else {
      int q = M[p][e];
      if(L[p] + 1 == L[q]){
        F[cur] = q;
      } else {
        int clone = ++ s;
        L[clone] = L[p] + 1;
        F[clone] = F[q];
        S[clone] = 0;
        for(int i = 0;i <= h;++ i)
          M[clone][i] = M[q][i];
        while(p != -1 && M[p][e] == q)
          M[p][e] = clone, p = F[p];
        F[cur] = F[q] = clone;
      }
    }
    last = cur;
  }
  vector <int> E[SIZ];
  void build(){
    for(int i = 1;i <= s;++ i){
      E[F[i]].push_back(i);
    }
  }
  i64 ans = 0;
  void dfs(int u){
    for(auto &v : E[u]){
      dfs(v), S[u] += S[v];
    }
    if(S[u] > 1)
      ans = max(ans, 1ll * S[u] * L[u]);
  }
}