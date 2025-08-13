#include "../header.cpp"
int n;
vector <int> E[MAXN];
queue  <int> Q;
int vis[MAXN], F[MAXN], col[MAXN], pre[MAXN], mat[MAXN], tmp;

int getfa(int x){
  return x == F[x] ? x : F[x] = getfa(F[x]);
}
int lca(int x, int y){
  for(++ tmp;;x = pre[mat[x]], swap(x, y)) 
    if(vis[x = getfa(x)] == tmp) return x; 
      else vis[x] = x ? tmp : 0;
}
void flower(int x, int y, int z){
  while(getfa(x) != z){
    pre[x] = y, y = mat[x], F[x] = F[y] = z;
    x = pre[y];
    if(col[y] == 2)
      Q.push(y), col[y] = 1;
  }
}
bool aug(int u){
  for(int i = 1;i <= n;++ i)
    col[i] = pre[i] = 0, F[i] = i;
  Q = queue<int>({ u }), col[u] = 1;
  while(!Q.empty()){
    auto x = Q.front(); Q.pop();
    for(auto &v: E[x]){
      int y = v, z;
      if(col[y] == 2) continue;
      if(col[y] == 1) {
        z = lca(x, y);
        flower(x, y, z), flower(y, x, z);
      } else 
      if(!mat[y]){
        for(pre[y] = x; y;){
          mat[y] = x = pre[y], swap(y,mat[x]);
        }
        return true;
      } else {
        pre[y] = x, col[y] = 2;
        Q.push(mat[y]), col[mat[y]] = 1;
      }
    }
  }
  return false;
}
