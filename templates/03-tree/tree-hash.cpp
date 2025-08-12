/**
## 用法

有根树求出每个子树的哈希值，儿子间顺序可交换。
**/
#include "../header.cpp"
u64 xor_shift(u64 x);
u64 H[MAXN];
vector <int> E[MAXN];
void dfs(int u, int f){
  H[u] = 1;
  for(auto &v: E[u]) if(v != f)
    dfs(v, u), H[u] += H[v];
  H[u] = xor_shift(H[u]); // !important
}