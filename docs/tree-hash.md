## 用法

给定大小为 $n$ 的以 $1$ 为根的树，计算 $h_i$ 表示子树 $i$ 的哈希值，计算有多少个本质不同的值。

```cpp
#include "../header.cpp"
u64 xor_shift(u64 x);
u64 H[MAXN];
vector <int> E[MAXN];
void dfs(int u, int f){
  H[u] = 1;
  for(auto &v: E[u]) if(v != f){
    dfs(v, u);
    H[u] += H[v];
  }
  H[u] = xor_shift(H[u]); // !important
}
int main(){
  int n;
  cin >> n;
  for(int i = 2;i <= n;++ i){
    int u, v;
    cin >> u >> v;
    E[u].push_back(v);
    E[v].push_back(u);
  }
  dfs(1, 0);
  sort(H + 1, H + 1 + n);
  cout << (unique(H + 1, H + 1 + n) - H - 1) << endl;
  return 0;
}
```
