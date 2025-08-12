```cpp
#include "../header.cpp"
// Li: 左儿子；Ri: 右儿子
int n, L[MAXN], R[MAXN], A[MAXN];
void build(){
  stack <int> S;
  A[n + 1] = -1e9;
  for(int i = 1;i <= n + 1;++ i){
    int v = 0;
    while(!S.empty() && A[S.top()] > A[i]){
      auto u = S.top();
      R[u] = v, v  = u, S.pop();
    }
    L[i] = v, S.push(i);
  }
}
```
