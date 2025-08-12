/**
## 用法

计算：$$s(i) = \sum_{d\mid i} f_{d}$$
**/
#include "../header.cpp"
unsigned A[MAXN];
int p, P[MAXN]; bool V[MAXN];
void solve(int n){
  for(int i = 2;i <= n;++ i){
    if(!V[i]){
      P[++ p] = i;
      for(int j = 1;j <= n / i;++ j){ // 前缀和
        A[j * i] += A[j];
      }
    }
    for(int j = 1;j <= p && P[j] <= n / i;++ j){
      V[i * P[j]] = true;
      if(i % P[j] == 0) break;
    }
  }
}