## 用法

在线计算 $x = [x_1, x_2, \cdots, x_n]$ 在模 $p$ 意义下的逆元。

```cpp
#include "../header.cpp"
pair<int, int> F[MAXN], G[MAXN];
int I[MAXN];
using u32 = uint32_t;
u32 read(u32 &seed);
int main(){
  ios :: sync_with_stdio(false);
  cin.tie(nullptr);
  u32 seed;
  int n, p;
  cin >> n >> p >> seed;
  int m = pow(p, 1.0 / 3.0);
  I[1] = 1;
  for(int i = 2;i <= p / m;++ i){
    I[i] = 1ll * (p / i) * (p - I[p % i]) % p;
  }
  for(int i = 1;i < m;++ i){
    for(int j = i + 1;j <= m;++ j){
      if(!F[i * m * m / j].second){
        F[i * m * m / j] = { i, j };
        G[i * m * m / j] = { i, j };
      }
    }
  }
  F[    0] = G[    0] = { 0, 1 };
  F[m * m] = G[m * m] = { 1, 1 };
  for(int i = 1;i <    m * m;++ i) if(!F[i].second)
    F[i] = F[i - 1];
  for(int i = m * m - 1;i >= 1;-- i) if(!G[i].second)
    G[i] = G[i + 1];
  int lastans = 0;
  for(int i = 1;i <= n;++ i){
    int a, inv;
    a = (read(seed) ^ lastans) % (p - 1) + 1;
    int w = 1ll * a * m * m / p;
    auto &yy1 = F[w].second;  // *avoid y1 in <cmath>
    if(1ll * a * yy1 % p <= p / m){
      inv = 1ll * I[1ll * a * yy1 % p] * yy1 % p;
    } else {
      auto &yy2 = G[w].second;
      inv = 1ll * I[1ll * a * (p - yy2) % p] * (p - yy2) % p;
    }
    lastans = inv;
  }
  cout << lastans << "\n";
  return 0;
}
```
