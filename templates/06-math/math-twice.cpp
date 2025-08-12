/**
## 用法

多次询问，每次询问给定奇素数 $p$ 以及 $y$，在 $\mathcal O(\log p)$ 复杂度计算 $x$ 使得 $x^2 \equiv 0 \pmod p$ 或者无解。
**/
#include "../header.cpp"
// 检查 x 在模 p 意义下是否有二次剩余
bool check(int x, int p){
  return power(x, (p - 1) / 2, p) == 1;
}
struct Node { int real, imag; };
Node mul(const Node a, const Node b, int p, int v){
  int nreal = (1ll * a.real * b.real + 1ll * a.imag * b.imag % p * v) % p;
  int nimag = (1ll * a.real * b.imag + 1ll * a.imag * b.real) % p;
  return { (nreal), nimag };
}
Node power(Node a, int b, int p, int v){
  Node r = { 1, 0 };
  while(b){
    if(b & 1) r = mul(r, a, p, v);
    b >>= 1,  a = mul(a, a, p, v);
  }
  return r;
}

mt19937 MT;
// 无解 x1 = x2 = -1，唯一解 x1 = x2
void solve(int n, int p, int &x1, int &x2){
  if(n == 0){ x1 = x2 =  0; return; }
  if(!check(n, p)){ x1 = x2 = -1; return; }
  int a, t;
  do {
    a = MT() % p;
  }while(check(t = (1ll * a * a - n + p) % p, p));
  Node u = { a, 1 };
  x1 = power(u, (p + 1) / 2, p, t).real;
  x2 = (p - x1) % p;
  if(x1 > x2) swap(x1, x2);
}
