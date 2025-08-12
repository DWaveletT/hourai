/**
## 例题
给定三元组序列 $(a_i, b_i, c_i)$，求解 $f(i) = \sum_{j} [a_j \le a_i \land b_j\le b_i \land c_j\le c_i]$。
**/
#include "../header.cpp"
struct Node{
  int id, a, b, c;
}A[MAXN], B[MAXN];
bool cmp(Node a, Node b){
  if(a.a != b.a) return a.a < b.a;
  if(a.b != b.b) return a.b < b.b;
  if(a.c != b.c) return a.c < b.c;
  return a.id < b.id;
}
int K[MAXN], H[MAXN];
int qread();
int n, m, D[MAXM];
namespace BIT{
  void increase(int x, int w){
    while(x <= m) D[x] += w, x += x & -x;
  }
  void decrease(int x, int w){
    while(x <= m) D[x] -= w, x += x & -x;
  }
  void query(int x, int &r){
    while(x) r += D[x], x -= x & -x;
  }
}
void cdq(int l, int r){
  if(l != r){
    int t = l + r >> 1; cdq(l, t), cdq(t + 1, r);
    int p = l, q = t + 1, u = l;
    while(p <= t && q <= r){
      if(A[p].b <= A[q].b)
        BIT :: increase(A[p].c, 1), B[u ++] = A[p ++];
       else
        BIT :: query(A[q].c, K[A[q].id]), B[u ++] = A[q ++];
    }
    while(p <= t) BIT :: increase(A[p].c, 1),     B[u ++] = A[p ++];
    while(q <= r) BIT :: query(A[q].c, K[A[q].id]), B[u ++] = A[q ++];
    up(l, t, i) BIT :: decrease(A[i].c, 1);
    up(l, r, i) A[i] = B[i];
  }
}
int main(){
  n = qread(), m = qread();
  up(1, n, i) A[i].id = i, A[i].a = qread(), A[i].b = qread(), A[i].c = qread();
  sort(A + 1, A + 1 + n, cmp), cdq(1, n);
  sort(A + 1, A + 1 + n, cmp);
  dn(n, 1, i){
    if(A[i].a == A[i + 1].a && A[i].b == A[i + 1].b && A[i].c == A[i + 1].c)
      K[A[i].id] = K[A[i + 1].id];
    H[K[A[i].id]] ++;
  }
  up(0, n - 1, i) printf("%d\n", H[i]);
  return 0;
}