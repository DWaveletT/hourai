/**
## 定义

$$
\begin{aligned}
z_i &= |\mathrm{lcp}(b, \mathrm{suffix}(b, i))| \\
\end{aligned}
$$
**/
#include "../header.cpp"
int Z[MAXN];
void exkmp(char A[]){
  int l = 0, r = 0; Z[1] = 0;
  for(int i = 2;A[i];++ i){
    Z[i] = i <= r ? min(r - i + 1, Z[i - l + 1]) : 0;
    while(A[Z[i] + 1] == A[i + Z[i]]) ++ Z[i];
    if(i + Z[i] - 1 > r)
      r = i + Z[i] - 1, l = i;
  }
}