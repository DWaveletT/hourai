#include "../header.cpp"
int n, m, A[MAXN], B[MAXN];
int C[MAXN], R[MAXN], P[MAXN], Q[MAXN];
char S[MAXN];
int main(){
  scanf("%s", S), n = strlen(S), m = 256;
  for(int i = 0;i < n;++ i) R[i] = S[i];
  for (int k = 1;k <= n;k <<= 1){
    for(int i = 0;i < n;++ i){
      Q[i] = ((i + k > n - 1) ? 0 : R[i + k]);
      P[i] = R[i];
      m = max(m, R[i]);
    }
#define fun(a, b, c) \
    memset(C, 0, sizeof(int) * (m + 1));          \
    for(int i = 0;i <  n;++ i) C[a] +=    1;      \
    for(int i = 1;i <= m;++ i) C[i] += C[i - 1];  \
    for(int i = n - 1;i >= 0;-- i) c[-- C[a]] = b;
    fun(Q[  i ],   i , B)
    fun(P[B[i]], B[i], A)
#undef fun
    int p = 1; R[A[0]] = 1;
    for(int i = 1;i <= n - 1;++ i){
      bool f1 = P[A[i]] == P[A[i - 1]];
      bool f2 = Q[A[i]] == Q[A[i - 1]];
      R[A[i]] = f1 && f2 ? R[A[i - 1]] : ++ p;
    }
    if (m == n) break;
  }
  for(int i = 0;i < n;++ i)
    printf("%u ", A[i] + 1);
  return 0;
}