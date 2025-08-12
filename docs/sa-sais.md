```cpp
#include "../header.cpp"
#define LTYPE 0
#define STYPE 1
void induce_sort(int n, int S[], int T[], int m, int LM[], int SA[], int C[]){
  vector <int> BL(n), BS(n), BM(n);
  fill(SA, SA + n, -1);
  for(int i = 0;i < n;++ i){        // 预处理桶
    BM[i] = BS[i] = C[i] - 1;
    BL[i] = i == 0 ? 0 : C[i - 1];
  }
  for(int i = m - 1;i >= 0;-- i)    // 放置 LMS 后缀
    SA[BM[S[LM[i]]] --] = LM[i];
  for(int i =   0, p;i  < n;++ i)   // 计算 L 类型后缀的位置
    if(SA[i] > 0 && T[p = SA[i] - 1] == LTYPE)
      SA[BL[S[p]] ++] = p;
  for(int i = n - 1, p;i >= 0;-- i) // 计算 S 类型后缀的位置
    if(SA[i] > 0 && T[p = SA[i] - 1] == STYPE)
      SA[BS[S[p]] --] = p;
}
// 长度 n，字符集 [0, n)，要求最后一个元素为 0
// 例如输入 ababa 传入 n = 6, S = [1 2 1 2 1 0]
void sais(int n, int S[], int SA[]){
  vector <int> T(n), C(n), I(n, -1);
  T[n - 1] = STYPE;
  for(int i = n - 2;i >= 0;-- i){   // 递推类型
    T[i] = S[i] == S[i + 1] ? T[i + 1] : (S[i] < S[i + 1] ? STYPE : LTYPE);
  }
  for(int i = 0;i < n;++ i)    // 统计个数
    C[S[i]] ++;
  for(int i = 1;i < n;++ i)    // 前缀累加
    C[i] += C[i - 1];
  vector <int> P;
  for(int i = 0;i < n;++ i){   // 统计 LMS 后缀
    if(T[i] == STYPE && (i == 0 || T[i - 1] == LTYPE)){
      I[i] = P.size(), P.push_back(i);
    }
  }
  int m = P.size(), tot = 0, cnt = 0;
  induce_sort(n, S, T.data(), m, P.data(), SA, C.data());
  vector <int> S0(m), SA0(m);
  for(int i = 0, x, y = -1;i < n;++ i){
    if((x = I[SA[i]]) != -1){
      if(tot == 0 || P[x + 1] - P[x] != P[y + 1] - P[y])
        tot ++;
      else for(int p1 = P[x], p2 = P[y];p2 <= P[y + 1];++ p1, ++ p2){
        if((S[p1] << 1 | T[p1]) != (S[p2] << 1 | T[p2])){
          tot ++; break;
        }
      }
      S0[y = x] = tot - 1;
    }
  }
  if(tot == m){
    for(int i = 0;i < m;++ i)
      SA0[S0[i]] = i;
  } else {
    sais(m, S0.data(), SA0.data());
  }
  for(int i = 0;i < m;++ i)
    S0[i] = P[SA0[i]];
  induce_sort(n, S, T.data(), m, S0.data(), SA, C.data());
}
int S[MAXN], SA[MAXN], H[MAXM], G[MAXM];
int main(){
  int n = 0, t = 0, m = 256;
  for(char c = cin.get();isgraph(c);c = cin.get()){
    S[n ++] = c;
    H[c] ++;
  }
  for(int i = 0;i < m;++ i){
    t += !!H[i], G[i] = t;
  }
  for(int i = 0;i < n;++ i){
    S[i] = G[S[i]];
  }
  sais(n + 1, S, SA);
  for(int i = 1;i <= n;++ i){
    cout << SA[i] + 1 << " ";
  }
  return 0;
}
```
