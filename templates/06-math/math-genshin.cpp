/**
## 用法

计算 $P$ 的最小原根。

原根表，其中 $P = r\times 2^{k}$，对应原根为 $g$。

$$
\begin{array}{c|c||c|c}
\hline\hline
\mathrm{Prime} & g & \mathrm{Prime} & g \\ \hline
104857601  & 3 & 7881299347898369 & 6 \\ 
167772161  & 3 & 31525197391593473 & 3 \\ 
469762049  & 3 & 180143985094819841 & 6 \\ 
998244353  & 3 & 1945555039024054273 & 5 \\ 
1004535809 & 3 & 4179340454199820289 & 3 \\ \hline\hline
\hline 
\end{array}
$$
**/
#include "../header.cpp"

int getphi(int x){
  int t = x, r = x;
  for(int i = 2;i <= x / i;++ i){
    if(t % i == 0){
      r = r / i * (i - 1);
      while(t % i == 0)
        t /= i;
    }
  }
  if(t != 1){
    r = r / t * (t - 1);
  }
  return r;
}
vector <int> getprime(int x){
  vector <int> p;
  int t = x;
  for(int i = 2;i <= x / i;++ i){
    if(t % i == 0){
      p.push_back(i);
      while(t % i == 0)
        t /= i;
    }
  }
  if(t != 1)
    p.push_back(x);
  return p;
}

bool test(int g, int m, int mm, vector<int> &P){
  for(auto &p: P){
    if(power(g, mm / p, m) == 1)
      return false;
  }
  return true;
}

int get_genshin(int m){
  int mm = getphi(m);
  vector <int> P = getprime(mm);
  for(int i = 1;;++ i){
    if(test(i, m, mm, P))
      return i;
  }
}