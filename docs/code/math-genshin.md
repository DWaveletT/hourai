## 用法

计算 $P$ 的最小原根。

## 附注

原根表，其中 $P = r\times 2^{k}$，对应原根为 $g$。

$$
\def\arraystretch{1.5}
\begin{array}{c|c|c|c||c|c|c|c}
\hline\hline
\mathrm{Prime} & r & k & g & \mathrm{Prime} & r & k & g \\ \hline
5                  & 1   & 2  & 2  & 3221225473          & 3  & 30 & 5 \\
17                 & 1   & 4  & 3  & 75161927681         & 35 & 31 & 3 \\
97                 & 3   & 5  & 5  & 77309411329         & 9  & 33 & 7 \\
193                & 3   & 6  & 5  & 206158430209        & 3  & 36 & 22 \\
257                & 1   & 8  & 3  & 2061584302081       & 15 & 37 & 7 \\
7681               & 15  & 9  & 17 & 2748779069441       & 5  & 39 & 3 \\
12289              & 3   & 12 & 11 & 6597069766657       & 3  & 41 & 5 \\
40961              & 5   & 13 & 3  & 39582418599937      & 9  & 42 & 5 \\
65537              & 1   & 16 & 3  & 79164837199873      & 9  & 43 & 5 \\
786433             & 3   & 18 & 10 & 263882790666241     & 15 & 44 & 7 \\
5767169            & 11  & 19 & 3  & 1231453023109121    & 35 & 45 & 3 \\
7340033            & 7   & 20 & 3  & 1337006139375617    & 19 & 46 & 3 \\
23068673           & 11  & 21 & 3  & 3799912185593857    & 27 & 47 & 5 \\
104857601          & 25  & 22 & 3  & 4222124650659841    & 15 & 48 & 19 \\
167772161          & 5   & 25 & 3  & 7881299347898369    & 7  & 50 & 6 \\
469762049          & 7   & 26 & 3  & 31525197391593473   & 7  & 52 & 3 \\
1004535809         & 479 & 21 & 3  & 180143985094819841  & 5  & 55 & 6 \\
2013265921         & 15  & 27 & 31 & 1945555039024054273 & 27 & 56 & 5 \\
2281701377         & 17  & 27 & 3  & 4179340454199820289 & 29 & 57 & 3 \\
\hline 
\end{array}
$$

```cpp
#include<bits/stdc++.h>
using namespace std;

int power(int a, int b, int p){
    int r = 1;
    while(b){
        if(b & 1) r = 1ll * r * a % p;
        b >>= 1,  a = 1ll * a * a % p;
    }
    return r;
}

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

int main(){
    cout << get_genshin(998244353) << endl;
    return 0;
}
```
