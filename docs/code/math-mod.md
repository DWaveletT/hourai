## 用法

调用 init 计算出 $S$ 和 $X$，得到计算 $x \bmod P = (x\times X) / 2^{60} + S$。

```cpp
#include<bits/stdc++.h>
using namespace std;
long long S = 0, X = 0;
void init(int MOD){
    while((1 << (S + 1)) < MOD)
        S ++;
    X = ((__int128)1 << 60 + S) / MOD + !!(((__int128)1 << 60 + S) % MOD);
    cerr << S << " " << X << endl;
}
int power(long long x, int y, int MOD){
    long long r = 1;
    while(y){
        if(y & 1){
            r = r * x;
            r = r - MOD * ((__int128)r * X >> 60 + S);
        }
        x = x * x;
        x = x - MOD * ((__int128)x * X >> 60 + S);
        y >>= 1;
    }
    return r;
}
int main(){
    init(998244353);
    cout << power(2, 10, 998244353) << endl;
    cout << power(2, 20, 998244353) << endl;
    cout << power(2, 30, 998244353) << endl;
    cout << power(2, 40, 998244353) << endl;
    return 0;
}
```
