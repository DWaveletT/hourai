## 定义

$$
\begin{aligned}
z^{(1)}_i &= |\mathrm{lcp}(b, \mathrm{suffix}(b, i))| \\
z^{(2)}_i &= |\mathrm{lcp}(b, \mathrm{suffix}(a, i))| \\
\end{aligned}
$$

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long i64;
const int MAXN = 2e7 + 3;
char A[MAXN], B[MAXN * 2];
int n, m, l, r, Z[MAXN * 2];
i64 ans1, ans2;
int main(){
    scanf("%s%s", A + 1, B + 1);
    n = strlen(A + 1);
    m = strlen(B + 1);
    l = 0, r = 0; Z[1] = 0, ans1 = m + 1;
    for(int i = 2;i <= m;++ i){
        if(i <= r) Z[i] = min(r - i + 1, Z[i - l + 1]);
        else       Z[i] = 0;
        while(B[Z[i] + 1] == B[i + Z[i]])
            ++ Z[i];
        if(i + Z[i] - 1 > r)
            r = i + Z[i] - 1, l = i;
        ans1 ^= 1ll * i * (Z[i] + 1);
    }
    l = 0, r = 0;
    Z[1] = 0, B[m + 1] = '#', strcat(B + 1, A + 1);
    for(int i = 2;i <= n + m + 1;++ i){
        if(i <= r) Z[i] = min(r - i + 1, Z[i - l + 1]);
        else       Z[i] = 0;
        while(B[Z[i] + 1] == B[i + Z[i]])
            ++ Z[i];
        if(i + Z[i] - 1 > r)
            r = i + Z[i] - 1, l = i;
    }
    for(int i = m + 2;i <= n + m + 1;++ i){
        ans2 ^= 1ll * (i - m - 1) * (Z[i] + 1);
    }
    printf("%lld\n%lld\n", ans1, ans2);
    return 0;
}
```
