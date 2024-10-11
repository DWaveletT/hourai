```cpp
#include <bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

int qread(){
    int w = 1, c, ret;
    while((c = getchar()) >  '9' || c <  '0') w = (c == '-' ? -1 : 1); ret = c - '0';
    while((c = getchar()) >= '0' && c <= '9') ret = ret * 10 + c - '0';
    return ret * w;
}

const int MAXN = 4e4 + 3;

int F[MAXN];

int main(){
    int n, m;
    cin >> n >> m;
    for(int i = 1;i <= n;++ i){
        int w, v, c;
        cin >> w >> v >> c;
        // w: value, v: volume, c: count
        for(int j = 0;j < v;++ j){
            deque <tuple<int, int> > Q;
            for(int k = 0;j + k * v <= m;++ k){
                int x = j + k * v;
                int f = F[x] - (x / v) * w;
                while(!Q.empty() && get<0>(Q.back ()) <= f)
                    Q.pop_back ();
                Q.push_back({f, x});
                while(!Q.empty() && get<1>(Q.front()) <  x - c * v)
                    Q.pop_front();
                F[x] = get<0>(Q.front()) + (x / v) * w;
            }
        }
    }
    cout << F[m] << endl;
    return 0;
}
```
