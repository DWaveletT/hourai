/**
## 用法
$n$ 个物品，$m$ 容量背包，第 $i$ 个物品重量为 $w_i$ 价值为 $v_i$ 共有 $c_i$ 个，计算不超过容量的情况下最多拿多少价值的物品。
**/

#include <bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

int qread();
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