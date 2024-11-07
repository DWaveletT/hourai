## 例题

给定 $n$ 个点，保证每三点不共线。要求找到一个简单多边形满足它不是凸包，使得该多边形面积最大。

```cpp
#include<bits/stdc++.h>
using namespace std;

using i64 = long long;

const int MAXN = 2e5 + 3;
int X[MAXN], Y[MAXN];

struct Frac {
    int a, b;
    Frac (int _a, int _b){
        if(_b < 0){
            a = -_a, b = -_b;
        } else {
            a =  _a, b =  _b;
        }
    }
};

struct Node {
    int x, y;
}P[MAXN];

bool operator < (const Frac A, const Frac B){
    return 1ll * A.a * B.b - 1ll * A.b * B.a < 0;
}
bool operator < (const Node A, const Node B){
    return A.x == B.x ? A.y > B.y : A.x < B.x;
}

const Frac intersect(Node A, Node B){
    int a = B.y - A.y;
    int b = A.x - B.x;
    assert(b != 0);
    if(b < 0){
        a = -a, b = -b;
    }
    return Frac(a, b);
}
bool F[MAXN];
int main(){
    int TT;
    cin >> TT;
    while(TT -- ){
        int n;
        cin >> n;
        int maxx = -1e9, minx = 1e9;
        for(int i = 1;i <= n;++ i){
            auto &[x, y] = P[i];
            cin >> x >> y;
            F[i] = false;
        }
        sort(P + 1, P + 1 + n);
        vector <int> Q1, Q2, Q;
        // Q1 计算上凸壳，Q2 计算下凸壳
        for(int i = 1;i <= n;++ i){
            auto &[x, y] = P[i];
            if(Q1.size() <= 1){
                Q1.push_back(i);
            } else {
                while(Q1.size() >= 2){
                    auto &[x1, y1] = P[Q1[Q1.size() - 1]];
                    auto &[x2, y2] = P[Q1[Q1.size() - 2]];
                    long long cmp = 1ll * (y - y1) * (x1 - x2) - 1ll * (x - x1) * (y1 - y2);
                    if(cmp > 0){
                        Q1.pop_back();
                    } else break;
                }
                Q1.push_back(i);
            }
            if(Q2.size() <= 1){
                Q2.push_back(i);
            } else {
                while(Q2.size() >= 2){
                    auto &[x1, y1] = P[Q2[Q2.size() - 1]];
                    auto &[x2, y2] = P[Q2[Q2.size() - 2]];
                    long long cmp = 1ll * (y - y1) * (x1 - x2) - 1ll * (x - x1) * (y1 - y2);
                    if(cmp < 0){
                        Q2.pop_back();
                    } else break;
                }
                Q2.push_back(i);
            }
        }

        Q = Q1;
        for(int i = Q2.size();i != 0;i --){
            if(i != Q2.size())
                Q.push_back(Q2[i - 1]);
        }
        long long area = 0;
        int x0 = P[Q[0]].x;
        int y0 = P[Q[0]].y;
        for(int i = 1;i + 1 < Q.size();++ i){
            auto &[x1, y1] = P[Q[    i]];
            auto &[x2, y2] = P[Q[i + 1]];
            area += 1ll * (x1 - x0) * (y2 - y0) - 1ll * (x2 - x0) * (y1 - y0);
        }
        area = -area;
        for(auto &i: Q1) F[i] = true;
        for(auto &i: Q2) F[i] = true;
        bool ok = false;
        for(int i = 1;i <= n;++ i) if(!F[i]){
            ok = true;
            maxx = max(maxx, P[i].x);
            minx = min(minx, P[i].x);
        }
        if(!ok){
            cout << -1 << "\n";
            continue;
        }
        vector <int>  L1;
        vector <int>  L2;
        // L1 插入 kx + b 维护下凸壳
        for(int i = 1;i <= n;++ i) if(!F[i]){
            auto &[k, b] = P[i];
            if(!L1.empty() && k == P[L1.back()].x)
                continue;
            while(L1.size() >= 2){
                auto &P1 = P[L1[L1.size() - 1]];
                auto &P2 = P[L1[L1.size() - 2]];
                Frac i1 = intersect(P1, P[i]);
                Frac i2 = intersect(P2, P[i]);
                if(i1 < i2){
                    L1.pop_back();
                } else break;
            }
            L1.push_back(i);
        }
        // L2 插入 kx + b 维护上凸壳
        for(int i = n;i >= 1;-- i) if(!F[i]){
            auto &[k, b] = P[i];
            if(!L2.empty() && k == P[L2.back()].x)
                continue;
            while(L2.size() >= 2){
                auto &P1 = P[L2[L2.size() - 1]];
                auto &P2 = P[L2[L2.size() - 2]];

                Frac i1 = intersect(P1, P[i]);
                Frac i2 = intersect(P2, P[i]);
                if(i1 < i2){
                    L2.pop_back();
                } else break;
            }
            L2.push_back(i);
        }
        vector <Frac> E1;
        E1.push_back(Frac( -2e9, 1 ));
        for(int i = 0;i + 1 < L1.size();++ i){
            auto &P1 = P[L1[i    ]];
            auto &P2 = P[L1[i + 1]];
            E1.push_back(intersect(P1, P2));
        }
        vector <Frac> E2;
        E2.push_back(Frac( -2e9, 1 ));
        for(int i = 0;i + 1 < L2.size();++ i){
            auto &P1 = P[L2[i    ]];
            auto &P2 = P[L2[i + 1]];
            E2.push_back(intersect(P1, P2));
        }
        long long ans = 0;
        for(int i = 0;i + 1 < Q.size();++ i){
            auto &[x1, y1] = P[Q[i    ]];
            auto &[x2, y2] = P[Q[i + 1]];
            long long w = 1ll * x2 * y1 - 1ll * x1 * y2;
            int A = y2 - y1;
            int B = x1 - x2;
            int x = 0, y = 0;
            if(B == 0){
                if(A > 0){
                    x = minx, y = 0;
                } else {
                    x = maxx, y = 0;
                }
            } else 
            if(B <  0){
                Frac K = Frac(-A, -B);
                int p = 0;
                for(int k = 20;k >= 0;-- k){
                    int pp = p | 1 << k;
                    if(pp < E1.size() && E1[pp] < K){
                        p = pp;
                    }
                }
                x = P[L1[p]].x;
                y = P[L1[p]].y;
            } else {
                Frac K = Frac( A,  B);
                int p = 0;
                for(int k = 20;k >= 0;-- k){
                    int pp = p | 1 << k;
                    if(pp < E2.size() && E2[pp] < K){
                        p = pp;
                    }
                }
                x = P[L2[p]].x;
                y = P[L2[p]].y;
            }
            ans = max(ans, area - (w + 1ll * A * x + 1ll * B * y));
        }
        // cerr << "ans = " << ans << endl;
        cout << ans << "\n";
    }
    return 0;
}
```
