#include <bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

namespace BLOCK{
    const int SIZ = 1e5 + 3;
    const int BSZ = 700;
    const int BNM = SIZ / BSZ;

    int S[BNM], H[SIZ];

    void modify(int p, int w){
        H[p]       += w;
        S[p / BSZ] += w;
    }

    int query(int l, int r){
        int p = l / BSZ;
        int q = r / BSZ;
        int ans = 0;
        if(p == q){
            for(int i = l;i <= r;++ i)
                ans += H[i];
        } else {
            for(int i = l;i <  p * BSZ;++ i)
                ans += S[i];
            for(int i = q * BSZ;i <= r;++ i)
                ans += S[i];
        }
        return ans;
    }

    int get_kth(int k){
        for(int i = 0;i <= BNM;++ i){
            if(k > S[i])
                k -= S[i];
            else {
                int l = i * BSZ;
                int r = l + BSZ - 1;
                for(int i = l;i <= r;++ i){
                    if(k > H[i])
                        k -= H[i];
                    else return i;
                }
            }
        }
        return -1;
    }
}
