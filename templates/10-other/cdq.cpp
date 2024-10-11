#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

void cdq(int l, int r){
    if(l != r){
        int t = l + r >> 1;
        cdq(l    , t);
        cdq(t + 1, r);
        int p = l, q = t + 1, u = l;
        while(p <= t && q <= r){
            if(A[p].b <= A[q].b)
                BIT :: increase(A[p].c, 1),       B[u ++] = A[p ++];
             else
                BIT :: query(A[q].c, K[A[q].id]), B[u ++] = A[q ++];
        }
        while(p <= t) BIT :: increase(A[p].c, 1),       B[u ++] = A[p ++];
        while(q <= r) BIT :: query(A[q].c, K[A[q].id]), B[u ++] = A[q ++];
        up(l, t, i) BIT :: decrease(A[i].c, 1);
        up(l, r, i) A[i] = B[i];
    }
}