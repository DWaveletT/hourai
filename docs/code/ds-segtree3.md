```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long i64;

const int INF = 2e9;
const int MAXN= 5e5 + 3;
int A[MAXN];
struct Node{
    i64 sum; int len, max1, max2, max_cnt, his_mx;
    Node():
        sum(0), max1(-INF), max2(-INF), max_cnt(0), his_mx(-INF), len(0) {}
    Node(int w):
        sum(w), max1(   w), max2(-INF), max_cnt(1), his_mx(   w), len(1) {}
    bool update(int w1, int w2, int h1, int h2){
        his_mx = max({his_mx, max1 + h1});
        max1 += w1, max2 += w2;
        sum += 1ll * w1 * max_cnt + 1ll * w2 * (len - max_cnt);
        return max1 > max2;
    }
};
struct Tag{
    int max_add, max_his_add, umx_add, umx_his_add; bool have;
    void update(int w1, int w2, int h1, int h2){
        max_his_add = max(max_his_add, max_add + h1);
        umx_his_add = max(umx_his_add, umx_add + h2);
        max_add += w1, umx_add += w2, have = true;
    }
    void clear(){
        max_add = max_his_add = umx_add = umx_his_add = have = 0;
    }
};
struct Node operator +(Node a, Node b){
    Node t;
    t.max1 = max(a.max1, b.max1);
    if(t.max1 != a.max1){
        if(a.max1 > t.max2) t.max2 = a.max1;
    } else{
        if(a.max2 > t.max2) t.max2 = a.max2;
        t.max_cnt += a.max_cnt;
    }
    if(t.max1 != b.max1){
        if(b.max1 > t.max2) t.max2 = b.max1;
    } else{
        if(b.max2 > t.max2) t.max2 = b.max2;
        t.max_cnt += b.max_cnt;
    }
    t.sum = a.sum + b.sum, t.len = a.len + b.len;
    t.his_mx = max(a.his_mx, b.his_mx);
    return t;
}
namespace Seg{
    const int SIZ = 2e6 + 3;
    struct Node W[SIZ]; struct Tag T[SIZ];
    #define lc(t) (t << 1)
    #define rc(t) (t << 1 | 1)
    void push_up(int t, int a, int b){
        W[t] = W[lc(t)] + W[rc(t)];
    }
    void push_down(int t, int a, int b){
        if(a == b) T[t].clear();
        if(T[t].have){
            int c = a + b >> 1, x = lc(t), y = rc(t);
            int w = max(W[x].max1, W[y].max1);
            int w1 = T[t].max_add, w2 = T[t].umx_add, w3 = T[t].max_his_add, w4 = T[t].umx_his_add;
            if(w == W[x].max1)
                W[x].update(w1, w2, w3, w4),
                T[x].update(w1, w2, w3, w4);
            else 
                W[x].update(w2, w2, w4, w4),
                T[x].update(w2, w2, w4, w4);
            if(w == W[y].max1)
                W[y].update(w1, w2, w3, w4),
                T[y].update(w1, w2, w3, w4);
            else 
                W[y].update(w2, w2, w4, w4),
                T[y].update(w2, w2, w4, w4);
            T[t].clear();
        }
    }
    void build(int t, int a, int b){
        if(a == b){W[t] = Node(A[a]), T[t].clear();} else {
            int c = a + b >> 1; T[t].clear();
            build(lc(t), a,     c);
            build(rc(t), c + 1, b);
            push_up(t, a, b);
        }
    }
    void modiadd(int t, int a, int b, int l, int r, int w){
        if(l <= a && b <= r){
            T[t].update(w, w, w, w);
            W[t].update(w, w, w, w);
        } else {
            int c = a + b >> 1; push_down(t, a, b);
            if(l <= c) modiadd(lc(t), a,     c, l, r, w);
            if(r >  c) modiadd(rc(t), c + 1, b, l, r, w);
            push_up(t, a, b);
        }
    }
    void modimin(int t, int a, int b, int l, int r, int w){
        if(l <= a && b <= r){
            if(w >= W[t].max1) return; else 
            if(w >  W[t].max2){
                int k = w - W[t].max1;
                T[t].update(k, 0, k, 0);
                W[t].update(k, 0, k, 0);
            } else {
                int c = a + b >> 1;
                push_down(t, a, b);
                modimin(lc(t), a,     c, l, r, w);
                modimin(rc(t), c + 1, b, l, r, w);
                push_up(t, a, b);
            }
        } else {
            int c = a + b >> 1; push_down(t, a, b);
            if(l <= c) modimin(lc(t), a,     c, l, r, w);
            if(r >  c) modimin(rc(t), c + 1, b, l, r, w);
            push_up(t, a, b);
        }
    }
    Node query(int t, int a, int b, int l, int r){
        if(l <= a && b <= r) return W[t];
        int c = a + b >> 1; Node ret; push_down(t, a, b);
        if(l <= c) ret = ret + query(lc(t), a,     c, l, r);
        if(r >  c) ret = ret + query(rc(t), c + 1, b, l, r);
        return ret;
    }
}
int qread();
int main(){  
    int n = qread(), m = qread();
    for(int i = 1;i <= n;++ i)
        A[i] = qread();
    Seg :: build(1, 1, n);
    for(int i = 1;i <= m;++ i){
        int op = qread();
        if(op == 1){
            int l = qread(), r = qread(), w = qread();
            Seg :: modiadd(1, 1, n, l, r, w);
        } else if(op == 2){
            int l = qread(), r = qread(), w = qread();
            Seg :: modimin(1, 1, n, l, r, w);
        } else if(op == 3){
            int l = qread(), r = qread();
            auto p = Seg :: query(1, 1, n, l, r);
            printf("%lld\n", p.sum);
        } else if(op == 4){
            int l = qread(), r = qread();
            auto p = Seg :: query(1, 1, n, l, r);
            printf("%d\n", p.max1);
        } else if(op == 5){
            int l = qread(), r = qread();
            auto p = Seg :: query(1, 1, n, l, r);
            printf("%d\n", p.his_mx);
        }
    }
    return 0;
}
```
