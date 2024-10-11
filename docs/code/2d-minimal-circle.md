```cpp
#include "2d.cpp"

point geto(point a, point b, point c) {
    double a1, a2, b1, b2, c1, c2;
    point ans(0, 0);
    a1 = 2 * (b.x - a.x), b1 = 2 * (b.y - a.y),
    c1 = sqr(b.x) - sqr(a.x) + sqr(b.y) - sqr(a.y);
    a2 = 2 * (c.x - a.x), b2 = 2 * (c.y - a.y),
    c2 = sqr(c.x) - sqr(a.x) + sqr(c.y) - sqr(a.y);
    if (equal(a1, 0)) {
        ans.y = c1 / b1;
        ans.x = (c2 - ans.y * b2) / a2;
    } else if (equal(b1, 0)) {
        ans.x = c1 / a1;
        ans.y = (c2 - ans.x * a2) / b2;
    } else {
        ans.x = (c2 * b1 - c1 * b2) / (a2 * b1 - a1 * b2);
        ans.y = (c2 * a1 - c1 * a2) / (b2 * a1 - b1 * a2);
    }
    return ans;
}

mt19937 MT;
circ minimal(vector <point> V){
    shuffle(V.begin(), V.end(), MT);

    point  o = V[0];
    double r = 0;
    for(int i = 0;i < V.size();++ i) {
        if (sign(dis(o, V[i]) - r) != 1) continue;
        o.x = (V[i].x + V[0].x) / 2;
        o.y = (V[i].y + V[0].y) / 2;
        r = dis(V[i], V[0]) / 2;
        for(int j = 0;j < i;++ j) {
            if (sign(dis(o, V[j]) - r) != 1) continue;
            o.x = (V[i].x + V[j].x) / 2;
            o.y = (V[i].y + V[j].y) / 2;
            r = dis(V[i], V[j]) / 2;
            for(int k = 0;k < j;++ k) {
                if (sign(dis(o, V[k]) - r) != 1) continue;
                o = geto(V[i], V[j], V[k]);
                r = dis(o, V[i]);
            }
        }
    }
    circ res;
    res.o = o;
    res.r = r;
    return res;
}
```
