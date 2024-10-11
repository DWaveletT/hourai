```cpp
#include "2d.cpp"

poly convex(vector <point> V){
    double t = PI * rand() / RAND_MAX;
    for(auto &p : V)
        p = rotate(p, t);
    poly res;
    sort(V.begin(), V.end(), cmpx);
    vector <point> L;
    vector <point> R;
    L.push_back(V.front());
    R.push_back(V.front());
    for(auto &p0 : V){
        while(L.size() >= 2){
            const point &p1 = L[L.size() - 1];
            const point &p2 = L[L.size() - 2];
            if(side(segm(p2, p1), p0) != -1)
                L.pop_back();
            else break;
        }
        L.push_back(p0);
        while(R.size() >= 2){
            const point &p1 = R[R.size() - 1];
            const point &p2 = R[R.size() - 2];
            if(side(segm(p2, p1), p0) !=  1)
                R.pop_back();
            else break;
        }
        R.push_back(p0);
    }
    for(int i = L.size() - 1;i >= 0;-- i)
        res.P.push_back(L[i]);
    for(int i = 1;i + 2 <= R.size();++ i)
        res.P.push_back(R[i]);
    for(auto &p : res.P)
        p = rotate(p, -t);
    return res;
}
```
