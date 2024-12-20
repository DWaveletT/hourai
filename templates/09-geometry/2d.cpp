#include <bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

int qread();

const double EPS = 1e-9;
const double PI  = acos(-1);

bool equal(double a, double b){
    return fabs(a - b) < EPS;
}

int sign(double a){
    if(equal(a, 0))
        return 0;
    return a > 0 ? 1 : -1;
}

double sqr(double x){
    return x * x;
}

struct vec{   // 二维向量
    double x;
    double y;
    vec(){}
    vec(double _x, double _y) : x(_x), y(_y){}
};

vec operator +(const vec &a, const vec &b){
    return vec(a.x + b.x, a.y + b.y);
}
vec operator -(const vec &a, const vec &b){
    return vec(a.x - b.x, a.y - b.y);
}
double mulp(const vec &a, const vec &b){
    return a.x * b.x + a.y * b.y;
}
double mulx(const vec &a, const vec &b){
    return a.x * b.y - a.y * b.x;
}
vec mul(const double &r, const vec &a){
    return vec(r * a.x, r * a.y);
}
bool equal(vec a, vec b){
    return equal(a.x, b.x) && equal(a.y, b.y);
}

using point = vec;

point rotate(point a, double t){
    double c = cos(t);
    double s = sin(t);
    return point(a.x * c - a.y * s, a.y * c + a.x * s);
}

bool cmpx(point a, point b){
    return sign(a.x - b.x) == -1;
}
bool cmpy(point a, point b){
    return sign(a.y - b.y) == -1;
}

struct line{    // 有向直线
    point o;
    vec p;
    line(point _o, vec _p) : o(_o), p(_p){}
};
struct segm{    // 有向线段
    point a, b;
    segm(point _a, point _b) : a(_a), b(_b){}
};

int side(line l, point p){
    return sign(mulx(l.p, p - l.o));
}
int side(segm s, point p){
    return sign(mulx(s.b - s.a, p - s.a));
}

bool parallel(line a, line b){
    return equal(0, mulx(a.p, b.p));
}

double abs(vec a){
    return sqrt(a.x * a.x + a.y * a.y);
}
double dis(point a, point b){
    return sqrt(sqr(a.x - b.x) + sqr(a.y - b.y));
}
double abs(segm s){
    return dis(s.a, s.b);
}
double dis(line a, point p){
    return abs(mulx(p - a.o, a.p)) / abs(a.p);
}

point intersection(line a, line b){
    return b.o + mul(mulx(b.o - a.o, a.p) / mulx(a.p, b.p), b.p);
}

bool intersect(double l1, double r1, double l2, double r2){
    if(l1 > r1) swap(l1, r1);
    if(l2 > r2) swap(l2, r2);
    if(equal(r1, l2) || equal(r2, l1))
        return true;
    return !equal(max(r1, r2) - min(l1, l2), r1 - l1 + r2 - l2);
}

bool intersect(segm s1, segm s2){
    bool fx = intersect(s1.a.x, s1.b.x, s2.a.x, s2.b.x);
    if(!fx) return false;
    bool fy = intersect(s1.a.y, s1.b.y, s2.a.y, s2.b.y);
    if(!fy) return false;
    bool g1 = side(s1, s2.a) * side(s1, s2.b) == 1;
    if(g1) return false;
    bool g2 = side(s2, s1.a) * side(s2, s1.b) == 1;
    if(g2) return false;
    return true;
}

struct circ{  // 二维圆形
    point o;
    double r;
};

struct poly{  // 二维多边形
    vector <point> P;
};

double area(point a, point b, point c){
    return abs(mulx(b - a, c - a)) / 2;
}

double area(const poly &P){
    double ans = 0;
    for(int i = 0;i < P.P.size();++ i){
        const point &l = P.P[i];
        const point &r = P.P[i + 1 == P.P.size() ? 0 : i + 1];
        ans += mulx(l, r);
    }
    return ans / 2;
}

