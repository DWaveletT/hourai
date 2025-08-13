#include "../header.cpp"

using db = double;
const db EPS = 1e-9, PI  = acos(-1);

bool equal(db a, db b){ return fabs(a - b) < EPS; }
int sign(db a){ if(equal(a, 0)) return 0; return a > 0 ? 1 : -1;}
db sqr(db x) { return x * x; }

struct v2{   // 二维向量
  db x, y;
  v2(db _x = 0, db _y = 0) : x(_x), y(_y){}
  db norm() const { return x * x + y * y; }
  db abs() const { return std::sqrt(x * x + y * y); }
  db arg() const { return atan2(y, x); }
};
v2 r90(v2 x) { return {-x.y, x.x}; }
v2 operator +(v2 a, v2 b){return {a.x + b.x, a.y + b.y}; }
v2 operator -(v2 a, v2 b){return {a.x - b.x, a.y - b.y}; }
v2 operator *(v2 a, db w){return {a.x * w, a.y * w}; }
v2 operator /(v2 a, db w){return {a.x / w, a.y / w}; }
db operator *(v2 a, v2 b){return a.x * b.y - a.y * b.x; }
db operator %(v2 a, v2 b){return a.x * b.x + a.y * b.y; }
bool equal(v2 a, v2 b){
  return equal(a.x, b.x) && equal(a.y, b.y);
}
using pp = v2;
pp rotate(pp a, db t){
  db c = cos(t), s = sin(t);
  return pp(a.x * c - a.y * s, a.y * c + a.x * s);
}
bool cmpx(pp a, pp b){ return sign(a.x - b.x) == -1; }
bool cmpy(pp a, pp b){ return sign(a.y - b.y) == -1; }
struct line{
  pp o; v2 p; line(pp _o, v2 _p): o(_o), p(_p){}
};
struct segm{
  pp a, b; segm(pp _a, pp _b): a(_a), b(_b){}
};
int side(line l, pp p){
  return sign(l.p * (p - l.o));
}
int side(segm s, pp p){
  return sign((s.b - s.a) * (p - s.a));
}
bool parallel(line a, line b){
  return equal(0, a.p * b.p);
}
db dis(pp a, pp b){
  return (a - b).abs();
}
db dis(line a, pp p){
  return abs((p - a.o) * a.p) / a.p.abs();
}
pp intersection(line a, line b){
  return b.o + b.p * (((b.o - a.o) * a.p) / (a.p * b.p));
}
bool intersect(db l1, db r1, db l2, db r2){
  if(l1 > r1) swap(l1, r1);
  if(l2 > r2) swap(l2, r2);
  if(equal(r1, l2) || equal(r2, l1))
    return true;
  return !equal(max(r1, r2) - min(l1, l2), r1 - l1 + r2 - l2);
}
bool intersect(segm s1, segm s2){
  return !(
    intersect(s1.a.x, s1.b.x, s2.a.x, s2.b.x) ||
    intersect(s1.a.y, s1.b.y, s2.a.y, s2.b.y) ||
    side(s1, s2.a) * side(s1, s2.b) == 1 ||
    side(s2, s1.a) * side(s2, s1.b) == 1
  );
}
