#include "../header.cpp"

using db = double;
const db EPS = 1e-9, PI  = acos(-1);

bool equal(db a, db b){ return fabs(a - b) < EPS; }
int sign(db a){ if(equal(a, 0)) return 0; return a > 0 ? 1 : -1;}
db sqr(db x) { return x * x; }

struct v2{   // 二维向量
  db x, y;
  v2(db _x = 0, db _y = 0) : x(_x), y(_y){}
  db norm() const {return (x * x + y * y);}
  db abs() const {return sqrt(x * x + y * y);}
  db arg() const {return atan2(y, x); }
};
v2 r90(v2 x) { return {-x.y, x.x}; }
v2 operator +(v2 a, v2 b){
  return {a.x + b.x, a.y + b.y}; }
v2 operator -(v2 a, v2 b){
  return {a.x - b.x, a.y - b.y}; }
v2 operator *(v2 a, db w){
  return {a.x * w, a.y * w}; }
v2 operator /(v2 a, db w){
  return {a.x / w, a.y / w}; }
db operator *(v2 a, v2 b){  // 叉乘，b > a 为负
  return a.x * b.y - a.y * b.x; }
db operator %(v2 a, v2 b){  // 点乘
  return a.x * b.x + a.y * b.y; }
bool equal(v2 a, v2 b){
  return equal(a.x, b.x) && equal(a.y, b.y);
}
using pp = v2;
pp rotate(pp a, db t){
  db c = cos(t), s = sin(t);
  return pp(a.x * c - a.y * s, a.y * c + a.x * s);
}
db dis(pp a, pp b){
  return (a - b).abs();
}

int half(pp x){ // 为 1 则 arg >= \pi
  return x.y < 0 || (x.y == 0 && x.x <= 0);
}
// int half(pp x){return x.y < -EPS || (std::fabs(x.y) < EPS && x.x < EPS);}

bool cmp(pp a, pp b) {  // arg a < arg b
  return half(a) == half(b) ? a * b > 0 : half(b);
}