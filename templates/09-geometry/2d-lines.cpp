// From Let it Rot
#include "2d.cpp"

struct line : pp {
  db z; // a * x + b * y + c (= or >) 0 
  line() = default;
  line(db a, db b, db c): pp{a, b}, z(c){} 

  // 有向平面 a -> b 左侧区域
  line(pp a, pp b): pp(r90(b - a)), z(a * b){}
  db operator()(pp a) const { // ax + by + c
    return a % pp(*this) + z;
  }
  line vertical() const {
    return {y, -x, 0};
  } // 过 O 的垂直线
  line parallel(pp o) {
    return {x, y, z - this -> operator()(o)};
  } // 过 O 的平行线
};

pp operator & (line x, line y) { // 求交
  return pp{
    pp{x.z, x.y} * pp{y.z, y.y},
    pp{x.x, x.z} * pp{y.x, y.z}
  } / -(pp(x) * pp(y));
} // 注意此处精度误差较大，res.y 需要较高精度

pp project(pp x, line l){   // 投影
  return x - pp(l) * (l(x) / l.norm());
}
pp reflact(pp x, line l){   // 对称
  return x - pp(l) * (l(x) / l.norm()) * 2;
}
db dis(line l, pp x = {0, 0}){  // 有向点距离
  return l(x) / l.abs();
}

bool is_parallel(line x, line y){ // 判断平行
  return equal(pp(x) * pp(y), 0);
}
bool is_vertical(line x, line y){ // 判断垂直
  return equal(pp(x) % pp(y), 0);
}
bool online(pp x, line l) {     // 判断点在线
  return equal(l(x), 0);
}

int ccw(pp a, pp b, pp c) { 
  int s = sign((b - a) * (c - a)); 
  if(s == 0) { 
    if(sign((b - a) % (c - a)) == -1)
      return 2; 
    if((c - a).norm() > (b - a).norm() + EPS)
      return -2; 
  }
  return s;
}

db det(line a, line b, line c) {
  pp A = a, B = b, C = c; 
  return c.z * (A * B) + a.z * (B * C) + b.z * (C * A);
}
db check(line a, line b, line c) { // sgn same as c(a & b), 0 if error
  return sign(det(a, b, c)) * sign(pp(a) * pp(b));
}
bool paraS(line a, line b) { // 射线同向
  return is_parallel(a, b) && pp(a)%pp(b) > 0;
}