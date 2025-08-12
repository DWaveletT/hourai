#include "../header.cpp"
u32 xorshift32(u32 &x){
  return x ^= x << 13, x ^= x >> 17, x ^= x << 5;
}
u64 xorshift64(u64 &x){
  return x ^= x << 13, x ^= x >> 7, x ^= x << 17;
}