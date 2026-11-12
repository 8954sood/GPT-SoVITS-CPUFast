// FP32 arithmetic from either original FP32 or losslessly stored coefficients.
#include <arm_neon.h>
#include <cstdint>

template<bool HALF, int PREFETCH, int LOCALITY = 1>
static void gemv(const float* x, const void* data, const float* bias, float* y, int64_t n, int64_t k) {
 for (int64_t row=0;row<n/8;++row) {
  const auto* h=static_cast<const float16_t*>(data)+row*k*8;
  const auto* f=static_cast<const float*>(data)+row*k*8;
  float32x4_t c00=vdupq_n_f32(0),c01=c00,c10=c00,c11=c00,c20=c00,c21=c00,c30=c00,c31=c00;
  for (int64_t i=0;i<k;i+=4) {
   if constexpr (PREFETCH>0) {
    if (i+PREFETCH<k) {
    if constexpr (HALF) __builtin_prefetch(h+(i+PREFETCH)*8,0,LOCALITY);
    else __builtin_prefetch(f+(i+PREFETCH)*8,0,LOCALITY);
    }
   }
   auto a=vld1q_f32(x+i);
#define STEP(R) {float32x4_t w0,w1; if constexpr(HALF) {auto w=vld1q_f16(h+(i+R)*8);w0=vcvt_f32_f16(vget_low_f16(w));w1=vcvt_f32_f16(vget_high_f16(w));} else {w0=vld1q_f32(f+(i+R)*8);w1=vld1q_f32(f+(i+R)*8+4);} c##R##0=vfmaq_laneq_f32(c##R##0,w0,a,R);c##R##1=vfmaq_laneq_f32(c##R##1,w1,a,R);}
   STEP(0) STEP(1) STEP(2) STEP(3)
#undef STEP
  }
  auto a=vaddq_f32(vaddq_f32(c00,c10),vaddq_f32(c20,c30));
  auto b=vaddq_f32(vaddq_f32(c01,c11),vaddq_f32(c21,c31));
  if (bias) {a=vaddq_f32(a,vld1q_f32(bias+row*8));b=vaddq_f32(b,vld1q_f32(bias+row*8+4));}
  vst1q_f32(y+row*8,a);vst1q_f32(y+row*8+4,b);
 }
}
extern "C" int cpufast_ar_abi(){return 1;}
extern "C" void cpufast_ar_gemv(const float* x,const void* w,const float* b,float* y,int64_t n,int64_t k,int64_t half,int64_t prefetch) {
#define CALL(H) switch(prefetch){case 64:gemv<H,64>(x,w,b,y,n,k);break;case 128:gemv<H,128>(x,w,b,y,n,k);break;case 192:gemv<H,192>(x,w,b,y,n,k);break;case 320:gemv<H,320>(x,w,b,y,n,k);break;case 384:gemv<H,384>(x,w,b,y,n,k);break;case 256:gemv<H,256>(x,w,b,y,n,k);break;default:gemv<H,0>(x,w,b,y,n,k);}
 if(half){CALL(true)}else{CALL(false)}
#undef CALL
}
static void gemv16(const float* x,const float16_t* w,const float* bias,float* y,int64_t n,int64_t k){
 for(int64_t row=0;row<n/16;++row){const auto* h=w+row*k*16;
  float32x4_t c00=vdupq_n_f32(0),c01=c00,c02=c00,c03=c00,c10=c00,c11=c00,c12=c00,c13=c00,c20=c00,c21=c00,c22=c00,c23=c00,c30=c00,c31=c00,c32=c00,c33=c00;
  for(int64_t i=0;i<k;i+=4){auto a=vld1q_f32(x+i);
#define STEP16(R) {auto lo=vld1q_f16(h+(i+R)*16);auto hi=vld1q_f16(h+(i+R)*16+8);c##R##0=vfmaq_laneq_f32(c##R##0,vcvt_f32_f16(vget_low_f16(lo)),a,R);c##R##1=vfmaq_laneq_f32(c##R##1,vcvt_f32_f16(vget_high_f16(lo)),a,R);c##R##2=vfmaq_laneq_f32(c##R##2,vcvt_f32_f16(vget_low_f16(hi)),a,R);c##R##3=vfmaq_laneq_f32(c##R##3,vcvt_f32_f16(vget_high_f16(hi)),a,R);}
   STEP16(0) STEP16(1) STEP16(2) STEP16(3)
#undef STEP16
  }
#define STORE16(J) {auto v=vaddq_f32(vaddq_f32(c0##J,c1##J),vaddq_f32(c2##J,c3##J));if(bias)v=vaddq_f32(v,vld1q_f32(bias+row*16+J*4));vst1q_f32(y+row*16+J*4,v);}
  STORE16(0) STORE16(1) STORE16(2) STORE16(3)
#undef STORE16
 }
}
