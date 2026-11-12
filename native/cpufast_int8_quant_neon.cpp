// Independent quantization-only probe; unchanged row4 SDOT and dequantization.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <cmath>
#include <vector>
#include <atomic>
#include <cfloat>
static std::atomic<int64_t> quant_neon_calls{0};
static at::Tensor arq_neon(const at::Tensor& x,const at::Tensor& w,const at::Tensor& scales,const at::Tensor& bias){
 quant_neon_calls.fetch_add(1,std::memory_order_relaxed);
 TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && w.scalar_type()==at::kChar && w.dim()==2 && x.scalar_type()==at::kFloat && x.dim()>0 && w.size(0)>0 && w.size(1)>0 && x.size(-1)==w.size(1) && w.size(1)==x.numel() && w.size(1)%16==0,"bad input");
 int64_t n=w.size(0),k=w.size(1); const float* a=x.const_data_ptr<float>(); float maxv=0;
 TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && scales.device().is_cpu() && bias.device().is_cpu() && scales.scalar_type()==at::kFloat && bias.scalar_type()==at::kFloat && scales.is_contiguous() && bias.is_contiguous() && scales.numel()==n && bias.numel()==n && n%4==0,"invalid int8 operands");
 // SIMD max is a selection over nonnegative finite values, with no sum
 // reassociation. Quantization uses FRINTI (current FPCR), like nearbyint.
 auto maxima=vdupq_n_f32(0); const auto largest=vdupq_n_f32(FLT_MAX);
 for(int64_t i=0;i<k;i+=4){
  auto values=vabsq_f32(vld1q_f32(a+i));
  TORCH_CHECK(vminvq_u32(vcleq_f32(values,largest))!=0,"non-finite input");
  maxima=vmaxq_f32(maxima,values);
 }
 maxv=vmaxvq_f32(maxima);
 float sx=maxv>0?maxv/127.0f:1.0f; float inv=1.0f/sx;
 std::vector<int8_t> q(k);
 if(!std::isfinite(inv)){
  // Keep the original expression for exceptional subnormal-scale inputs.
  for(int64_t i=0;i<k;i++)q[i]=int8_t(std::max(-127,std::min(127,int(std::nearbyint(a[i]*inv)))));
 }else{
  auto multiplier=vdupq_n_f32(inv);auto low=vdupq_n_s32(-127),high=vdupq_n_s32(127);
  for(int64_t i=0;i<k;i+=16){
   int32x4_t values[4];
   for(int j=0;j<4;j++){
    auto rounded=vrndiq_f32(vmulq_f32(vld1q_f32(a+i+j*4),multiplier));
    values[j]=vminq_s32(high,vmaxq_s32(low,vcvtq_s32_f32(rounded)));
   }
   auto first=vcombine_s16(vqmovn_s32(values[0]),vqmovn_s32(values[1]));
   auto second=vcombine_s16(vqmovn_s32(values[2]),vqmovn_s32(values[3]));
   vst1q_s8(q.data()+i,vcombine_s8(vqmovn_s16(first),vqmovn_s16(second)));
  }
 }
 auto shape=x.sizes().vec(); shape.back()=n; auto y=at::empty(shape,x.options()); auto dst=y.mutable_data_ptr<float>();auto wp=w.const_data_ptr<int8_t>();auto s=scales.const_data_ptr<float>();auto b=bias.const_data_ptr<float>();
 TORCH_CHECK(n%4==0,"four-row geometry required");
 at::parallel_for(0,n/4,4,[&](int64_t begin,int64_t end){for(int64_t rb=begin;rb<end;rb++){int64_t r=rb*4; auto v0=vdupq_n_s32(0),v1=v0,v2=v0,v3=v0;const int8_t* p=wp+r*k;for(int64_t j=0;j<k;j+=16){auto z=vld1q_s8(q.data()+j);v0=vdotq_s32(v0,vld1q_s8(p+j),z);v1=vdotq_s32(v1,vld1q_s8(p+k+j),z);v2=vdotq_s32(v2,vld1q_s8(p+2*k+j),z);v3=vdotq_s32(v3,vld1q_s8(p+3*k+j),z);}dst[r]=float(vaddvq_s32(v0))*sx*s[r]+b[r];dst[r+1]=float(vaddvq_s32(v1))*sx*s[r+1]+b[r+1];dst[r+2]=float(vaddvq_s32(v2))*sx*s[r+2]+b[r+2];dst[r+3]=float(vaddvq_s32(v3))*sx*s[r+3]+b[r+3];}});
 return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("int8_linear_quant_neon(Tensor x, Tensor w, Tensor scales, Tensor bias) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("int8_linear_quant_neon",TORCH_FN(arq_neon));}

TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("int8_quant_neon_calls() -> int",[](){return quant_neon_calls.load(std::memory_order_relaxed);});}
#ifdef CPUFAST_REPLACE_INT8_QUANT_NEON
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("int8_linear",TORCH_FN(arq_neon));}
#endif
