// Shape-specific FFN2 row8; exact original quantization and row4 fallback.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <cmath>
#include <vector>
#include <atomic>
static std::atomic<int64_t> ffn2_rows8_calls{0},ffn2_fallback_calls{0};
static at::Tensor arq_ffn2_rows8(const at::Tensor& x,const at::Tensor& w,const at::Tensor& scales,const at::Tensor& bias){
 TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && w.scalar_type()==at::kChar && w.dim()==2 && x.scalar_type()==at::kFloat && x.dim()>0 && w.size(0)>0 && w.size(1)>0 && x.size(-1)==w.size(1) && w.size(1)==x.numel() && w.size(1)%16==0,"bad input");
 int64_t n=w.size(0),k=w.size(1); const float* a=x.const_data_ptr<float>(); float maxv=0;
 TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && scales.device().is_cpu() && bias.device().is_cpu() && scales.scalar_type()==at::kFloat && bias.scalar_type()==at::kFloat && scales.is_contiguous() && bias.is_contiguous() && scales.numel()==n && bias.numel()==n && n%4==0,"invalid int8 operands");
 for(int64_t i=0;i<k;i++){TORCH_CHECK(std::isfinite(a[i]),"non-finite input");maxv=std::max(maxv,std::abs(a[i]));} float sx=maxv>0?maxv/127.0f:1.0f; float inv=1.0f/sx;
 std::vector<int8_t> q(k); for(int64_t i=0;i<k;i++)q[i]=int8_t(std::max(-127,std::min(127,int(std::nearbyint(a[i]*inv)))));
 auto shape=x.sizes().vec(); shape.back()=n; auto y=at::empty(shape,x.options()); auto dst=y.mutable_data_ptr<float>();auto wp=w.const_data_ptr<int8_t>();auto s=scales.const_data_ptr<float>();auto b=bias.const_data_ptr<float>();
 TORCH_CHECK(n%4==0,"four-row geometry required");
 if(n==512 && k==2048){
  ffn2_rows8_calls.fetch_add(1,std::memory_order_relaxed);
 at::parallel_for(0,n/8,4,[&](int64_t begin,int64_t end){
  for(int64_t rb=begin;rb<end;++rb){
   const int64_t r=rb*8; const int8_t* p=wp+r*k;
   auto v0=vdupq_n_s32(0),v1=v0,v2=v0,v3=v0,v4=v0,v5=v0,v6=v0,v7=v0;
   for(int64_t j=0;j<k;j+=16){
    auto z=vld1q_s8(q.data()+j);
    v0=vdotq_s32(v0,vld1q_s8(p+j),z);
    v1=vdotq_s32(v1,vld1q_s8(p+k+j),z);
    v2=vdotq_s32(v2,vld1q_s8(p+2*k+j),z);
    v3=vdotq_s32(v3,vld1q_s8(p+3*k+j),z);
    v4=vdotq_s32(v4,vld1q_s8(p+4*k+j),z);
    v5=vdotq_s32(v5,vld1q_s8(p+5*k+j),z);
    v6=vdotq_s32(v6,vld1q_s8(p+6*k+j),z);
    v7=vdotq_s32(v7,vld1q_s8(p+7*k+j),z);
   }
   dst[r]=float(vaddvq_s32(v0))*sx*s[r]+b[r];
   dst[r+1]=float(vaddvq_s32(v1))*sx*s[r+1]+b[r+1];
   dst[r+2]=float(vaddvq_s32(v2))*sx*s[r+2]+b[r+2];
   dst[r+3]=float(vaddvq_s32(v3))*sx*s[r+3]+b[r+3];
   dst[r+4]=float(vaddvq_s32(v4))*sx*s[r+4]+b[r+4];
   dst[r+5]=float(vaddvq_s32(v5))*sx*s[r+5]+b[r+5];
   dst[r+6]=float(vaddvq_s32(v6))*sx*s[r+6]+b[r+6];
   dst[r+7]=float(vaddvq_s32(v7))*sx*s[r+7]+b[r+7];
  }
 });
 }else{
  ffn2_fallback_calls.fetch_add(1,std::memory_order_relaxed);
 at::parallel_for(0,n/4,4,[&](int64_t begin,int64_t end){for(int64_t rb=begin;rb<end;rb++){int64_t r=rb*4; auto v0=vdupq_n_s32(0),v1=v0,v2=v0,v3=v0;const int8_t* p=wp+r*k;for(int64_t j=0;j<k;j+=16){auto z=vld1q_s8(q.data()+j);v0=vdotq_s32(v0,vld1q_s8(p+j),z);v1=vdotq_s32(v1,vld1q_s8(p+k+j),z);v2=vdotq_s32(v2,vld1q_s8(p+2*k+j),z);v3=vdotq_s32(v3,vld1q_s8(p+3*k+j),z);}dst[r]=float(vaddvq_s32(v0))*sx*s[r]+b[r];dst[r+1]=float(vaddvq_s32(v1))*sx*s[r+1]+b[r+1];dst[r+2]=float(vaddvq_s32(v2))*sx*s[r+2]+b[r+2];dst[r+3]=float(vaddvq_s32(v3))*sx*s[r+3]+b[r+3];}});
 }

 return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("int8_linear_ffn2_rows8(Tensor x, Tensor w, Tensor scales, Tensor bias) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("int8_linear_ffn2_rows8",TORCH_FN(arq_ffn2_rows8));}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){
 m.def("int8_ffn2_rows8_calls() -> int",[](){return ffn2_rows8_calls.load(std::memory_order_relaxed);});
 m.def("int8_ffn2_fallback_calls() -> int",[](){return ffn2_fallback_calls.load(std::memory_order_relaxed);});
}
#ifdef CPUFAST_REPLACE_INT8_FFN2_ROWS8
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("int8_linear",TORCH_FN(arq_ffn2_rows8));}
#endif
