// Isolated W8A8 multi-row FFN prefill research. Never part of default builds.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <algorithm>
#include <cmath>

static at::Tensor pack_int8_rhs(const at::Tensor& w) {
  TORCH_CHECK(w.device().is_cpu() && w.scalar_type()==at::kChar &&
      w.is_contiguous() && w.dim()==2 && w.size(0)>0 && w.size(1)>0,
      "int8 output-major contiguous [N,K] weights required");
  const int64_t n=w.size(0), k=w.size(1), nt=(n+3)/4, kt=(k+15)/16;
  auto packed=at::empty({nt,kt,4,16},w.options());
  auto dst=packed.mutable_data_ptr<int8_t>(); const auto src=w.const_data_ptr<int8_t>();
  at::parallel_for(0,nt,1,[&](int64_t begin,int64_t end){
    for(int64_t tile=begin;tile<end;++tile) for(int64_t kb=0;kb<kt;++kb)
      for(int col=0;col<4;++col) for(int j=0;j<16;++j) {
        const int64_t row=tile*4+col, input=kb*16+j;
        dst[((tile*kt+kb)*4+col)*16+j]=(row<n && input<k)?src[row*k+input]:0;
      }
  });
  return packed;
}

static at::Tensor int8_prefill(const at::Tensor& x,const at::Tensor& w,
    const at::Tensor& scales,const at::Tensor& bias,int64_t input_features,bool relu) {
  TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && scales.device().is_cpu() && bias.device().is_cpu() &&
      x.scalar_type()==at::kFloat && w.scalar_type()==at::kChar && scales.scalar_type()==at::kFloat && bias.scalar_type()==at::kFloat &&
      x.is_contiguous() && w.is_contiguous() && scales.is_contiguous() && bias.is_contiguous() &&
      x.dim()==2 && x.size(0)>1 && x.size(1)==input_features && input_features>0 && x.size(1)<=65536 &&
      w.dim()==4 && w.size(2)==4 && w.size(3)==16 && scales.dim()==1 && bias.dim()==1 &&
      bias.numel()>0 && scales.numel()==bias.numel() &&
      w.size(0)==(bias.numel()+3)/4 && w.size(1)==(x.size(1)+15)/16,
      "multi-row FP32 X, packed int8 RHS and matching scales/bias required; no GEMV");
  const int64_t m=x.size(0),k=x.size(1),n=bias.numel(),mt=(m+3)/4,nt=(n+3)/4,kt=(k+15)/16;
  auto aq=at::empty({mt,kt,4,16},x.options().dtype(at::kChar));
  auto xs=at::empty({m},x.options()); auto xp=x.const_data_ptr<float>();
  auto ap=aq.mutable_data_ptr<int8_t>(); auto sp=xs.mutable_data_ptr<float>();
  for(int64_t j=0;j<n;++j) TORCH_CHECK(std::isfinite(scales.const_data_ptr<float>()[j]) && scales.const_data_ptr<float>()[j]>0.f && std::isfinite(bias.const_data_ptr<float>()[j]),"invalid dequant scale/bias");
  // One independently quantized activation scale per real prompt token.
  at::parallel_for(0,mt,1,[&](int64_t begin,int64_t end){
    for(int64_t t=begin;t<end;++t) {
      float inv[4]={1.f,1.f,1.f,1.f};
      for(int r=0;r<4 && t*4+r<m;++r) {
        float maximum=0.f; const auto row=xp+(t*4+r)*k;
        for(int64_t j=0;j<k;++j){TORCH_CHECK(std::isfinite(row[j]),"nonfinite activation");maximum=std::max(maximum,std::abs(row[j]));}
        const float scale=maximum>0.f?maximum/127.f:1.f;
        TORCH_CHECK(scale>0.f && std::isfinite(1.f/scale),"activation scale outside representable range");
        sp[t*4+r]=scale; inv[r]=1.f/scale;
      }
      for(int64_t kb=0;kb<kt;++kb) for(int r=0;r<4;++r) for(int j=0;j<16;++j) {
        const int64_t row=t*4+r, col=kb*16+j;
        ap[((t*kt+kb)*4+r)*16+j]=(row<m && col<k)?
          int8_t(std::max(-127,std::min(127,int(std::nearbyint(xp[row*k+col]*inv[r]))))):0;
      }
    }
  });
  auto y=at::empty({m,n},x.options()); auto dst=y.mutable_data_ptr<float>();
  auto wp=w.const_data_ptr<int8_t>(); auto sw=scales.const_data_ptr<float>(); auto bp=bias.const_data_ptr<float>();
  at::parallel_for(0,mt,1,[&](int64_t begin,int64_t end){
    for(int64_t t=begin;t<end;++t) for(int64_t output=0;output<nt;++output) {
      int32x4_t sums[4][4];
      for(int r=0;r<4;++r) for(int c=0;c<4;++c) sums[r][c]=vdupq_n_s32(0);
      for(int64_t kb=0;kb<kt;++kb) {
        // Share each four-column RHS load across four prompt rows: true GEMM.
        const auto b=wp+((output*kt+kb)*4)*16;
        const auto a=ap+((t*kt+kb)*4)*16;
        int8x16_t weights[4]={vld1q_s8(b),vld1q_s8(b+16),vld1q_s8(b+32),vld1q_s8(b+48)};
        for(int r=0;r<4;++r) {
          const auto activation=vld1q_s8(a+r*16);
          for(int c=0;c<4;++c) sums[r][c]=vdotq_s32(sums[r][c],weights[c],activation);
        }
      }
      for(int r=0;r<4 && t*4+r<m;++r) for(int c=0;c<4 && output*4+c<n;++c) {
        const int64_t col=output*4+c;
        float value=float(vaddvq_s32(sums[r][c]))*sp[t*4+r]*sw[col]+bp[col];
        if(relu && value<0.f)value=0.f;
        dst[(t*4+r)*n+col]=value;
      }
    }
  });
  return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m) {
  m.def("prefill_int8_pack_rhs(Tensor weight_q) -> Tensor");
  m.def("prefill_int8_linear(Tensor input, Tensor packed, Tensor scales, Tensor bias, int input_features, bool relu) -> Tensor");
}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m) {
  m.impl("prefill_int8_pack_rhs",TORCH_FN(pack_int8_rhs));
  m.impl("prefill_int8_linear",TORCH_FN(int8_prefill));
}
