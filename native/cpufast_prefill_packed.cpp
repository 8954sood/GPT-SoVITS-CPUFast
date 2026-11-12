// Separate FP32 prefill research operator; no engine/default build integration.
// The included Arm ACL kernel retains its MIT copyright/license notice.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <algorithm>
#define arm_gemm cpufast_prefill_arm_gemm
#include "acl_sgemm_8x12.cpp"
#undef arm_gemm

static at::Tensor pack_rhs(const at::Tensor& w) {
  TORCH_CHECK(w.device().is_cpu() && w.scalar_type()==at::kFloat &&
      w.dim()==2 && w.is_contiguous() && w.size(0)>=4 && w.size(0)%4==0 &&
      w.size(1)>0,"FP32 contiguous [K,N] prefill weights required");
  const auto k=w.size(0), n=w.size(1), groups=(n+11)/12;
  auto packed=at::empty({groups,k,12},w.options());
  const auto input=w.const_data_ptr<float>(); auto output=packed.mutable_data_ptr<float>();
  at::parallel_for(0,groups,1,[&](int64_t begin,int64_t end) {
    for (int64_t g=begin;g<end;++g) for (int64_t j=0;j<k;++j)
      for (int c=0;c<12;++c) output[(g*k+j)*12+c]=g*12+c<n ? input[j*n+g*12+c] : 0.f;
  });
  return packed;
}

static at::Tensor prefill_linear(const at::Tensor& x,const at::Tensor& w,
                                const at::Tensor& bias,bool relu) {
  TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && bias.device().is_cpu() &&
      x.scalar_type()==at::kFloat && w.scalar_type()==at::kFloat && bias.scalar_type()==at::kFloat &&
      x.is_contiguous() && w.is_contiguous() && bias.is_contiguous() &&
      x.dim()==2 && w.dim()==3 && bias.dim()==1 && x.size(0)>1 &&
      x.size(1)>=4 && x.size(1)%4==0 && x.size(1)==w.size(1) && w.size(2)==12 &&
      bias.size(0)>0 && (bias.size(0)+11)/12==w.size(0),
      "Prefill requires FP32 multi-row X, packed RHS and bias; no GEMV fallback");
  const auto rows=x.size(0), k=x.size(1), n=bias.size(0), groups=w.size(0), tiles=(rows+7)/8;
  auto a=at::empty({tiles,k,8},x.options());
  auto ap=a.mutable_data_ptr<float>(); const auto xp=x.const_data_ptr<float>();
  at::parallel_for(0,tiles,1,[&](int64_t begin,int64_t end) {
    for (int64_t t=begin;t<end;++t) for (int64_t j=0;j<k;++j)
      for (int r=0;r<8;++r) ap[(t*k+j)*8+r]=t*8+r<rows ? xp[(t*8+r)*k+j] : 0.f;
  });
  auto y=at::empty({rows,n},x.options());
  auto yp=y.mutable_data_ptr<float>(); const auto bp=w.const_data_ptr<float>(), biasp=bias.const_data_ptr<float>();
  at::parallel_for(0,tiles,1,[&](int64_t begin,int64_t end) {
    // Bounded per-thread tile buffer; bias and FFN1 ReLU fuse into final stores.
    auto scratch=at::empty({groups,8,12},x.options()); auto cp=scratch.mutable_data_ptr<float>();
    for (int64_t t=begin;t<end;++t) {
      cpufast_prefill_arm_gemm::a64_sgemm_asimd_8x12(ap+t*k*8,bp,cp,1,int(groups),int(k));
      for (int r=0;r<8 && t*8+r<rows;++r) for (int64_t g=0;g<groups;++g)
        for (int c=0;c<12 && g*12+c<n;++c) {
          float value=cp[g*96+r*12+c]+biasp[g*12+c];
          if (relu && value<0.f) value=0.f;
          yp[(t*8+r)*n+g*12+c]=value;
        }
    }
  });
  return y;
}

TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m) {
  m.def("prefill_pack_rhs(Tensor weight_t) -> Tensor");
  m.def("prefill_packed_linear(Tensor input, Tensor packed, Tensor bias, bool relu) -> Tensor");
}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m) {
  m.impl("prefill_pack_rhs",TORCH_FN(pack_rhs));
  m.impl("prefill_packed_linear",TORCH_FN(prefill_linear));
}
