#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
static at::Tensor hgemm(const at::Tensor& x,const at::Tensor& w){
 TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && x.scalar_type()==at::kHalf && w.scalar_type()==at::kHalf && x.is_contiguous() && w.is_contiguous() && x.dim()==4 && w.dim()==4 && x.size(0)==w.size(0) && x.size(2)==w.size(2) && x.size(3)==8 && w.size(3)==12,"invalid geometry");
 int64_t nf=x.size(0),m=x.size(1),n=w.size(1),k=x.size(2);auto y=at::empty({nf,m,n,8,12},x.options().dtype(at::kFloat));auto xp=reinterpret_cast<const float16_t*>(x.const_data_ptr<at::Half>());auto wp=reinterpret_cast<const float16_t*>(w.const_data_ptr<at::Half>());auto out=y.mutable_data_ptr<float>();
 at::parallel_for(0,nf*m,1,[&](int64_t begin,int64_t end){for(int64_t task=begin;task<end;task++){int64_t f=task/m,t=task%m;auto a=xp+(f*m+t)*k*8;for(int64_t g=0;g<n;g++){auto b=wp+(f*n+g)*k*12;float16x8_t acc[12];for(int c=0;c<12;c++)acc[c]=vdupq_n_f16(0);for(int64_t j=0;j<k;j++){auto z=vld1q_f16(a+j*8);for(int c=0;c<12;c++)acc[c]=vfmaq_n_f16(acc[c],z,b[j*12+c]);}alignas(16) float16_t tmp[12][8];for(int c=0;c<12;c++)vst1q_f16(tmp[c],acc[c]);auto p=out+(task*n+g)*96;for(int r=0;r<8;r++)for(int c=0;c<12;c++)p[r*12+c]=float(tmp[c][r]);}}});return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("fft_hgemm(Tensor x, Tensor w) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("fft_hgemm",TORCH_FN(hgemm));}
