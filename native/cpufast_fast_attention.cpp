// Exact per-dimension token order, eight independent value accumulators.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <cmath>
#include <vector>
static at::Tensor head_attn(const at::Tensor& q,const at::Tensor& keys,const at::Tensor& values,int64_t length,int64_t heads,const std::optional<at::Tensor>& mask){
 TORCH_CHECK(q.device().is_cpu() && keys.device().is_cpu() && values.device().is_cpu() && q.scalar_type()==at::kFloat && keys.scalar_type()==at::kFloat && values.scalar_type()==at::kFloat && q.is_contiguous() && keys.is_contiguous() && values.is_contiguous() && keys.dim()==4 && keys.size(0)==1 && heads>0 && keys.size(1)==heads && values.sizes()==keys.sizes() && length>0 && length<=keys.size(2),"head layout required");
 int64_t dim=keys.size(3),maxlen=keys.size(2); TORCH_CHECK(q.numel()==heads*dim && dim%4==0,"bad dimensions");const bool* blocked=nullptr;if(mask){TORCH_CHECK(mask->is_contiguous() && mask->scalar_type()==at::kBool && (mask->numel()==length || mask->numel()==heads*length),"bad mask");blocked=mask->const_data_ptr<bool>();}
 auto y=at::empty(q.sizes(),q.options());auto qp=q.const_data_ptr<float>();auto kp=keys.const_data_ptr<float>();auto vp=values.const_data_ptr<float>();auto out=y.mutable_data_ptr<float>();float scale=1.f/std::sqrt(float(dim));
 at::parallel_for(0,heads,1,[&](int64_t first,int64_t last){std::vector<float> scores(length);for(int64_t h=first;h<last;h++){const float* k=kp+h*maxlen*dim;const float* v=vp+h*maxlen*dim;float mx=-INFINITY;for(int64_t t=0;t<length;t++){auto a=vdupq_n_f32(0);for(int64_t j=0;j<dim;j+=4)a=vfmaq_f32(a,vld1q_f32(qp+h*dim+j),vld1q_f32(k+t*dim+j));float z=blocked&&blocked[(mask->numel()==length ? 0 : h*length)+t]?-INFINITY:vaddvq_f32(a)*scale;scores[t]=z;mx=std::max(mx,z);}if(mx==-INFINITY){for(int64_t j=0;j<dim;j++)out[h*dim+j]=0;continue;}float sum=0;for(int64_t t=0;t<length;t++){scores[t]=std::exp(scores[t]-mx);sum+=scores[t];}float inv=1.f/sum;if(dim==32){
  auto a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
  for(int64_t t=0;t<length;t++){
   const float weight=scores[t]*inv;const float* row=v+t*32;
   a0=vfmaq_n_f32(a0,vld1q_f32(row),weight);
   a1=vfmaq_n_f32(a1,vld1q_f32(row+4),weight);
   a2=vfmaq_n_f32(a2,vld1q_f32(row+8),weight);
   a3=vfmaq_n_f32(a3,vld1q_f32(row+12),weight);
   a4=vfmaq_n_f32(a4,vld1q_f32(row+16),weight);
   a5=vfmaq_n_f32(a5,vld1q_f32(row+20),weight);
   a6=vfmaq_n_f32(a6,vld1q_f32(row+24),weight);
   a7=vfmaq_n_f32(a7,vld1q_f32(row+28),weight);
  }
  auto target=out+h*32;
  vst1q_f32(target,a0);vst1q_f32(target+4,a1);vst1q_f32(target+8,a2);vst1q_f32(target+12,a3);
  vst1q_f32(target+16,a4);vst1q_f32(target+20,a5);vst1q_f32(target+24,a6);vst1q_f32(target+28,a7);
 }else{for(int64_t j=0;j<dim;j+=4){auto a=vdupq_n_f32(0);for(int64_t t=0;t<length;t++)a=vfmaq_n_f32(a,vld1q_f32(v+t*dim+j),scores[t]*inv);vst1q_f32(out+h*dim+j,a);}}}});return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("head_attn(Tensor q, Tensor keys, Tensor values, int length, int heads, Tensor? mask) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("head_attn",TORCH_FN(head_attn));}
