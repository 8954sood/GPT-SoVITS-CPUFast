// Isolated four-token QK scheduling; each token keeps original dimension FMA order.
// Canonical exact dim32 value scheduling is unchanged.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <cmath>
#include <vector>
#include <atomic>
static std::atomic<int64_t> head_qk4_calls{0};
static at::Tensor head_attn_qk4(const at::Tensor& q,const at::Tensor& keys,const at::Tensor& values,int64_t length,int64_t heads,const std::optional<at::Tensor>& mask){
 head_qk4_calls.fetch_add(1,std::memory_order_relaxed);
 TORCH_CHECK(q.device().is_cpu() && keys.device().is_cpu() && values.device().is_cpu() && q.scalar_type()==at::kFloat && keys.scalar_type()==at::kFloat && values.scalar_type()==at::kFloat && q.is_contiguous() && keys.is_contiguous() && values.is_contiguous() && keys.dim()==4 && keys.size(0)==1 && heads>0 && keys.size(1)==heads && values.sizes()==keys.sizes() && length>0 && length<=keys.size(2),"head layout required");
 int64_t dim=keys.size(3),maxlen=keys.size(2); TORCH_CHECK(q.numel()==heads*dim && dim%4==0,"bad dimensions");const bool* blocked=nullptr;if(mask){TORCH_CHECK(mask->is_contiguous() && mask->scalar_type()==at::kBool && (mask->numel()==length || mask->numel()==heads*length),"bad mask");blocked=mask->const_data_ptr<bool>();}
 auto y=at::empty(q.sizes(),q.options());auto qp=q.const_data_ptr<float>();auto kp=keys.const_data_ptr<float>();auto vp=values.const_data_ptr<float>();auto out=y.mutable_data_ptr<float>();float scale=1.f/std::sqrt(float(dim));
 at::parallel_for(0,heads,1,[&](int64_t first,int64_t last){std::vector<float> scores(length);for(int64_t h=first;h<last;h++){const float* k=kp+h*maxlen*dim;const float* v=vp+h*maxlen*dim;float mx=-INFINITY;int64_t tqk=0; if(dim==32){
 const auto q0=vld1q_f32(qp+h*32+0);
 const auto q1=vld1q_f32(qp+h*32+4);
 const auto q2=vld1q_f32(qp+h*32+8);
 const auto q3=vld1q_f32(qp+h*32+12);
 const auto q4=vld1q_f32(qp+h*32+16);
 const auto q5=vld1q_f32(qp+h*32+20);
 const auto q6=vld1q_f32(qp+h*32+24);
 const auto q7=vld1q_f32(qp+h*32+28);
 for(;tqk+3<length;tqk+=4){
  auto a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0;
  const float* k0=k+(tqk+0)*32;
  const float* k1=k+(tqk+1)*32;
  const float* k2=k+(tqk+2)*32;
  const float* k3=k+(tqk+3)*32;
  a0=vfmaq_f32(a0,q0,vld1q_f32(k0+0));
  a1=vfmaq_f32(a1,q0,vld1q_f32(k1+0));
  a2=vfmaq_f32(a2,q0,vld1q_f32(k2+0));
  a3=vfmaq_f32(a3,q0,vld1q_f32(k3+0));
  a0=vfmaq_f32(a0,q1,vld1q_f32(k0+4));
  a1=vfmaq_f32(a1,q1,vld1q_f32(k1+4));
  a2=vfmaq_f32(a2,q1,vld1q_f32(k2+4));
  a3=vfmaq_f32(a3,q1,vld1q_f32(k3+4));
  a0=vfmaq_f32(a0,q2,vld1q_f32(k0+8));
  a1=vfmaq_f32(a1,q2,vld1q_f32(k1+8));
  a2=vfmaq_f32(a2,q2,vld1q_f32(k2+8));
  a3=vfmaq_f32(a3,q2,vld1q_f32(k3+8));
  a0=vfmaq_f32(a0,q3,vld1q_f32(k0+12));
  a1=vfmaq_f32(a1,q3,vld1q_f32(k1+12));
  a2=vfmaq_f32(a2,q3,vld1q_f32(k2+12));
  a3=vfmaq_f32(a3,q3,vld1q_f32(k3+12));
  a0=vfmaq_f32(a0,q4,vld1q_f32(k0+16));
  a1=vfmaq_f32(a1,q4,vld1q_f32(k1+16));
  a2=vfmaq_f32(a2,q4,vld1q_f32(k2+16));
  a3=vfmaq_f32(a3,q4,vld1q_f32(k3+16));
  a0=vfmaq_f32(a0,q5,vld1q_f32(k0+20));
  a1=vfmaq_f32(a1,q5,vld1q_f32(k1+20));
  a2=vfmaq_f32(a2,q5,vld1q_f32(k2+20));
  a3=vfmaq_f32(a3,q5,vld1q_f32(k3+20));
  a0=vfmaq_f32(a0,q6,vld1q_f32(k0+24));
  a1=vfmaq_f32(a1,q6,vld1q_f32(k1+24));
  a2=vfmaq_f32(a2,q6,vld1q_f32(k2+24));
  a3=vfmaq_f32(a3,q6,vld1q_f32(k3+24));
  a0=vfmaq_f32(a0,q7,vld1q_f32(k0+28));
  a1=vfmaq_f32(a1,q7,vld1q_f32(k1+28));
  a2=vfmaq_f32(a2,q7,vld1q_f32(k2+28));
  a3=vfmaq_f32(a3,q7,vld1q_f32(k3+28));
  float z0=blocked&&blocked[(mask->numel()==length ? 0 : h*length)+tqk+0]?-INFINITY:vaddvq_f32(a0)*scale;
  scores[tqk+0]=z0;mx=std::max(mx,z0);
  float z1=blocked&&blocked[(mask->numel()==length ? 0 : h*length)+tqk+1]?-INFINITY:vaddvq_f32(a1)*scale;
  scores[tqk+1]=z1;mx=std::max(mx,z1);
  float z2=blocked&&blocked[(mask->numel()==length ? 0 : h*length)+tqk+2]?-INFINITY:vaddvq_f32(a2)*scale;
  scores[tqk+2]=z2;mx=std::max(mx,z2);
  float z3=blocked&&blocked[(mask->numel()==length ? 0 : h*length)+tqk+3]?-INFINITY:vaddvq_f32(a3)*scale;
  scores[tqk+3]=z3;mx=std::max(mx,z3);
 }
}
for(;tqk<length;tqk++){auto a=vdupq_n_f32(0);for(int64_t j=0;j<dim;j+=4)a=vfmaq_f32(a,vld1q_f32(qp+h*dim+j),vld1q_f32(k+tqk*dim+j));float z=blocked&&blocked[(mask->numel()==length ? 0 : h*length)+tqk]?-INFINITY:vaddvq_f32(a)*scale;scores[tqk]=z;mx=std::max(mx,z);}if(mx==-INFINITY){for(int64_t j=0;j<dim;j++)out[h*dim+j]=0;continue;}float sum=0;for(int64_t t=0;t<length;t++){scores[t]=std::exp(scores[t]-mx);sum+=scores[t];}float inv=1.f/sum;if(dim==32){
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
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("head_attn_qk4(Tensor q, Tensor keys, Tensor values, int length, int heads, Tensor? mask) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("head_attn_qk4",TORCH_FN(head_attn_qk4));}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("head_qk4_calls() -> int",[](){return head_qk4_calls.load(std::memory_order_relaxed);});}
#ifdef CPUFAST_REPLACE_HEAD_QK4
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("head_attn",TORCH_FN(head_attn_qk4));}
#endif
