// Fused FP32 F(4,3) transforms; source model coefficients remain unchanged.
#include <arm_neon.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>
static at::Tensor winograd_input(const at::Tensor& x) {
 TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat && x.dim()==3 && x.is_contiguous(),"Contiguous CPU FP32 streams required");
 const int64_t batch=x.size(0),channels=x.size(1),length=x.size(2),tiles=(length+1)/4;
 TORCH_CHECK(length>=3,"Stream too short");
 auto y=at::empty({6,channels,batch*tiles},x.options());const float* data=x.const_data_ptr<float>();float* out=y.mutable_data_ptr<float>();const int64_t plane=channels*batch*tiles;
 for(int64_t c=0;c<channels;++c)for(int64_t b=0;b<batch;++b) {
  const float* row=data+(b*channels+c)*length;int64_t t=0;
  for(;t+4<=tiles && t*4+17<length;t+=4) {
   auto q=vld4q_f32(row+t*4);auto a=q.val[0],v=q.val[1],z=q.val[2],d=q.val[3];
   auto e=vextq_f32(a,vdupq_n_f32(row[t*4+16]),1);auto f=vextq_f32(v,vdupq_n_f32(row[t*4+17]),1);
   float32x4_t values[6]={
    vaddq_f32(vsubq_f32(vmulq_n_f32(a,4),vmulq_n_f32(z,5)),e),
    vaddq_f32(vaddq_f32(vsubq_f32(vmulq_n_f32(v,-4),vmulq_n_f32(z,4)),d),e),
    vaddq_f32(vsubq_f32(vsubq_f32(vmulq_n_f32(v,4),vmulq_n_f32(z,4)),d),e),
    vaddq_f32(vaddq_f32(vsubq_f32(vmulq_n_f32(v,-2),z),vmulq_n_f32(d,2)),e),
    vaddq_f32(vsubq_f32(vsubq_f32(vmulq_n_f32(v,2),z),vmulq_n_f32(d,2)),e),
    vaddq_f32(vsubq_f32(vmulq_n_f32(v,4),vmulq_n_f32(d,5)),f)};
   for(int p=0;p<6;++p)vst1q_f32(out+p*plane+c*batch*tiles+b*tiles+t,values[p]);
  }
  for(;t<tiles;++t){float a[6];for(int j=0;j<6;++j)a[j]=t*4+j<length?row[t*4+j]:0;const float v[6]={4*a[0]-5*a[2]+a[4],-4*a[1]-4*a[2]+a[3]+a[4],4*a[1]-4*a[2]-a[3]+a[4],-2*a[1]-a[2]+2*a[3]+a[4],2*a[1]-a[2]-2*a[3]+a[4],4*a[1]-5*a[3]+a[5]};for(int p=0;p<6;++p)out[p*plane+c*batch*tiles+b*tiles+t]=v[p];}
 }
 return y;
}
static at::Tensor winograd_output(const at::Tensor& products,int64_t batch,int64_t dilation,int64_t length,const std::optional<at::Tensor>& bias) {
 TORCH_CHECK(products.device().is_cpu() && products.scalar_type()==at::kFloat && products.dim()==3 && products.is_contiguous() && products.size(0)==6,"Invalid Winograd products");
 TORCH_CHECK(batch>0 && dilation>0 && length>0 && products.size(2)%(batch*dilation)==0,"Invalid output geometry");
 const auto channels=products.size(1),tiles=products.size(2)/(batch*dilation),row_size=products.size(2),plane=channels*row_size;
 if(bias)TORCH_CHECK(bias->device().is_cpu() && bias->scalar_type()==at::kFloat && bias->is_contiguous() && bias->numel()==channels,"Invalid bias");
 auto y=at::empty({batch,channels,length},products.options());auto src=products.const_data_ptr<float>();auto out=y.mutable_data_ptr<float>();const float* bs=bias?bias->const_data_ptr<float>():nullptr;
 for(int64_t b=0;b<batch;++b)for(int64_t c=0;c<channels;++c) {
  int64_t first=0;
  if(dilation==1) {
   for(;first+4<=tiles && first*4+15<length;first+=4) {
    auto index=c*row_size+b*tiles+first;
    auto m0=vld1q_f32(src+index),m1=vld1q_f32(src+plane+index),m2=vld1q_f32(src+2*plane+index),m3=vld1q_f32(src+3*plane+index),m4=vld1q_f32(src+4*plane+index),m5=vld1q_f32(src+5*plane+index);
    float32x4x4_t v;
    v.val[0]=vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(m0,m1),m2),m3),m4);
    v.val[1]=vaddq_f32(vsubq_f32(m1,m2),vmulq_n_f32(vsubq_f32(m3,m4),2));
    v.val[2]=vaddq_f32(vaddq_f32(m1,m2),vmulq_n_f32(vaddq_f32(m3,m4),4));
    v.val[3]=vaddq_f32(vaddq_f32(vsubq_f32(m1,m2),vmulq_n_f32(vsubq_f32(m3,m4),8)),m5);
    if(bs){auto bias=vdupq_n_f32(bs[c]);for(int j=0;j<4;++j)v.val[j]=vaddq_f32(v.val[j],bias);}
    vst4q_f32(out+(b*channels+c)*length+first*4,v);
   }
  }
  for(int64_t t=first;t<tiles;++t)for(int64_t r=0;r<dilation;++r) {
   int64_t index=c*row_size+(b*dilation+r)*tiles+t;float m[6];for(int j=0;j<6;++j)m[j]=src[j*plane+index];
   const float v[4]={m[0]+m[1]+m[2]+m[3]+m[4],m[1]-m[2]+2*(m[3]-m[4]),m[1]+m[2]+4*(m[3]+m[4]),m[1]-m[2]+8*(m[3]-m[4])+m[5]};
   for(int j=0;j<4;++j){auto p=(t*4+j)*dilation+r;if(p<length)out[(b*channels+c)*length+p]=v[j]+(bs?bs[c]:0.f);}
  }
 }
 return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("winograd_input(Tensor x) -> Tensor");m.def("winograd_output(Tensor products, int batch, int dilation, int length, Tensor? bias) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("winograd_input",TORCH_FN(winograd_input));m.impl("winograd_output",TORCH_FN(winograd_output));}
