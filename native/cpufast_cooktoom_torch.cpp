// Generic small FP32 Cook-Toom transforms. No learned coefficient conversion.
#include <arm_neon.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>
#include <vector>
struct Term{int index;float value;};
static std::vector<std::vector<Term>> terms(const at::Tensor& matrix){
 TORCH_CHECK(matrix.device().is_cpu() && matrix.scalar_type()==at::kFloat && matrix.dim()==2 && matrix.is_contiguous(),"CPU FP32 transform required");
 std::vector<std::vector<Term>> rows(matrix.size(0));auto p=matrix.const_data_ptr<float>();
 for(int i=0;i<matrix.size(0);++i)for(int j=0;j<matrix.size(1);++j)if(p[i*matrix.size(1)+j]!=0)rows[i].push_back({j,p[i*matrix.size(1)+j]});return rows;
}
static at::Tensor cook_input(const at::Tensor& x,const at::Tensor& matrix,int64_t m){
 TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat && x.dim()==3 && x.is_contiguous() && m>0,"Contiguous CPU streams required");
 auto coefficients=terms(matrix);const auto alpha=matrix.size(0),batch=x.size(0),channels=x.size(1),length=x.size(2),r=alpha-m+1,tiles=(length-r+m)/m;
 TORCH_CHECK(matrix.size(1)==alpha && alpha<=16 && r>0 && tiles>0,"Invalid Cook-Toom geometry");
 auto y=at::empty({alpha,channels,batch*tiles},x.options());auto src=x.const_data_ptr<float>();auto dst=y.mutable_data_ptr<float>();const auto plane=channels*batch*tiles;
 for(int64_t c=0;c<channels;++c)for(int64_t b=0;b<batch;++b)for(int64_t t=0;t<tiles;t+=4){
  float32x4_t input[16];for(int j=0;j<alpha;++j){float v[4];for(int z=0;z<4;++z){auto pos=(t+z)*m+j;v[z]=t+z<tiles && pos<length?src[(b*channels+c)*length+pos]:0;}input[j]=vld1q_f32(v);}
  for(int p=0;p<alpha;++p){auto acc=vdupq_n_f32(0);for(auto a:coefficients[p])acc=vaddq_f32(acc,vmulq_n_f32(input[a.index],a.value));auto index=p*plane+c*batch*tiles+b*tiles+t;if(t+4<=tiles)vst1q_f32(dst+index,acc);else{float v[4];vst1q_f32(v,acc);for(int z=0;t+z<tiles;++z)dst[index+z]=v[z];}}
 }
 return y;
}
static at::Tensor cook_output(const at::Tensor& products,const at::Tensor& matrix,int64_t batch,int64_t dilation,int64_t length){
 TORCH_CHECK(products.device().is_cpu() && products.scalar_type()==at::kFloat && products.dim()==3 && products.is_contiguous() && batch>0 && dilation>0 && length>0,"Invalid Cook-Toom output");
 auto coefficients=terms(matrix);const auto m=matrix.size(0),alpha=products.size(0),channels=products.size(1),rows=products.size(2),tiles=rows/(batch*dilation),plane=channels*rows;
 TORCH_CHECK(matrix.size(1)==alpha && alpha<=16 && m<=16 && rows%(batch*dilation)==0,"Invalid inverse transform");
 auto y=at::empty({batch,channels,length},products.options());auto src=products.const_data_ptr<float>();auto dst=y.mutable_data_ptr<float>();
 for(int64_t b=0;b<batch;++b)for(int64_t c=0;c<channels;++c)for(int64_t d=0;d<dilation;++d)for(int64_t t=0;t<tiles;t+=4){
  auto base=c*rows+(b*dilation+d)*tiles+t;float32x4_t input[16];for(int p=0;p<alpha;++p){if(t+4<=tiles)input[p]=vld1q_f32(src+p*plane+base);else{float v[4]={0,0,0,0};for(int z=0;t+z<tiles;++z)v[z]=src[p*plane+base+z];input[p]=vld1q_f32(v);}}
  for(int j=0;j<m;++j){auto acc=vdupq_n_f32(0);for(auto a:coefficients[j])acc=vaddq_f32(acc,vmulq_n_f32(input[a.index],a.value));float v[4];vst1q_f32(v,acc);for(int z=0;z<4 && t+z<tiles;++z){auto pos=((t+z)*m+j)*dilation+d;if(pos<length)dst[(b*channels+c)*length+pos]=v[z];}}
 }
 return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){m.def("cook_input(Tensor input, Tensor matrix, int m) -> Tensor");m.def("cook_output(Tensor products, Tensor matrix, int batch, int dilation, int length) -> Tensor");}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("cook_input",TORCH_FN(cook_input));m.impl("cook_output",TORCH_FN(cook_output));}
