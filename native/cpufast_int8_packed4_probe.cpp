// Exact row4 SDOT with cache-line packed, derived INT8 weights.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <cmath>
#include <vector>
#include <atomic>
#include <mutex>
#include <unordered_map>
// A research-only bounded cache. Production adoption must pack in the model
// wrapper rather than retain arbitrary input weights in a global map.
struct PackedWeight { at::Tensor original, packed; int64_t version; };
static std::mutex packed_mutex;
static std::unordered_map<const void*,PackedWeight> packed_cache;
static std::atomic<int64_t> packed_calls{0}, packed_hits{0}, packed_misses{0};
static at::Tensor pack_weight(const at::Tensor& w) {
 return w.view({w.size(0)/4,4,w.size(1)/16,16}).permute({0,2,1,3}).contiguous();
}
static at::Tensor cached_weight(const at::Tensor& w) {
 if (w.is_inference()) return pack_weight(w); // no version counter: never reuse
 std::lock_guard<std::mutex> guard(packed_mutex);
 auto key=w.const_data_ptr<int8_t>(); auto found=packed_cache.find(key);
 if(found!=packed_cache.end() && found->second.version==w._version()
    && found->second.original.sizes()==w.sizes()) {
  packed_hits.fetch_add(1,std::memory_order_relaxed); return found->second.packed;
 }
 if(packed_cache.size()>=256) packed_cache.clear();
 auto p=pack_weight(w);
 packed_cache[key]=PackedWeight{w,p,w._version()};
 packed_misses.fetch_add(1,std::memory_order_relaxed); return p;
}
static at::Tensor arq_packed4(const at::Tensor& x,const at::Tensor& w,const at::Tensor& scales,const at::Tensor& bias){
 packed_calls.fetch_add(1,std::memory_order_relaxed);
 TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && w.scalar_type()==at::kChar && w.dim()==2 && x.scalar_type()==at::kFloat && x.dim()>0 && w.size(0)>0 && w.size(1)>0 && x.size(-1)==w.size(1) && w.size(1)==x.numel() && w.size(1)%16==0,"bad input");
 int64_t n=w.size(0),k=w.size(1); const float* a=x.const_data_ptr<float>(); float maxv=0;
 TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && scales.device().is_cpu() && bias.device().is_cpu() && scales.scalar_type()==at::kFloat && bias.scalar_type()==at::kFloat && scales.is_contiguous() && bias.is_contiguous() && scales.numel()==n && bias.numel()==n && n%4==0,"invalid int8 operands");
 for(int64_t i=0;i<k;i++){TORCH_CHECK(std::isfinite(a[i]),"non-finite input");maxv=std::max(maxv,std::abs(a[i]));} float sx=maxv>0?maxv/127.0f:1.0f; float inv=1.0f/sx;
 std::vector<int8_t> q(k); for(int64_t i=0;i<k;i++)q[i]=int8_t(std::max(-127,std::min(127,int(std::nearbyint(a[i]*inv)))));
 auto shape=x.sizes().vec(); shape.back()=n; auto y=at::empty(shape,x.options()); auto dst=y.mutable_data_ptr<float>();auto packed=cached_weight(w);auto wp=packed.const_data_ptr<int8_t>();auto s=scales.const_data_ptr<float>();auto b=bias.const_data_ptr<float>();
 TORCH_CHECK(n%4==0,"four-row geometry required");
 at::parallel_for(0,n/4,4,[&](int64_t begin,int64_t end){for(int64_t rb=begin;rb<end;rb++){int64_t r=rb*4; auto v0=vdupq_n_s32(0),v1=v0,v2=v0,v3=v0;const int8_t* p=wp+rb*k*4;for(int64_t j=0;j<k;j+=16){auto z=vld1q_s8(q.data()+j);v0=vdotq_s32(v0,vld1q_s8(p+j*4),z);v1=vdotq_s32(v1,vld1q_s8(p+j*4+16),z);v2=vdotq_s32(v2,vld1q_s8(p+j*4+32),z);v3=vdotq_s32(v3,vld1q_s8(p+j*4+48),z);}dst[r]=float(vaddvq_s32(v0))*sx*s[r]+b[r];dst[r+1]=float(vaddvq_s32(v1))*sx*s[r+1]+b[r+1];dst[r+2]=float(vaddvq_s32(v2))*sx*s[r+2]+b[r+2];dst[r+3]=float(vaddvq_s32(v3))*sx*s[r+3]+b[r+3];}});
 return y;
}
TORCH_LIBRARY_FRAGMENT(cpufast_cpp,m){
 m.def("int8_linear_packed4(Tensor x, Tensor w, Tensor scales, Tensor bias) -> Tensor");
 m.def("int8_packed4_stats() -> int[]", [](){
  std::lock_guard<std::mutex> guard(packed_mutex); int64_t bytes=0;
  for(const auto& item:packed_cache) bytes+=item.second.packed.nbytes();
  return std::vector<int64_t>{packed_calls.load(),packed_hits.load(),packed_misses.load(),int64_t(packed_cache.size()),bytes};
 });
 m.def("int8_packed4_clear() -> ()", [](){std::lock_guard<std::mutex> guard(packed_mutex);packed_cache.clear();});
}
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("int8_linear_packed4",TORCH_FN(arq_packed4));}
#ifdef CPUFAST_REPLACE_INT8_PACKED4
TORCH_LIBRARY_IMPL(cpufast_cpp,CPU,m){m.impl("int8_linear",TORCH_FN(arq_packed4));}
#endif
