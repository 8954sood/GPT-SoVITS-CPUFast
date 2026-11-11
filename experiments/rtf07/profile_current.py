import torch, time, json
from pathlib import Path
from TTS_infer_pack.TTS import TTS, TTS_Config
from cpufast_engine import optimize
from cpufast_optimizations import PackedConv1d, PhaseUpsample1d
from cpufast_fft import FFTConv1d
from cpufast_winograd import WinogradConv1d

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class JapaneseTTS(TTS):
    def init_bert_weights(self, path):
        pass


p = JapaneseTTS(
    TTS_Config(
        {
            "custom": {
                "device": "cpu",
                "is_half": False,
                "version": "v2ProPlus",
                "t2s_weights_path": "/baseline/models/s1v3.ckpt",
                "vits_weights_path": "/work/assets/v2Pro/s2Gv2ProPlus.pth",
                "cnhuhbert_base_path": "/baseline/models/chinese-hubert-base",
                "bert_base_path": "/work",
            }
        }
    )
)
r = optimize(p)
inputs = {
    "text": "今日は穏やかな天気ですね。少し休んでから、また一緒に続きを進めましょう。",
    "text_lang": "ja",
    "ref_audio_path": "/work/reference_segment.wav",
    "prompt_text": "スマートフォンからフィーチャーフォンまで、マルチデバイスに対応。",
    "prompt_lang": "ja",
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 1.0,
    "text_split_method": "cut0",
    "batch_size": 1,
    "seed": 42,
    "parallel_infer": True,
    "vits_parallel_infer": False,
    "fragment_interval": 0.3,
}
for i in range(3):
    list(r.run(inputs))
hooks = []
times = []
for name, m in p.vits_model.dec.named_modules():
    if not isinstance(m, (PackedConv1d, PhaseUpsample1d, FFTConv1d, WinogradConv1d)):
        continue

    def pre(m, a):
        m._start = time.perf_counter()

    def post(m, a, o, name=name):
        times.append(
            {
                "name": name,
                "class": type(m).__name__,
                "shape": list(a[0].shape),
                "seconds": time.perf_counter() - m._start,
            }
        )

    hooks.extend([m.register_forward_pre_hook(pre), m.register_forward_hook(post)])
list(r.run(inputs))
for h in hooks:
    h.remove()
Path("/repo/.experiments/current_layers.json").write_text(
    json.dumps(sorted(times, key=lambda a: -a["seconds"]), indent=2)
)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    list(r.run(inputs))
rows = [
    {
        "name": x.key,
        "self_cpu_us": x.self_cpu_time_total,
        "total_cpu_us": x.cpu_time_total,
        "calls": x.count,
    }
    for x in prof.key_averages()
]
Path("/repo/.experiments/current_ops.json").write_text(
    json.dumps(sorted(rows, key=lambda a: -a["self_cpu_us"]), indent=2)
)
