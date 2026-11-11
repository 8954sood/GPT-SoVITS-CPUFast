import torch, time, json, gc, statistics
from pathlib import Path
from TTS_infer_pack.TTS import TTS, TTS_Config
from cpufast_engine import optimize
from cpufast_fft import FFTConv1d, tuned_size
from cpufast_cooktoom import CookToomPhase, CookToomConv1d
from cpufast_optimizations import PhaseUpsample1d

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
engine = optimize(p)
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
capture = []


def pre(m, a, k):
    if not capture:
        capture.append(
            (
                tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in a),
                {
                    n: v.clone() if isinstance(v, torch.Tensor) else v
                    for n, v in k.items()
                },
            )
        )


h = p.vits_model.dec.register_forward_pre_hook(pre, with_kwargs=True)
for i in range(2):
    list(engine.run(inputs))
h.remove()
args, kw = capture[0]
dec = p.vits_model.dec
base_ups = list(dec.ups)
new_ups = []
for old in base_ups:
    if old.taps == 1:
        new_ups.append(old)
        continue
    w = old.weight
    raw = torch.nn.ConvTranspose1d(
        w.shape[0], w.shape[1], w.shape[2], stride=old.stride, padding=old.padding
    )
    raw.weight = old.weight
    raw.bias = old.bias
    raw.eval()
    new_ups.append(CookToomPhase(raw, m=6 if old.stride[0] >= 8 else 4, native=True))
rows = []
ref = None
with torch.no_grad():
    for label in [
        "baseline",
        "cook_phase",
        "baseline_repeat",
        "cook_phase_repeat",
        "fft_tuned_cook",
    ]:
        for i, layer in enumerate(new_ups if "cook" in label else base_ups):
            dec.ups[i] = layer
        if label == "fft_tuned_cook":
            for module in dec.modules():
                if isinstance(module, FFTConv1d):
                    module.n_fft = tuned_size(
                        module.weight.shape[1], module.weight.shape[-1], module.dilation
                    )
                    module.hop = module.n_fft - module.effective_kernel + 1
                    module._key = None
                    module._packed = None
            gc.collect()
        times = []
        cpu = []
        for i in range(5):
            t = time.perf_counter()
            c = time.process_time()
            out = dec(*args, **kw)
            cpu.append(time.process_time() - c)
            times.append(time.perf_counter() - t)
        if ref is None:
            ref = out.clone()
        delta = (out - ref).double()
        row = {
            "method": label,
            "seconds": times,
            "cpu": cpu,
            "max_abs": float(delta.abs().max()),
            "snr_db": None
            if float(delta.square().mean()) == 0
            else float(
                10 * torch.log10(ref.double().square().mean() / delta.square().mean())
            ),
        }
        rows.append(row)
        Path("/repo/.experiments/decoder_sweep.json").write_text(
            json.dumps(rows, indent=2)
        )
        print("DECODER", row, flush=True)
