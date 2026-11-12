"""Capture actual teacher contexts and rank AR/decoder deletions; no speed claims."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

from student_runtime import ar_subset


class TraceTransformer:
    def __init__(self, inner):
        self.inner = inner
        self.prefill = None
        self.steps = []

    @property
    def blocks(self):
        return self.inner.blocks

    def process_prompt(self, x, mask, *args, **kwargs):
        self.prefill = (x.detach().clone(), mask.detach().clone())
        return self.inner.process_prompt(x, mask, *args, **kwargs)

    def decode_next_token(self, x, *args, **kwargs):
        self.steps.append(x.detach().clone())
        return self.inner.decode_next_token(x, *args, **kwargs)


def full_context(trace):
    prefix, old_mask = trace.prefill
    x = torch.cat([prefix, *trace.steps], dim=1)
    n, p = x.shape[1], prefix.shape[1]
    mask = torch.ones((*old_mask.shape[:-2], n, n), dtype=torch.bool)
    mask[..., :p, :p] = old_mask
    for row in range(p, n):
        mask[..., row, : row + 1] = False
    return x, mask, p - 1


def logits_for(transformer, x, mask, start, predictor):
    hidden, _, _, _ = transformer.process_prompt(x, mask, 0, None, True)
    return predictor(hidden[:, start:]).reshape(-1, predictor.weight.shape[0])


def compare_logits(teacher, student, eos):
    log_t, log_s = F.log_softmax(teacher, -1), F.log_softmax(student, -1)
    kl = (log_t.exp() * (log_t - log_s)).sum(-1)
    return {
        "mean_token_kl": float(kl.mean()),
        "max_token_kl": float(kl.max()),
        "teacher_argmax_agreement": float(
            (teacher.argmax(-1) == student.argmax(-1)).float().mean()
        ),
        "teacher_eos_argmax_positions": (teacher.argmax(-1) == eos)
        .nonzero()
        .flatten()
        .tolist(),
        "student_eos_argmax_positions": (student.argmax(-1) == eos)
        .nonzero()
        .flatten()
        .tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("s1", "s2", "hubert", "reference"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument(
        "--languages", nargs="+", choices=("ja", "ko", "en"), default=["ja", "ko", "en"]
    )
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from TTS_infer_pack.TTS import TTS, TTS_Config

    from cpufast_engine import optimize
    from cpufast_integrity import ar_digest
    from cpufast_optimizations import parameter_digest

    class NonChineseTTS(TTS):
        def init_bert_weights(self, base_path):
            pass

    pipeline = NonChineseTTS(
        TTS_Config(
            {
                "custom": {
                    "device": "cpu",
                    "is_half": False,
                    "version": "v2ProPlus",
                    "t2s_weights_path": str(args.s1),
                    "vits_weights_path": str(args.s2),
                    "cnhuhbert_base_path": str(args.hubert),
                    "bert_base_path": ".",
                }
            }
        )
    )
    decoder = pipeline.t2s_model.model
    stock = decoder.t2s_transformer
    before = {"ar": ar_digest(stock), "vits": parameter_digest(pipeline.vits_model)}
    engine = optimize(
        pipeline,
        head_major=True,
        cook_phase=True,
        fft_layout=True,
        packed_fft=True,
        fused_finish=True,
        tap_small=True,
        ar_threads=args.threads,
        acoustic_threads=args.threads,
    )
    optimized = decoder.t2s_transformer
    generator = pipeline.vits_model.dec
    cases = {
        "ja": "今日は穏やかな天気ですね。少し休んでから、また一緒に続きを進めましょう。",
        "ko": "오늘은 날씨가 참 좋네요. 잠깐 쉬었다가 함께 이야기를 이어 가요.",
        "en": "The weather is pleasant today. Let us take a short break and continue together.",
    }
    report = {
        "scope": "Teacher-forcing deletion sensitivity on one reference; not independent quality or target RTF",
        "ar_structure": {
            "layers": len(stock.blocks),
            "hidden": decoder.model_dim,
            "heads": decoder.num_head,
            "ffn": stock.blocks[0].mlp.w1_t.shape[1],
        },
        "decoder_structure": {
            "stages": generator.num_upsamples,
            "branches_per_stage": generator.num_kernels,
            "resblocks": len(generator.resblocks),
        },
        "cases": [],
    }
    for lang in args.languages:
        trace = TraceTransformer(optimized)
        decoder.t2s_transformer = trace
        branch_outputs = {}
        hooks = []
        for i, block in enumerate(generator.resblocks):

            def hook(module, inputs, output, index=i, captured=branch_outputs):
                # Original Generator subsequently mutates branch 0's output in-place.
                captured[index] = output.detach().clone()

            hooks.append(block.register_forward_hook(hook))
        inputs = {
            "text": cases[lang],
            "text_lang": lang,
            "ref_audio_path": str(args.reference),
            "prompt_text": args.prompt_text,
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
        try:
            outputs = list(engine.run(inputs))
        finally:
            for hook in hooks:
                hook.remove()
            decoder.t2s_transformer = optimized
        if len(outputs) != 1 or trace.prefill is None:
            raise RuntimeError("This pilot requires one nonstreaming synthesis chunk")
        rate, audio = outputs[0]
        sf.write(args.output / f"{lang}_teacher.wav", audio, rate)
        x, mask, start = full_context(trace)
        with torch.no_grad():
            base_logits = logits_for(
                stock, x, mask, start, decoder.ar_predict_layer
            ).detach()
            ar_rows = []
            for remove in range(len(stock.blocks)):
                keep = [i for i in range(len(stock.blocks)) if i != remove]
                begin = time.perf_counter()
                logits = logits_for(
                    ar_subset(stock, keep), x, mask, start, decoder.ar_predict_layer
                )
                row = {
                    "removed_layer": remove,
                    "analysis_seconds": time.perf_counter() - begin,
                    **compare_logits(base_logits, logits, decoder.EOS),
                }
                ar_rows.append(row)
                print(lang, "AR deletion", remove, row["mean_token_kl"], flush=True)
            branch_rows = []
            for stage in range(generator.num_upsamples):
                values = [
                    branch_outputs[stage * generator.num_kernels + j]
                    for j in range(generator.num_kernels)
                ]
                teacher = sum(values) / len(values)
                for remove in range(len(values)):
                    candidate = sum(v for j, v in enumerate(values) if j != remove) / (
                        len(values) - 1
                    )
                    mse = float((candidate.double() - teacher.double()).square().mean())
                    scale = float(teacher.double().square().mean())
                    branch_rows.append(
                        {
                            "stage": stage,
                            "removed_branch": remove,
                            "relative_feature_mse": mse / max(scale, 1e-20),
                        }
                    )
            torch.save(
                {
                    "inputs": x.cpu(),
                    "mask": mask.cpu(),
                    "prediction_start": start,
                    "teacher_logits": base_logits.cpu(),
                    "language": lang,
                    "text": cases[lang],
                    "split": "pilot_selection_not_heldout",
                    "reference_count": 1,
                },
                args.output / f"{lang}_ar_context.pt",
            )
        report["cases"].append(
            {
                "language": lang,
                "text": cases[lang],
                "sample_rate": rate,
                "audio_seconds": len(audio) / rate,
                "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "teacher_prediction_count": base_logits.shape[0],
                "prefill_length": trace.prefill[0].shape[1],
                "ar_deletions": ar_rows,
                "decoder_branch_deletions": branch_rows,
            }
        )
        (args.output / "analysis.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        del branch_outputs, trace, x, mask, base_logits
    report["teacher_unchanged"] = before == {
        "ar": ar_digest(stock),
        "vits": parameter_digest(pipeline.vits_model),
    }
    if not report["teacher_unchanged"]:
        raise RuntimeError("Teacher tensors changed")
    (args.output / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
