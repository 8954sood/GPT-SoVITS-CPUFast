"""Free-generation listening comparison for explicit, unapproved students."""

import argparse
import functools
import hashlib
import json
import time
from pathlib import Path

import soundfile as sf
import torch

from student_runtime import StudentGenerator


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("s1", "s2", "hubert", "reference", "spec", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--decoder-student", type=Path)
    parser.add_argument(
        "--export-decoder",
        action="store_true",
        help="Also save a decoder copy; normally omitted to avoid duplicate large model files",
    )
    parser.add_argument(
        "--ar-fp32-cache",
        action="store_true",
        help="Preserve general FP32 posttraining coefficients in native AR cache",
    )
    parser.add_argument("--variant", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument(
        "--languages", nargs="+", choices=("ja", "ko", "en"), default=["ja", "ko", "en"]
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh output directory required")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from TTS_infer_pack.TTS import TTS, TTS_Config

    from cpufast_engine import optimize
    from cpufast_integrity import ar_digest
    from cpufast_optimizations import parameter_digest

    class NonChineseTTS(TTS):
        def init_bert_weights(self, base_path):
            pass

    file_before = {str(p): digest(p) for p in (args.s1, args.s2)}
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
    spec = json.loads(args.spec.read_text())
    if args.variant in ("C", "D"):
        pipeline.vits_model.dec = StudentGenerator(
            pipeline.vits_model.dec,
            spec["decoder_branches"],
            channel_keep=spec.get("decoder_channel_keep"),
        )
        if args.decoder_student is not None:
            trained = torch.load(
                args.decoder_student, map_location="cpu", weights_only=True
            )
            if trained["student_spec"]["decoder_branches"] != spec["decoder_branches"]:
                raise ValueError("Decoder checkpoint/spec mismatch")
            pipeline.vits_model.dec.load_state_dict(trained["state_dict"], strict=True)
        # Export only when explicitly requested; the authoritative input is kept.
        if args.export_decoder:
            with (args.output / "decoder_student.pt").open("xb") as source:
                torch.save(
                    {
                        "teacher_s2_sha256": file_before[str(args.s2)],
                        "student_spec": spec,
                        "state_dict": pipeline.vits_model.dec.state_dict(),
                        "quality_approved": False,
                    },
                    source,
                )
    decoder = pipeline.t2s_model.model
    before = {
        "ar": ar_digest(decoder.t2s_transformer),
        "vits": parameter_digest(pipeline.vits_model),
    }
    engine = optimize(
        pipeline,
        head_major=True,
        cook_phase=True,
        fft_layout=True,
        packed_fft=True,
        fused_finish=True,
        tap_small=True,
        ar_threads=2,
        acoustic_threads=2,
        ar_fp32_cache=args.ar_fp32_cache,
        ar_prefetch=0 if args.ar_fp32_cache else 256,
    )
    semantic = []
    original = decoder.infer_panel_batch_infer

    @functools.wraps(original)
    def infer(*a, **kw):
        result = original(*a, **kw)
        for token, index in zip(*result):
            semantic.append(
                {
                    "tokens": int(index),
                    "sha256": hashlib.sha256(
                        token[-int(index) :].detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                }
            )
        return result

    decoder.infer_panel_batch_infer = infer
    cases = {
        "ja": "今日は穏やかな天気ですね。少し休んでから、また一緒に続きを進めましょう。",
        "ko": "오늘은 날씨가 참 좋네요. 잠깐 쉬었다가 함께 이야기를 이어 가요.",
        "en": "The weather is pleasant today. Let us take a short break and continue together.",
    }
    report = {
        "scope": "Local free-generation listening smoke; not N1 performance",
        "variant": args.variant,
        "ar_cache_storage": "FP32" if args.ar_fp32_cache else "exact FP16",
        "quality_approved": False,
        "ar_layers": decoder.num_layers,
        "decoder_branches": spec["decoder_branches"]
        if args.variant in ("C", "D")
        else [[0, 1, 2]] * 5,
        "decoder_channel_widths": [len(k) for k in spec["decoder_channel_keep"]]
        if spec.get("decoder_channel_keep") is not None
        else None,
        "cases": [],
    }
    for lang in args.languages:
        semantic.clear()
        request = {
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
        begin = time.perf_counter()
        outputs = list(engine.run(request))
        if len(outputs) != 1:
            raise RuntimeError("Expected one nonstreaming output")
        rate, audio = outputs[0]
        if (
            rate != 32000
            or len(audio) == 0
            or not torch.isfinite(torch.as_tensor(audio)).all()
        ):
            raise RuntimeError("Expected finite nonempty original-rate V2ProPlus audio")
        sf.write(args.output / f"{lang}_{args.variant}.wav", audio, rate)
        report["cases"].append(
            {
                "language": lang,
                "text": cases[lang],
                "sample_rate": rate,
                "audio_seconds": len(audio) / rate,
                "smoke_elapsed_seconds": time.perf_counter() - begin,
                "semantic": list(semantic),
                "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
            }
        )
        (args.output / "comparison.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    after = {
        "ar": ar_digest(decoder.t2s_transformer),
        "vits": parameter_digest(pipeline.vits_model),
    }
    report["student_parameters_unchanged_during_inference"] = before == after
    report["checkpoint_files_unchanged"] = file_before == {
        str(p): digest(p) for p in (args.s1, args.s2)
    }
    if not all(
        report[k]
        for k in (
            "student_parameters_unchanged_during_inference",
            "checkpoint_files_unchanged",
        )
    ):
        raise RuntimeError("Parameter/file integrity failure")
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
