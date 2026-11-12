"""Collect CPU-only teacher contexts and decoder latents using V5-rendered voices."""

import argparse
import json
from pathlib import Path

import torch

from experiments.student.analyze_teacher import (
    TraceTransformer,
    full_context,
    logits_for,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("s1", "s2", "hubert", "voices", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument(
        "--texts", type=Path, help="Explicit language/train/validation text corpus"
    )
    parser.add_argument("--speakers", nargs="+", help="Explicit source speaker IDs")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh output required")
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
    # Raw teacher decoder remains separate from all optimized execution modules.
    torch.save(
        {"state_dict": pipeline.vits_model.dec.state_dict()},
        args.output / "teacher_decoder.pt",
    )
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
    )
    optimized = decoder.t2s_transformer
    voices = json.loads(args.voices.read_text())["voices"]
    if args.speakers:
        voices = [v for v in voices if v["speaker"] in args.speakers]
        if {v["speaker"] for v in voices} != set(args.speakers):
            raise ValueError("Requested source speaker not found")
    texts = {
        "ja": [
            "窓を開けると、遠くから鳥の声が聞こえてきました。",
            "明日の朝は少し早く起きて、駅までゆっくり歩きます。",
        ],
        "ko": [
            "창문을 여니 멀리서 새가 지저귀는 소리가 들렸어요.",
            "내일 아침에는 조금 일찍 일어나서 역까지 천천히 걸어갈게요.",
        ],
        "en": [
            "When I opened the window, I could hear birds singing in the distance.",
            "Tomorrow morning I will get up early and walk slowly to the station.",
        ],
    }
    manifest = {
        "scope": "Small CPU distillation pilot; V5 synthetic voice conditioning, V2ProPlus token/latent teacher",
        "train": [],
        "validation": [],
        "records": [],
        "collection_complete": False,
    }
    if args.texts:
        selected = json.loads(args.texts.read_text())
        texts = {}
        for lang in ("ja", "ko", "en"):
            train, validation = selected[lang]["train"], selected[lang]["validation"]
            if not train or not validation or set(train) & set(validation):
                raise ValueError("Nonempty disjoint text splits required per language")
            texts[lang] = [("train", t) for t in train] + [
                ("validation", t) for t in validation
            ]
    else:
        texts = {
            lang: [("train" if i == 0 else "validation", t) for i, t in enumerate(rows)]
            for lang, rows in texts.items()
        }
    for voice_idx, voice in enumerate(voices):
        for lang, utterances in texts.items():
            for text_idx, (split, text) in enumerate(utterances):
                key = f"voice{voice_idx}_{lang}_{text_idx}"
                trace = TraceTransformer(optimized)
                decoder.t2s_transformer = trace
                capture = {}

                def pre(module, inputs, kwargs, captured=capture):
                    captured["latent"] = inputs[0].detach().clone()
                    captured["conditioning"] = kwargs["g"].detach().clone()

                def post(module, inputs, output, captured=capture):
                    captured["waveform"] = output.detach().clone()

                h1 = pipeline.vits_model.dec.register_forward_pre_hook(
                    pre, with_kwargs=True
                )
                h2 = pipeline.vits_model.dec.register_forward_hook(post)
                request = {
                    "text": text,
                    "text_lang": lang,
                    "ref_audio_path": str(args.voices.parent / voice["path"]),
                    "prompt_text": voice["text"],
                    "prompt_lang": voice["language"],
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
                    outputs = list(engine.run(request))
                finally:
                    h1.remove()
                    h2.remove()
                    decoder.t2s_transformer = optimized
                if len(outputs) != 1 or not capture:
                    raise RuntimeError("One chunk required")
                x, mask, start = full_context(trace)
                with torch.no_grad():
                    logits = logits_for(
                        stock, x, mask, start, decoder.ar_predict_layer
                    ).detach()
                context = {
                    "inputs": x.cpu(),
                    "mask": mask.cpu(),
                    "prediction_start": start,
                    "teacher_logits": logits.cpu(),
                    "language": lang,
                    "text": text,
                    "speaker": voice["speaker"],
                    "split": split,
                }
                context_name = f"{key}_ar.pt"
                torch.save(context, args.output / context_name)
                capture.update(
                    {
                        "language": lang,
                        "text": text,
                        "speaker": voice["speaker"],
                        "split": split,
                    }
                )
                latent_name = f"{key}_decoder.pt"
                torch.save(capture, args.output / latent_name)
                manifest[split].append(context_name)
                manifest["records"].append(
                    {
                        "id": key,
                        "split": split,
                        "ar_context": context_name,
                        "decoder_context": latent_name,
                        "language": lang,
                        "speaker": voice["speaker"],
                        "voice_sha256": voice["waveform_sha256"],
                        "sample_rate": outputs[0][0],
                        "audio_seconds": len(outputs[0][1]) / outputs[0][0],
                    }
                )
                (args.output / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
                )
                print(key, split, "captured", flush=True)
    manifest["teacher_unchanged"] = before == {
        "ar": ar_digest(stock),
        "vits": parameter_digest(pipeline.vits_model),
    }
    if not manifest["teacher_unchanged"]:
        raise RuntimeError("Teacher altered")
    manifest["collection_complete"] = True
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
