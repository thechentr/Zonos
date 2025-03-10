import torch
import torchaudio
import gradio as gr
from os import getenv
import os

from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device
from timer import Timer
import numpy as np
import regex

import logging

logging.basicConfig(level=logging.INFO)

CURRENT_MODEL_TYPE = "Zyphra/Zonos-v0.1-transformer"
CURRENT_MODEL = None

wav, sr = torchaudio.load(os.path.join("assets", "voice.wav"))

SPEAKER_AUDIO_PATH = os.path.join("assets", "voice.wav")
SPEAKER_EMBEDDING = None


def load_model_if_needed():
    global CURRENT_MODEL, SPEAKER_EMBEDDING
    if CURRENT_MODEL is None:
        print(f"Loading {CURRENT_MODEL_TYPE} model...")
        CURRENT_MODEL = Zonos.from_pretrained(CURRENT_MODEL_TYPE, device=device)
        CURRENT_MODEL.requires_grad_(False).eval()
        print(f"{CURRENT_MODEL_TYPE} model loaded successfully!")

    wav, sr = torchaudio.load(SPEAKER_AUDIO_PATH)
    SPEAKER_EMBEDDING = CURRENT_MODEL.make_speaker_embedding(wav, sr)
    SPEAKER_EMBEDDING = SPEAKER_EMBEDDING.to(device, dtype=torch.bfloat16)
    return CURRENT_MODEL



def generate_audio(
    text,
    language,
    speaker_audio,
    prefix_audio=None,
    e1=1,
    e2=0.05,
    e3=0.05,
    e4=0.05,
    e5=0.05,
    e6=0.05,
    e7=0.1,
    e8=0.2,
    vq_single=0.78,
    fmax=24000,
    pitch_std=45,
    speaking_rate=15,
    dnsmos_ovrl=4,
    speaker_noised=False,
    cfg_scale=2,
    top_p=0,
    top_k=0,
    min_p=0,
    linear=0.5,
    confidence=0.4,
    quadratic=0,
    unconditional_keys=['emotion'],
    chunk_size=100,
):
    """
    Generates audio based on the provided UI parameters.
    We do NOT use language_id or ctc_loss even if the model has them.
    """
    logging.info(f"text: {text}")
    logging.info(f"language: {language}")
    logging.info(f"speaker_audio: {speaker_audio}")
    logging.info(f"prefix_audio: {prefix_audio}")
    logging.info(f"e1: {e1}, e2: {e2}, e3: {e3}, e4: {e4}, e5: {e5}, e6: {e6}, e7: {e7}, e8: {e8}")
    logging.info(f"vq_single: {vq_single}")
    logging.info(f"fmax: {fmax}, pitch_std: {pitch_std}, speaking_rate: {speaking_rate}")
    logging.info(f"dnsmos_ovrl: {dnsmos_ovrl}, speaker_noised: {speaker_noised}")
    logging.info(f"cfg_scale: {cfg_scale}, top_p: {top_p}, top_k: {top_k}, min_p: {min_p}")
    logging.info(f"linear: {linear}, confidence: {confidence}, quadratic: {quadratic}")
    logging.info(f"unconditional_keys: {unconditional_keys}, chunk_size: {chunk_size}")

    with Timer('load model'):
        selected_model = load_model_if_needed()

    torch.manual_seed(123)

    speaker_noised_bool = bool(speaker_noised)
    fmax = float(fmax)
    pitch_std = float(pitch_std)
    speaking_rate = float(speaking_rate)
    dnsmos_ovrl = float(dnsmos_ovrl)
    cfg_scale = float(cfg_scale)
    top_p = float(top_p)
    top_k = int(top_k)
    min_p = float(min_p)
    linear = float(linear)
    confidence = float(confidence)
    quadratic = float(quadratic)
    max_new_tokens = 86 * 30

    # This is a bit ew, but works for now.
    global SPEAKER_AUDIO_PATH, SPEAKER_EMBEDDING

    with Timer('speaker_audio'):
        if speaker_audio is not None and SPEAKER_AUDIO_PATH != speaker_audio:
            print("Computed speaker embedding")
            wav, sr = torchaudio.load(speaker_audio)
            SPEAKER_EMBEDDING = selected_model.make_speaker_embedding(wav, sr)
            SPEAKER_EMBEDDING = SPEAKER_EMBEDDING.to(device, dtype=torch.bfloat16)
            SPEAKER_AUDIO_PATH = speaker_audio

    with Timer('prefix_audio'):
        audio_prefix_codes = None
        if prefix_audio is not None:
            wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
            wav_prefix = wav_prefix.mean(0, keepdim=True)
            wav_prefix = selected_model.autoencoder.preprocess(wav_prefix, sr_prefix)
            wav_prefix = wav_prefix.to(device, dtype=torch.float32)
            audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))

    emotion_tensor = torch.tensor(list(map(float, [e1, e2, e3, e4, e5, e6, e7, e8])), device=device)

    vq_val = float(vq_single)
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

    cond_dict = make_cond_dict(
        text=text,
        language=language,
        speaker=SPEAKER_EMBEDDING,
        emotion=emotion_tensor,
        vqscore_8=vq_tensor,
        fmax=fmax,
        pitch_std=pitch_std,
        speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised_bool,
        device=device,
        unconditional_keys=unconditional_keys,
    )
    conditioning = selected_model.prepare_conditioning(cond_dict)

    # 用于存储累计的 token 列表，每个 token 的形状假设为 [batch, num_codebooks]
    token_generator = selected_model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        batch_size=1,
        sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence, quad=quadratic),
    )

    accumulated_tokens = []
    sr_out = selected_model.autoencoder.sampling_rate

    for token in token_generator:
        accumulated_tokens.append(token)
        # 每当累计的 token 数达到 chunk_size 时，组合并解码
        if len(accumulated_tokens) == chunk_size:
            # 将 list 中的 token 沿新的最后一维堆叠，形状为 [batch, num_codebooks, chunk_size]
            chunk_codes = torch.stack(accumulated_tokens, dim=-1)
            # 调用 autoencoder 进行解码，返回音频 tensor（假设形状为 [batch, 1, samples]）
            wav_chunk = selected_model.autoencoder.decode(chunk_codes).cpu().detach()
            # 若需要对输出进行 squeeze 处理（例如去除 batch 或 channel 维度），可以：
            if wav_chunk.dim() == 2 and wav_chunk.size(0) > 1:
                wav_chunk = wav_chunk[0:1, :]
            wav_chunk = (wav_chunk.squeeze().numpy() * 32767).astype('int16')
            # yield 当前 chunk 解码后的音频片段
            yield (sr_out, wav_chunk)

            # 清空累积 token 列表，准备下一组
            accumulated_tokens = []
        
    # 若循环结束后还有不足 chunk_size 的 token，仍然解码输出
    if accumulated_tokens:
        chunk_codes = torch.stack(accumulated_tokens, dim=-1)
        wav_chunk = selected_model.autoencoder.decode(chunk_codes).cpu().detach()
        if wav_chunk.dim() == 2 and wav_chunk.size(0) > 1:
            wav_chunk = wav_chunk[0:1, :]
        wav_chunk = (wav_chunk.squeeze().numpy() * 32767).astype('int16')
        yield (sr_out, wav_chunk)


def build_interface():

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Text to Synthesize",
                    value="Hi, I am Kai from FlashIntel. I am reaching out because we found that you have visited our website recently. Are you available for a quick chat?",
                    lines=4,
                    max_length=500,  # approximately
                )
                language = gr.Dropdown(
                    choices=supported_language_codes,
                    value="en-us",
                    label="Language Code",
                    info="Select a language code.",
                )
            speaker_audio = gr.Audio(
                value="assets/voice.wav",
                label="Optional Speaker Audio (for cloning)",
                type="filepath",
            )

        with gr.Column():
            generate_button = gr.Button("Generate Audio")
            output_audio = gr.Audio(label="Generated Audio", type="numpy", autoplay=True, streaming=True)


        # Generate audio on button click
        generate_button.click(
            fn=generate_audio,
            inputs=[
                text,
                language,
                speaker_audio,
            ],
            outputs=[output_audio],
        )

    return demo


if __name__ == "__main__":
    load_model_if_needed()
    demo = build_interface()
    share = getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    demo.launch(server_name="0.0.0.0", server_port=8002, share=share)