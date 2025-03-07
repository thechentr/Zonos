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


def update_ui():
    """
    Dynamically show/hide UI elements based on the model's conditioners.
    We do NOT display 'language_id' or 'ctc_loss' even if they exist in the model.
    """
    model = load_model_if_needed()
    cond_names = [c.name for c in model.prefix_conditioner.conditioners]
    print("Conditioners in this model:", cond_names)

    text_update = gr.update(visible=("espeak" in cond_names))
    language_update = gr.update(visible=("espeak" in cond_names))
    speaker_audio_update = gr.update(visible=("speaker" in cond_names))
    prefix_audio_update = gr.update(visible=True)
    emotion1_update = gr.update(visible=("emotion" in cond_names))
    emotion2_update = gr.update(visible=("emotion" in cond_names))
    emotion3_update = gr.update(visible=("emotion" in cond_names))
    emotion4_update = gr.update(visible=("emotion" in cond_names))
    emotion5_update = gr.update(visible=("emotion" in cond_names))
    emotion6_update = gr.update(visible=("emotion" in cond_names))
    emotion7_update = gr.update(visible=("emotion" in cond_names))
    emotion8_update = gr.update(visible=("emotion" in cond_names))
    vq_single_slider_update = gr.update(visible=("vqscore_8" in cond_names))
    fmax_slider_update = gr.update(visible=("fmax" in cond_names))
    pitch_std_slider_update = gr.update(visible=("pitch_std" in cond_names))
    speaking_rate_slider_update = gr.update(visible=("speaking_rate" in cond_names))
    dnsmos_slider_update = gr.update(visible=("dnsmos_ovrl" in cond_names))
    speaker_noised_checkbox_update = gr.update(visible=("speaker_noised" in cond_names))
    unconditional_keys_update = gr.update(
        choices=[name for name in cond_names if name not in ("espeak", "language_id")]
    )

    return (
        text_update,
        language_update,
        speaker_audio_update,
        prefix_audio_update,
        emotion1_update,
        emotion2_update,
        emotion3_update,
        emotion4_update,
        emotion5_update,
        emotion6_update,
        emotion7_update,
        emotion8_update,
        vq_single_slider_update,
        fmax_slider_update,
        pitch_std_slider_update,
        speaking_rate_slider_update,
        dnsmos_slider_update,
        speaker_noised_checkbox_update,
        unconditional_keys_update,
    )


def generate_audio(
    text,
    language,
    speaker_audio,
    prefix_audio,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    vq_single,
    fmax,
    pitch_std,
    speaking_rate,
    dnsmos_ovrl,
    speaker_noised,
    cfg_scale,
    top_p,
    top_k,
    min_p,
    linear,
    confidence,
    quadratic,
    unconditional_keys,
    chunk_size = 100
):
    """
    Generates audio based on the provided UI parameters.
    We do NOT use language_id or ctc_loss even if the model has them.
    """
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


    print(f"text: {text}")
    print(f"language: {language}")
    if SPEAKER_EMBEDDING is not None:
        print(f"speaker: {SPEAKER_EMBEDDING.shape}")
    if emotion_tensor is not None:
        print(f"emotion: {emotion_tensor.shape}")
    if vq_tensor is not None:
        print(f"vqscore_8: {vq_tensor.shape}")
    print(f"fmax: {fmax}")
    print(f"pitch_std: {pitch_std}")
    print(f"speaking_rate: {speaking_rate}")
    print(f"dnsmos_ovrl: {dnsmos_ovrl}")
    print(f"speaker_noised: {speaker_noised_bool}")
    print(f"unconditional_keys: {unconditional_keys}")
    print(f'cfg_scale: {cfg_scale}')
    print(f'top_p, top_k, min_p, linear, confidence, quadratic: {top_p}, {top_k}, {min_p}, {linear}, {confidence}, {quadratic}')

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
    # if "hybrid" in ZonosBackbone.supported_architectures:
    #     supported_models.append("Zyphra/Zonos-v0.1-hybrid")
    # else:
    #     print(
    #         "| The current ZonosBackbone does not support the hybrid architecture, meaning only the transformer model will be available in the model selector.\n"
    #         "| This probably means the mamba-ssm library has not been installed."
    #     )

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
            prefix_audio = gr.Audio(
                value="assets/silence_100ms.wav",
                label="Optional Prefix Audio (continue from this audio)",
                type="filepath",
            )
            with gr.Column():
                speaker_audio = gr.Audio(
                    label="Optional Speaker Audio (for cloning)",
                    type="filepath",
                )
                speaker_noised_checkbox = gr.Checkbox(label="Denoise Speaker?", value=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Conditioning Parameters")
                dnsmos_slider = gr.Slider(1.0, 5.0, value=4.0, step=0.1, label="DNSMOS Overall")
                fmax_slider = gr.Slider(0, 24000, value=24000, step=1, label="Fmax (Hz)")
                vq_single_slider = gr.Slider(0.5, 0.8, 0.78, 0.01, label="VQ Score")
                pitch_std_slider = gr.Slider(0.0, 300.0, value=45.0, step=1, label="Pitch Std")
                speaking_rate_slider = gr.Slider(5.0, 30.0, value=15.0, step=0.5, label="Speaking Rate")

            with gr.Column():
                gr.Markdown("## Generation Parameters")
                cfg_scale_slider = gr.Slider(1.0, 5.0, 2.0, 0.1, label="CFG Scale")

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### NovelAi's unified sampler")
                    linear_slider = gr.Slider(-2.0, 2.0, 0.5, 0.01, label="Linear (set to 0 to disable unified sampling)", info="High values make the output less random.")
                    #Conf's theoretical range is between -2 * Quad and 0.
                    confidence_slider = gr.Slider(-2.0, 2.0, 0.40, 0.01, label="Confidence", info="Low values make random outputs more random.")
                    quadratic_slider = gr.Slider(-2.0, 2.0, 0.00, 0.01, label="Quadratic", info="High values make low probablities much lower.")
                with gr.Column():
                    gr.Markdown("### Legacy sampling")
                    top_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Top P")
                    min_k_slider = gr.Slider(0.0, 1024, 0, 1, label="Min K")
                    min_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Min P")

        with gr.Accordion("Advanced Parameters", open=False):
            gr.Markdown(
                "### Unconditional Toggles\n"
                "Checking a box will make the model ignore the corresponding conditioning value and make it unconditional.\n"
                'Practically this means the given conditioning feature will be unconstrained and "filled in automatically".'
            )
            with gr.Row():
                unconditional_keys = gr.CheckboxGroup(
                    [
                        "speaker",
                        "emotion",
                        "vqscore_8",
                        "fmax",
                        "pitch_std",
                        "speaking_rate",
                        "dnsmos_ovrl",
                        "speaker_noised",
                    ],
                    value=[],
                    label="Unconditional Keys",
                )

            gr.Markdown(
                "### Emotion Sliders\n"
                "Warning: The way these sliders work is not intuitive and may require some trial and error to get the desired effect.\n"
                "Certain configurations can cause the model to become unstable. Setting emotion to unconditional may help."
            )
            with gr.Row():
                emotion1 = gr.Slider(0.0, 1.0, 1.0, 0.05, label="Happiness")
                emotion2 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Sadness")
                emotion3 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Disgust")
                emotion4 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Fear")
            with gr.Row():
                emotion5 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Surprise")
                emotion6 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Anger")
                emotion7 = gr.Slider(0.0, 1.0, 0.1, 0.05, label="Other")
                emotion8 = gr.Slider(0.0, 1.0, 0.2, 0.05, label="Neutral")

        with gr.Column():
            generate_button = gr.Button("Generate Audio")
            output_audio = gr.Audio(label="Generated Audio", type="numpy", autoplay=True, streaming=True)


        # On page load, trigger the same UI refresh
        demo.load(
            fn=update_ui,
            inputs=[],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # Generate audio on button click
        generate_button.click(
            fn=generate_audio,
            inputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                top_p_slider,
                min_k_slider,
                min_p_slider,
                linear_slider,
                confidence_slider,
                quadratic_slider,
                unconditional_keys,
            ],
            outputs=[output_audio],
        )

    return demo


if __name__ == "__main__":
    load_model_if_needed()
    demo = build_interface()
    share = getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    demo.launch(server_name="0.0.0.0", server_port=8002, share=share)