from typing import Iterator

import gradio as gr
import torch

from model import get_input_token_length, run

DEFAULT_SYSTEM_PROMPT = """\
Du bist ein hilfreicher, respektvoller und ehrlicher Assistent. Antworte immer so hilfreich wie möglich. Deine Antworten sollten keinen schädlichen, unethischen, rassistischen, sexistischen, toxischen, gefährlichen oder illegalen Inhalt enthalten.

Wenn eine Frage keinen Sinn ergibt oder faktisch nicht kohärent ist, erkläre warum, anstatt etwas Unkorrektes zu antworten. Wenn du die Antwort auf eine Frage nicht weißt, teile bitte keine falschen Informationen."""
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 512
MAX_INPUT_TOKEN_LENGTH = 4000

DESCRIPTION = """
# Llama-2 13B Chat German
Dieser Space ist eine Demo für das [Llama-2-13b-chat-german](https://huggingface.co/jphme/Llama-2-13b-chat-german) Modell, welches auf Meta´s Llama2 basiert und zusätzlich mit einem Datensatz auf deutsche Konversationen und "Retrieval" fein-abgestimmt wurde.

Während das "normale" Llama2 Chat Modell große Schwierigkeiten hat, auf Deutsch zu antworten, sollte dieses Modell deutlich bessere Antworten liefer. Für einen Vergleich finden Sie [hier](https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat) eine Demo des "normalen" Llama2 Chat Modells.

Bitte beachten Sie, dass aufgrund der Größe und "Quantisierung" des Modells die sprachliche Qualität der Antworten bei längeren Texten noch deutlich hinter größeren Modellen bzw. ChatGPT&co liegt. Sollte es Interesse geben, werde ich weitere Versionen mit größeren Modellen und basierend auf besseren Llama2-Modellen veröffentlichen.
"""

LICENSE = """
<p/>
---
As a derivate work of [Llama-2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat) by Meta,
this demo is governed by the original [license](https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/USE_POLICY.md).
"""

if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'


def clear_and_save_textbox(message: str) -> tuple[str, str]:
    return '', message


def display_input(message: str,
                  history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    history.append((message, ''))
    return history


def delete_prev_fn(
        history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    try:
        message, _ = history.pop()
    except IndexError:
        message = ''
    return history, message or ''


def generate(
    message: str,
    history_with_input: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Iterator[list[tuple[str, str]]]:
    if max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise ValueError

    history = history_with_input[:-1]
    generator = run(message, history, system_prompt, max_new_tokens, temperature, top_p, top_k)
    try:
        first_response = next(generator)
        yield history + [(message, first_response)]
    except StopIteration:
        yield history + [(message, '')]
    for response in generator:
        yield history + [(message, response)]


def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
    generator = generate(message, [], DEFAULT_SYSTEM_PROMPT, 1024, 1, 0.95, 50)
    for x in generator:
        pass
    return '', x


def check_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> None:
    input_token_length = get_input_token_length(message, chat_history, system_prompt)
    if input_token_length > MAX_INPUT_TOKEN_LENGTH:
        raise gr.Error(f'Der gesamte Input-Text ist zu lang ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Bitte löschen Sie die Chat-Historie und versuchen es dann erneut.')


with gr.Blocks(css='style.css') as demo:
    with gr.Group():
        chatbot = gr.Chatbot(label='Chatbot')
        with gr.Row():
            textbox = gr.Textbox(
                container=False,
                show_label=False,
                placeholder='Geben Sie eine Nachricht ein...',
                scale=10,
            )
            submit_button = gr.Button('Abschicken',
                                      variant='primary',
                                      scale=1,
                                      min_width=0)
    with gr.Row():
        retry_button = gr.Button('🔄  Retry', variant='secondary')
        undo_button = gr.Button('↩️ Rückgängig', variant='secondary')
        clear_button = gr.Button('🗑️  Löschen', variant='secondary')

    saved_input = gr.State()

    with gr.Accordion(label='Erweiterte Einstellungen', open=False):
        system_prompt = gr.Textbox(label='System prompt',
                                   value=DEFAULT_SYSTEM_PROMPT,
                                   lines=6)
        max_new_tokens = gr.Slider(
            label='Max new tokens',
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        )
        temperature = gr.Slider(
            label='Temperatur',
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.6,
        )
        top_p = gr.Slider(
            label='Top-p (nucleus sampling)',
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.95,
        )
        top_k = gr.Slider(
            label='Top-k',
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        )

    gr.Examples(
        examples=[
            'Hallo, wie geht es dir?',
            'Was ist die Hauptstadt von Nordrhein-Westfalen?',
            'Erkläre mir die parlamentarische Demokratie in einfacher Sprache!',
            'Was ist der Sinn des Lebens?',
            "Nenne 5 Gründe, warum lokale open-source KI-Modelle wichtig sind.",
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=True,
    )

    gr.Markdown(LICENSE)

    textbox.submit(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    button_event_preprocess = submit_button.click(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    retry_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    undo_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=lambda x: x,
        inputs=[saved_input],
        outputs=textbox,
        api_name=False,
        queue=False,
    )

    clear_button.click(
        fn=lambda: ([], ''),
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    )

demo.queue(max_size=20).launch()
