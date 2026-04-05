import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM-135M")


def load_model():
    global tokenizer, model
    model_name = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM-135M")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


tokenizer = None
model = None


def get_logprobs(prompt: str) -> dict:
    global tokenizer, model

    if not prompt.strip():
        return {"html": "", "json": {"error": "Please enter a prompt"}}

    if tokenizer is None or model is None:
        load_model()

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        logits = outputs.logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

        token_ids = inputs["input_ids"][0]

        logprobs_data = []
        for i in range(1, len(token_ids)):
            token_id = token_ids[i].item()
            logprob = log_probs[i - 1, token_id].item()

            top_logprobs = log_probs[i - 1].topk(5)
            top_alternatives = []
            for tok_id, lp in zip(
                top_logprobs.indices.tolist(), top_logprobs.values.tolist()
            ):
                top_alternatives.append(
                    {"token": tokenizer.decode([tok_id]), "logprob": lp}
                )

            logprobs_data.append(
                {
                    "position": i,
                    "token": tokenizer.decode([token_id]),
                    "logprob": logprob,
                    "top_alternatives": top_alternatives,
                }
            )

        response_text = tokenizer.decode(token_ids[1:])

        html = build_html_visualization(response_text, logprobs_data)

        return html, {"response": response_text, "logprobs": logprobs_data}
    except Exception as e:
        return f"<div class='error'>Error: {str(e)}</div>", {"error": str(e)}


def build_html_visualization(text: str, logprobs_data: list) -> str:
    html = """
    <style>
    .token-container {
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
        padding: 20px;
    }
    .token-row {
        display: flex;
        align-items: center;
        margin: 8px 0;
        padding: 8px;
        border-radius: 6px;
        background: #f8f9fa;
        transition: all 0.2s;
    }
    .token-row:hover {
        background: #e9ecef;
        transform: translateX(4px);
    }
    .token-position {
        width: 40px;
        color: #868e96;
        font-size: 12px;
    }
    .token-text {
        flex: 1;
        padding: 6px 12px;
        background: #228be6;
        color: white;
        border-radius: 4px;
        font-weight: 500;
    }
    .token-logprob {
        width: 100px;
        text-align: right;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .logprob-high { background: #40c057; color: white; }
    .logprob-medium { background: #fab005; color: #495057; }
    .logprob-low { background: #fa5252; color: white; }
    
    .top-alternatives {
        margin-left: 20px;
        margin-top: 4px;
        padding: 8px;
        background: #f1f3f5;
        border-radius: 4px;
        font-size: 13px;
    }
    .alt-token {
        display: inline-block;
        margin: 2px 4px;
        padding: 2px 8px;
        background: white;
        border-radius: 3px;
        border: 1px solid #dee2e6;
    }
    .summary-stats {
        display: flex;
        gap: 20px;
        margin-bottom: 20px;
        padding: 15px;
        background: #e7f5ff;
        border-radius: 8px;
    }
    .stat-box {
        text-align: center;
    }
    .stat-value {
        font-size: 24px;
        font-weight: bold;
        color: #1971c2;
    }
    .stat-label {
        font-size: 12px;
        color: #868e96;
    }
    .response-text {
        padding: 15px;
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        margin-bottom: 20px;
        white-space: pre-wrap;
    }
    </style>
    <div class="token-container">
    """

    avg_logprob = (
        sum(d["logprob"] for d in logprobs_data) / len(logprobs_data)
        if logprobs_data
        else 0
    )
    max_logprob = max(d["logprob"] for d in logprobs_data) if logprobs_data else 0
    min_logprob = min(d["logprob"] for d in logprobs_data) if logprobs_data else 0

    html += f"""
    <div class="summary-stats">
        <div class="stat-box">
            <div class="stat-value">{len(logprobs_data)}</div>
            <div class="stat-label">Tokens</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{avg_logprob:.2f}</div>
            <div class="stat-label">Avg Logprob</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{max_logprob:.2f}</div>
            <div class="stat-label">Max Logprob</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{min_logprob:.2f}</div>
            <div class="stat-label">Min Logprob</div>
        </div>
    </div>
    <h3>Response:</h3>
    <div class="response-text">{text}</div>
    <h3>Token Logprobs (click to expand alternatives):</h3>
    """

    for data in logprobs_data:
        logprob = data["logprob"]
        if logprob > -1:
            logprob_class = "logprob-high"
        elif logprob > -3:
            logprob_class = "logprob-medium"
        else:
            logprob_class = "logprob-low"

        alt_html = "".join(
            [
                f"<span class='alt-token'>{alt['token']} ({alt['logprob']:.2f})</span>"
                for alt in data["top_alternatives"]
            ]
        )

        html += f"""
        <div class="token-row">
            <div class="token-position">#{data["position"]}</div>
            <div class="token-text">{data["token"]}</div>
            <div class="token-logprob {logprob_class}">{logprob:.3f}</div>
        </div>
        <div class="top-alternatives">
            <strong>Alternatives:</strong> {alt_html}
        </div>
        """

    html += "</div>"
    return html


def clear_inputs():
    return "", {"html": "", "json": {}}


with gr.Blocks(title="Logprobs Visualizer") as demo:
    gr.Markdown("# 🔢 Logprobs Visualizer")
    gr.Markdown(
        "Enter a prompt to see token-level logprobs and top alternatives from the model."
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(
                label="Prompt", placeholder="Enter your prompt here...", lines=4
            )
        with gr.Column(scale=1):
            with gr.Row():
                submit_btn = gr.Button("Generate", variant="primary")
                clear_btn = gr.Button("Clear")

    with gr.Row():
        with gr.Column():
            html_output = gr.HTML(label="Visualization")
        with gr.Column():
            json_output = gr.JSON(label="Raw Data")

    submit_btn.click(
        fn=get_logprobs, inputs=[prompt_input], outputs=[html_output, json_output]
    )

    clear_btn.click(
        fn=clear_inputs, inputs=[], outputs=[prompt_input, json_output, html_output]
    )

    gr.Markdown("---")
    gr.Markdown(
        "**Tips:** Higher logprobs indicate the model is more confident in that token. Values closer to 0 mean higher probability."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
