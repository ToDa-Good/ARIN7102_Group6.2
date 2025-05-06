import gradio as gr
import time
from main_predict import All_predict

custom_css = """
footer {visibility: hidden}

body {
    margin: auto !important;
    height: 100vh !important;
    width: 80% !important;
}

.container {
    height: 90vh !important;
    display: flex !important;
    flex-direction: column !important;
}

#header {
    text-align: center;
    color: white;
    border-radius: 10px;
    padding: 10px 0;
}

#header h2 {
    margin: 0 !important;
}

.message {
    padding: 5px !important;
    border-radius: 5px !important;
}

.user-message {
    border: 1px solid #bbdefb !important;
    margin-left: 20% !important;
    max-width: 600px !important;
    font-size: 14px !important;
}

.bot-message {
    border: 1px solid #eee !important;
    margin-right: 20% !important;
    max-width: 600px !important;
    font-size: 14px !important;
}

::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    background: transparent !important;
}
#chatbot {
    flex: 1 !important;
    min-height: 300px !important;
    height: auto !important;
    overflow-y: auto !important;
}
#chatbot .message {
    overflow-y: hidden !important;
}
"""


def get_response(message):
    all_predict = All_predict()
    return all_predict.getResult(message)


def chat(message, history):
    if message.strip() == "":
        return "", history

    bot_message = get_response(message)

    history.append((message, bot_message))

    for i in range(len(bot_message)):
        time.sleep(0.01)
        history[-1] = (history[-1][0], bot_message[: i + 1])
        yield history


with gr.Blocks(
    title="Online Entrepreneurship Education Chatbot",
    css=custom_css,
    theme=gr.themes.Ocean(),
) as demo:
    with gr.Column(elem_classes="container"):

        gr.HTML(
            """
        <div id="header">
            <h2>🤖 Online Entrepreneurship Education Chatbot</h2>
        </div>
        """
        )

        chatbot = gr.Chatbot(elem_id="chatbot", show_label=False)

        with gr.Row():
            msg = gr.Textbox(
                scale=4,
                placeholder="Ask about entrepreneurship...",
                container=False,
                autofocus=True,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)

    msg.submit(chat, [msg, chatbot], chatbot)
    submit_btn.click(chat, [msg, chatbot], chatbot)
    clear_btn.click(lambda: None, None, chatbot, queue=False)

    submit_btn.click(lambda: "", None, msg)
    msg.submit(lambda: "", None, msg)

if __name__ == "__main__":
    demo.queue().launch()
