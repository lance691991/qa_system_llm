import json
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import argparse

model_ckpt = "/home/ml/qa_system_llm/model/merged_pt1"

st.set_page_config(page_title="问答测试")
st.title("问答测试")


@st.cache_resource
def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_ckpt,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.generation_config = GenerationConfig.from_pretrained(
        model_ckpt
    )
    # peft_model_ckpt = "/home/ml/qa_system_llm/model/v2_1"
    # model.load_adapter(peft_model_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(
        model_ckpt,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("请输入问题")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    model, tokenizer = init_model()
    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            for response in model.chat(tokenizer, messages, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)
        clear_chat_history()
        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()