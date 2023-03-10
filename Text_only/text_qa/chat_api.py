import os
import openai
from loguru import logger
import streamlit as st
from sacred import Experiment

st.title("Hello")
openai.api_key = ""

logger.add('chat_log.log',retention="20 days")

sys_content = st.text_input(label="你希望我扮演什么角色: ")
# sys_content = input("您希望我扮演一个什么角色: ")
logger.info("system_role: " + sys_content)
con_messages = [{"role": "system", "content": sys_content},]
i = 0
# 

system = {"role": "system", "content": sys_content}
question = st.text_input(label="请输入你的问题: ")
if st.button("Enter"):
        # question = input("请输入你的问题: ")question = st.text_input(label="请输入你的问题: ")
    logger.info("user_question: " + question)
    new_quesiton = {"role": "user", "content": question}
    con_messages.append(new_quesiton)
    result = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=con_messages
    )
    sentences = result["choices"]
    answer = sentences[0]["message"]
    answer_role = answer['role']
    answer_content = answer["content"]
    st.write("answer: ", answer_content)
    logger.info("answer: " + answer_content)
    con_messages.append({"role": answer_role, "content": answer_content})
