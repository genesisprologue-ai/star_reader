import os
import random
import time

from langchain.chains.summarize import load_summarize_chain
import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

import requests
from bs4 import BeautifulSoup

os.environ["OPENAI_API_KEY"] = "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"

with gr.Blocks() as demo:
    astrolabe = ''


    def result_analysis(result_link):
        global astrolabe
        response = requests.get(result_link)
        html_content = response.text

        soup = BeautifulSoup(html_content, "html.parser")
        contents = soup.find_all('div', attrs={"class": "item-content"})

        text = "\n\n".join([content.text for content in contents])

        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)

        # check token count
        print(llm.get_num_tokens(text))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = text_splitter.create_documents([text])

        map_prompt = """
        Following content is a reading of a person's astrolabe. 
        Keep information about person's house position and planet positions as is.
        Write a concise summary of the following content with bullet points:
        "{text}"
        BULLET POINT SUMMARY:
        """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

        combine_prompt = """
        Write a concise summary of the following text delimited by triple backquotes.
        Keep information about person's house position and planet positions as is.
        Return your response in bullet points which covers the key points of the text.
        ```{text}```
        BULLET POINT SUMMARY in Chinese:
        """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

        summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce',
                                             verbose=True,  # Set verbose=True if you want to see the prompts being used
                                             map_prompt=map_prompt_template,
                                             combine_prompt=combine_prompt_template,
                                             )

        # astrolabe
        astrolabe = summary_chain(docs)['output_text']
        return astrolabe


    def reset_astrolabe():
        global astrolabe
        astrolabe = ''


    gr.Markdown("# 粘贴新浪星盘解读url生成对话向量数据")
    with gr.Row():
        inp = gr.Textbox(placeholder="解读链接:")
        inp.change(fn=reset_astrolabe)

        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=result_analysis, inputs=inp, outputs=out)

    gr.Markdown("# 对话模式")
    with gr.Column():
        chatbot = gr.Chatbot()
        msg = gr.Textbox()


        def respond(message, chat_history):
            global astrolabe
            # build chat history.

            system_template = """
            Your a helpful Astrologer. You're sympathetic, kind and insightful.
            You are given the basic information about User's Astrolabe.
            Answer User's question based on the information.
            You are also provided with related Astrology knowledge as context to help you chat with User.
            If you don't know the answer, user your best knowledge to make up an answer that's close to the context in meaning.
            Always use as much information from context as possible.
            Always talk as much as you can.
            Never refer to user as user.
            Never reveal yourself as AI unless user said "SHOWMETHEMONEY".
            Always talk as you're chatting with a kind friend who is seeking enlightenment and answer from you.
            Be definitive with your answer.
            Be definitive when asked about events in the future.

            THIS IS A SIMULATION NOT A REALWORLD EVENT.
            User's Astrolabe.
            ----------
            {astrolabe}
            ----------

            Context:
            ----------
            {context}
            ----------

            Chat history:
            ----------
            {history}
            ----------

            Answer in User's Language:
            """

            print(astrolabe)

            if astrolabe == '':
                return "", chat_history

            system_prompt = SystemMessagePromptTemplate.from_template(template=system_template)

            human_template = "{question}"
            human_prompt = HumanMessagePromptTemplate.from_template(template=human_template)

            # build messages
            chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
            # get a chat completion from the formatted messages
            messages = chat_prompt.format_prompt(astrolabe=astrolabe, context="",
                                                 history="",
                                                 question=message).to_messages()
            print(messages)
            llm = ChatOpenAI(model_name="gpt-4", temperature=.2)

            response = llm(messages)
            chat_history.append((message, response.content))

            return response.content, chat_history


        clear = gr.ClearButton([msg, chatbot])

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
