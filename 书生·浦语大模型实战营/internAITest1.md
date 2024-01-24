# 基于 InternLM 和 LangChain构建知识库

## 环境准备

- 安装依赖
    ```bash
    pip install langchain==0.0.292
    pip install gradio==4.4.0
    pip install chromadb==0.4.15
    pip install sentence-transformers==2.2.2
    pip install unstructured==0.10.30
    pip install markdown==3.3.7
    ```
- 用huggingface_hub下载开源词向量模型 Sentence Transformer
    ```python
    import os
    # 下载模型
    os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /root/model/sentence-transformer')
    ```
- 下载 NLTK 相关资源
    ```bash
    cd /root
    git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
    cd nltk_data
    mv packages/*  ./
    cd tokenizers
    unzip punkt.zip
    cd ../taggers
    unzip averaged_perceptron_tagger.zip
    ```
- 克隆tutorial代码
    ```bash
    cd /root/code
    git clone https://github.com/
    InternLM/tutorial
    ```
-  数据收集

    下载海人工智能实验室开源的一系列大模型工具开源仓库作为语料库来源
    ```bash
    git clone https://gitee.com/open-compass/opencompass.git
    git clone https://gitee.com/InternLM/lmdeploy.git
    git clone https://gitee.com/InternLM/xtuner.git
    git clone https://gitee.com/InternLM/InternLM-XComposer.git
    git clone https://gitee.com/InternLM/lagent.git
    git clone https://gitee.com/InternLM/InternLM.git
    ```
- 编写如下代码
    ```python
    # 首先导入所需第三方库
    from langchain.document_loaders import UnstructuredFileLoader
    from langchain.document_loaders import UnstructuredMarkdownLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from tqdm import tqdm
    import os

    # 获取文件路径函数
    def get_files(dir_path):
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, dirnames, filenames in os.walk(dir_path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    # 加载文件函数
    def get_text(dir_path):
        # args：dir_path，目标文件夹路径
        # 首先调用上文定义的函数得到目标文件路径列表
        file_lst = get_files(dir_path)
        # docs 存放加载之后的纯文本对象
        docs = []
        # 遍历所有目标文件
        for one_file in tqdm(file_lst):
            file_type = one_file.split('.')[-1]
            if file_type == 'md':
                loader = UnstructuredMarkdownLoader(one_file)
            elif file_type == 'txt':
                loader = UnstructuredFileLoader(one_file)
            else:
                # 如果是不符合条件的文件，直接跳过
                continue
            docs.extend(loader.load())
        return docs

    # 目标文件夹
    tar_dir = [
        "/root/code/InternLM",
        "/root/code/InternLM-XComposer",
        "/root/code/lagent",
        "/root/code/lmdeploy",
        "/root/code/opencompass",
        "/root/code/xtuner"
    ]

    # 加载目标文件
    docs = []
    for dir_path in tar_dir:
        docs.extend(get_text(dir_path))

    # 对文本进行分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    # 加载开源词向量模型
    embeddings = HuggingFaceEmbeddings(model_name="/root/model/sentence-transformer")

    # 构建向量数据库
    # 定义持久化路径
    persist_directory = 'data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
    # 将加载的向量数据库持久化到磁盘上
    vectordb.persist()
    ```
- 运行上述脚本，即可在本地构建已持久化的向量数据库，后续直接导入该数据库即可，无需重复构建。
    ```bash
    python demo.py
    ```
## InternLM 接入 LangChain

- 编写`LLM.py`文件，从`LangChain.llms.base.LLM `类继承一个子类，重写构造函数、_call函数，从而实现将InternLM接入到LangChain中。
    ```python
    from langchain.llms.base import LLM
    from typing import Any, List, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    class InternLM_LLM(LLM):
        # 基于本地 InternLM 自定义 LLM 类
        tokenizer : AutoTokenizer = None
        model: AutoModelForCausalLM = None

        def __init__(self, model_path :str):
            # model_path: InternLM 模型路径
            # 从本地初始化模型
            super().__init__()
            print("正在从本地加载模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
            self.model = self.model.eval()
            print("完成本地模型的加载")

        def _call(self, prompt : str, stop: Optional[List[str]] = None,
                    run_manager: Optional[CallbackManagerForLLMRun] = None,
                    **kwargs: Any):
            # 重写调用函数
            system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
            - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
            - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
            """
            
            messages = [(system_prompt, '')]
            response, history = self.model.chat(self.tokenizer, prompt , history=messages)
            return response
            
        @property
        def _llm_type(self) -> str:
            return "InternLM"
    ```
- 构建检索问答链&部署web demo

    调用一个LangChain提供的`RetrievalQA`对象，通过初始化时填入已构建的数据库和自定义 LLM 作为参数，来简便地完成检索增强问答的全流程，LangChain 会自动完成基于用户提问进行检索、获取相关文档、拼接为合适的 Prompt 并交给 LLM 问答的全部流程。

    ```python
    # 导入必要的库
    import gradio as gr
    from langchain.vectorstores import Chroma
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    import os
    from LLM import InternLM_LLM
    from langchain.prompts import PromptTemplate

    def load_chain():
        # 加载问答链
        # 定义 Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="/root/model/sentence-transformer")

        # 向量数据库持久化路径
        persist_directory = 'data_base/vector_db/chroma'

        # 加载数据库
        vectordb = Chroma(
            persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
            embedding_function=embeddings
        )

        llm = InternLM_LLM(model_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b")

        # 你可以修改这里的 prompt template 来试试不同的问答效果
        template = """请使用以下提供的上下文来回答用户的问题。如果无法从上下文中得到答案，请回答你不知道，并总是使用中文回答。
        提供的上下文：
        ···
        {context}
        ···
        用户的问题: {question}
        你给的回答:"""

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                        template=template)

        # 运行 chain
        from langchain.chains import RetrievalQA

        qa_chain = RetrievalQA.from_chain_type(llm,
                                            retriever=vectordb.as_retriever(),
                                            return_source_documents=True,
                                            chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
        
        return qa_chain

    class Model_center():
        """
        存储问答 Chain 的对象 
        """
        def __init__(self):
            self.chain = load_chain()

        def qa_chain_self_answer(self, question: str, chat_history: list = []):
            """
            调用不带历史记录的问答链进行回答
            """
            if question == None or len(question) < 1:
                return "", chat_history
            try:
                chat_history.append(
                    (question, self.chain({"query": question})["result"]))
                return "", chat_history
            except Exception as e:
                return e, chat_history


    model_center = Model_center()

    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):   
            with gr.Column(scale=15):
                gr.Markdown("""<h1><center>InternLM</center></h1>
                    <center>书生浦语</center>
                    """)
            # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=450, show_copy_button=True)
                # 创建一个文本框组件，用于输入 prompt。
                msg = gr.Textbox(label="Prompt/问题")

                with gr.Row():
                    # 创建提交按钮。
                    db_wo_his_btn = gr.Button("Chat")
                with gr.Row():
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(
                        components=[chatbot], value="Clear console")
                    
            # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                                msg, chatbot], outputs=[msg, chatbot])
            
        gr.Markdown("""提醒：<br>
        1. 初始化数据库时间可能较长，请耐心等待。
        2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
        """)
    # threads to consume the request
    gr.close_all()
    # 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
    # demo.launch(share=True, server_port=int(os.environ['PORT1']))
    # 直接启动
    demo.launch()

- 效果展示
    ![Alt text](image.png)












