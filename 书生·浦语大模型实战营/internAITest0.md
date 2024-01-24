# 书生·浦语大模型 0

### 环境配置&hello world

- bash 进入 conda 环境
- 拷贝本地的 pytorch 环境
- 键入以下代码
  ```bash
  conda create --name internlm-demo --clone=/root/share/conda_envs/internlm-base
  ```
- 激活环境
  ```bash
  conda activate internlm-demo
  ```
- 升级pip
  ``` python
  python -m pip install --upgrade pip
  pip install modelscope==1.9.5
  pip install transformers==4.35.2
  pip install streamlit==1.24.0
  pip install sentencepiece==0.1.99
  pip install accelerate==0.24.1
  ```
- 复制模型
  ```bash
  mkdir -p /root/model/Shanghai_AI_Laboratory
  cp -r /root/share/temp/model_repos/internlm-chat-7b /root/model/Shanghai_AI_Laboratory
  ```
- 准备代码，下载InternLM
  ```bash
  cd /root/code
  git clone https://gitee.com/internlm/InternLM.git
  ```
- 修改``code/InternLM/web_demo.py``为当前模型路径
- 新建一个 demo 文件，填入如下代码
  ```python
  import torch
  from transformers import AutoTokenizer, AutoModelForCausalLM


  model_name_or_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b"

  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
  model = model.eval()

  system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
  - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
  - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
  """

  messages = [(system_prompt, '')]

  print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

  while True:
      input_text = input("User  >>> ")
      input_text = input_text.replace(' ', '')
      if input_text == "exit":
          break
      response, history = model.chat(tokenizer, input_text, history=messages)
      messages.append((input_text, response))
      print(f"robot >>> {response}")
  ``` 
- 运行 demo 文件
  ```bash
  python /root/code/InternLM/cli_demo.py
  ```
- 加载完毕后可以开始对话
  ![命令行与模型对话](assets/image-1.png)
- web demo
  - 配置本地端口，在本地终端运行
    ```bash
    ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 34104
    ```
  - 运行 web demo
    ```bash
      cd /root/code/InternLM
      streamlit run web_demo.py --server.address 127.0.0.1 --server.port 6006
    ```
    ![web demo](assets/image-3.png)
- 300 字小故事作业
  ![小故事](assets/image-4.png)

### Lagent Demo

- 安装 Lagent
  - 下载 Lagent
    ```bash
    cd /root/code
    git clone https://gitee.com/internlm/lagent.git
    ```
  - 切换到教程版本
    - ```bash
      cd /root/code/lagent
      git checkout 511b03889010c4811b1701abb153e02b8e94fb5e
      ```
  - 安装源码
    ```bash
    pip install -e .
    ```
  - 替换 example 代码
    ```python
    import copy
    import os

    import streamlit as st
    from streamlit.logger import get_logger

    from lagent.actions import ActionExecutor, GoogleSearch, PythonInterpreter
    from lagent.agents.react import ReAct
    from lagent.llms import GPTAPI
    from lagent.llms.huggingface import HFTransformerCasualLM


    class SessionState:

        def init_state(self):
            """Initialize session state variables."""
            st.session_state['assistant'] = []
            st.session_state['user'] = []

            #action_list = [PythonInterpreter(), GoogleSearch()]
            action_list = [PythonInterpreter()]
            st.session_state['plugin_map'] = {
                action.name: action
                for action in action_list
            }
            st.session_state['model_map'] = {}
            st.session_state['model_selected'] = None
            st.session_state['plugin_actions'] = set()

        def clear_state(self):
            """Clear the existing session state."""
            st.session_state['assistant'] = []
            st.session_state['user'] = []
            st.session_state['model_selected'] = None
            if 'chatbot' in st.session_state:
                st.session_state['chatbot']._session_history = []


    class StreamlitUI:

        def __init__(self, session_state: SessionState):
            self.init_streamlit()
            self.session_state = session_state

        def init_streamlit(self):
            """Initialize Streamlit's UI settings."""
            st.set_page_config(
                layout='wide',
                page_title='lagent-web',
                page_icon='./docs/imgs/lagent_icon.png')
            # st.header(':robot_face: :blue[Lagent] Web Demo ', divider='rainbow')
            st.sidebar.title('模型控制')

        def setup_sidebar(self):
            """Setup the sidebar for model and plugin selection."""
            model_name = st.sidebar.selectbox(
                '模型选择：', options=['gpt-3.5-turbo','internlm'])
            if model_name != st.session_state['model_selected']:
                model = self.init_model(model_name)
                self.session_state.clear_state()
                st.session_state['model_selected'] = model_name
                if 'chatbot' in st.session_state:
                    del st.session_state['chatbot']
            else:
                model = st.session_state['model_map'][model_name]

            plugin_name = st.sidebar.multiselect(
                '插件选择',
                options=list(st.session_state['plugin_map'].keys()),
                default=[list(st.session_state['plugin_map'].keys())[0]],
            )

            plugin_action = [
                st.session_state['plugin_map'][name] for name in plugin_name
            ]
            if 'chatbot' in st.session_state:
                st.session_state['chatbot']._action_executor = ActionExecutor(
                    actions=plugin_action)
            if st.sidebar.button('清空对话', key='clear'):
                self.session_state.clear_state()
            uploaded_file = st.sidebar.file_uploader(
                '上传文件', type=['png', 'jpg', 'jpeg', 'mp4', 'mp3', 'wav'])
            return model_name, model, plugin_action, uploaded_file

        def init_model(self, option):
            """Initialize the model based on the selected option."""
            if option not in st.session_state['model_map']:
                if option.startswith('gpt'):
                    st.session_state['model_map'][option] = GPTAPI(
                        model_type=option)
                else:
                    st.session_state['model_map'][option] = HFTransformerCasualLM(
                        '/root/model/Shanghai_AI_Laboratory/internlm-chat-7b')
            return st.session_state['model_map'][option]

        def initialize_chatbot(self, model, plugin_action):
            """Initialize the chatbot with the given model and plugin actions."""
            return ReAct(
                llm=model, action_executor=ActionExecutor(actions=plugin_action))

        def render_user(self, prompt: str):
            with st.chat_message('user'):
                st.markdown(prompt)

        def render_assistant(self, agent_return):
            with st.chat_message('assistant'):
                for action in agent_return.actions:
                    if (action):
                        self.render_action(action)
                st.markdown(agent_return.response)

        def render_action(self, action):
            with st.expander(action.type, expanded=True):
                st.markdown(
                    "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>插    件</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                    + action.type + '</span></p>',
                    unsafe_allow_html=True)
                st.markdown(
                    "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>思考步骤</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                    + action.thought + '</span></p>',
                    unsafe_allow_html=True)
                if (isinstance(action.args, dict) and 'text' in action.args):
                    st.markdown(
                        "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'> 执行内容</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",  # noqa E501
                        unsafe_allow_html=True)
                    st.markdown(action.args['text'])
                self.render_action_results(action)

        def render_action_results(self, action):
            """Render the results of action, including text, images, videos, and
            audios."""
            if (isinstance(action.result, dict)):
                st.markdown(
                    "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'> 执行结果</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",  # noqa E501
                    unsafe_allow_html=True)
                if 'text' in action.result:
                    st.markdown(
                        "<p style='text-align: left;'>" + action.result['text'] +
                        '</p>',
                        unsafe_allow_html=True)
                if 'image' in action.result:
                    image_path = action.result['image']
                    image_data = open(image_path, 'rb').read()
                    st.image(image_data, caption='Generated Image')
                if 'video' in action.result:
                    video_data = action.result['video']
                    video_data = open(video_data, 'rb').read()
                    st.video(video_data)
                if 'audio' in action.result:
                    audio_data = action.result['audio']
                    audio_data = open(audio_data, 'rb').read()
                    st.audio(audio_data)


    def main():
        logger = get_logger(__name__)
        # Initialize Streamlit UI and setup sidebar
        if 'ui' not in st.session_state:
            session_state = SessionState()
            session_state.init_state()
            st.session_state['ui'] = StreamlitUI(session_state)

        else:
            st.set_page_config(
                layout='wide',
                page_title='lagent-web',
                page_icon='./docs/imgs/lagent_icon.png')
            # st.header(':robot_face: :blue[Lagent] Web Demo ', divider='rainbow')
        model_name, model, plugin_action, uploaded_file = st.session_state[
            'ui'].setup_sidebar()

        # Initialize chatbot if it is not already initialized
        # or if the model has changed
        if 'chatbot' not in st.session_state or model != st.session_state[
                'chatbot']._llm:
            st.session_state['chatbot'] = st.session_state[
                'ui'].initialize_chatbot(model, plugin_action)

        for prompt, agent_return in zip(st.session_state['user'],
                                        st.session_state['assistant']):
            st.session_state['ui'].render_user(prompt)
            st.session_state['ui'].render_assistant(agent_return)
        # User input form at the bottom (this part will be at the bottom)
        # with st.form(key='my_form', clear_on_submit=True):

        if user_input := st.chat_input(''):
            st.session_state['ui'].render_user(user_input)
            st.session_state['user'].append(user_input)
            # Add file uploader to sidebar
            if uploaded_file:
                file_bytes = uploaded_file.read()
                file_type = uploaded_file.type
                if 'image' in file_type:
                    st.image(file_bytes, caption='Uploaded Image')
                elif 'video' in file_type:
                    st.video(file_bytes, caption='Uploaded Video')
                elif 'audio' in file_type:
                    st.audio(file_bytes, caption='Uploaded Audio')
                # Save the file to a temporary location and get the path
                file_path = os.path.join(root_dir, uploaded_file.name)
                with open(file_path, 'wb') as tmpfile:
                    tmpfile.write(file_bytes)
                st.write(f'File saved at: {file_path}')
                user_input = '我上传了一个图像，路径为: {file_path}. {user_input}'.format(
                    file_path=file_path, user_input=user_input)
            agent_return = st.session_state['chatbot'].chat(user_input)
            st.session_state['assistant'].append(copy.deepcopy(agent_return))
            logger.info(agent_return.inner_steps)
            st.session_state['ui'].render_assistant(agent_return)


    if __name__ == '__main__':
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root_dir = os.path.join(root_dir, 'tmp_dir')
        os.makedirs(root_dir, exist_ok=True)
        main()
    ```
  - 运行 Lagent
    ```bash
    streamlit run /root/code/lagent/examples/react_web_demo.py --server.address 127.0.0.1 --server.port 6006
    ```
    ![运行 lagent](assets/image-5.png)
  - 效果展示&作业
    ![lagent web demo](assets/image-6.png)
    ![lagent web demo1](assets/image-7.png)

### 浦语·灵笔图文理解创作 Demo
  - 在 conda 环境中克隆一个已有的pytorch 2.0.1 的环境
    ```bash
    /root/share/install_conda_env_internlm_base.sh xcomposer-demo
    ```
  - 激活环境
    ```bash
    conda activate xcomposer-demo
    ```
  - 安装依赖
    ```bash
    pip install transformers==4.33.1 timm==0.4.12 sentencepiece==0.1.99 gradio==3.44.4 markdown2==2.4.10 xlsxwriter==3.1.2 einops accelerate
    ```
  - 下载模型，从本地复制
    ```bash
    mkdir -p /root/model/Shanghai_AI_Laboratory
    cp -r /root/share/temp/model_repos/internlm-xcomposer-7b /root/model/Shanghai_AI_Laboratory
    ```
  - 克隆InternLM-XComposer 仓库的代码
    ```bash
    cd /root/code
    git clone https://gitee.com/internlm/InternLM-XComposer.git
    cd /root/code/InternLM-XComposer
    git checkout 3e8c79051a1356b9c388a6447867355c0634932d  # 最好保证和教程的 commit 版本一致
    ```
  - 运行 demo
    ```bash
    cd /root/code/InternLM-XComposer
    python examples/web_demo.py  \
      --folder /root/model/Shanghai_AI_Laboratory/internlm-xcomposer-7b \
      --num_gpus 1 \
      --port 6006
    ```
  - 效果展示
    ![xcomposer demo](assets/image-9.png)
    ![看见上海博物馆](assets/image-10.png)
    ![图文并茂](assets/image-11.png)
    ![多模态](assets/image-12.png)

  

### 作业
  - hugging face 安装
    ```bash
    pip install -U huggingface_hub
    ```
  - 新建 python 文件，下载模型
    ```python
    import os
    # 下载模型
    os.system('huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir your_path')
    ```
  - 下载模型中部分文件
    ```python
    import os
    from huggingface_hub import hf_hub_download  # Load model directly

    hf_hub_download(repo_id="internlm/internlm-7b", filename="config.json", local_dir="/root/model/Shanghai_AI_Laboratory/internlm-chat-20b")
    ```
    ![下载结果](assets/image-8.png)


 