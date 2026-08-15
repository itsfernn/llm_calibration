import backoff
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_together import ChatTogether

load_dotenv()


def get_llm(model_name="gpt-4.1-mini", temperature=0.7, max_tokens = 300, **kwargs):
    short_names = {
        "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
        "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    }

    config = {
        "model": short_names.get(model_name, model_name),
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }
    if config["model"].startswith("gpt"):
        llm =  ChatOpenAI(**config)
    else:
        llm = ChatTogether(**config)

    return llm


class Chat:
    def __init__(
        self, developer_prompt=None, model="gpt-4o-mini", temperature=1.0, **kwargs
    ):
        self.chat = []
        if developer_prompt is not None:
            self.add_message(developer_prompt, "system")

        full_model_names = {
            "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
            "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        }

        self.config = {
            "model": full_model_names.get(model, model),
            "temperature": temperature,
            "max_tokens": 500,
            **kwargs,
        }
        if self.config["model"].startswith("gpt"):
            self.model = ChatOpenAI(**self.config)
        else:
            self.model = ChatTogether(**self.config)

    def reset(self):
        self.chat = [self.chat[0]]

    def add_message(self, prompt, role):
        self.chat.append({"content": prompt, "role": role})

    @backoff.on_exception(backoff.expo, Exception, max_time=60, max_tries=6)
    def response(self):
        response = self.model.invoke(self.chat)
        message = response.content
        assert isinstance(message, str), (
            f"Expected string response, got {type(message)}"
        )
        message = message.strip()
        return message

    def raw_response(self):
        response = self.model.invoke(self.chat)
        return response


    def set_system_prompt(self, prompt):
        self.chat[0]["content"] = prompt

    def __str__(self):
        """
        Returns a human-readable string representation of the chat history and settings.
        """
        chat_repr = "\n".join(
            f"{entry['role'].capitalize()}: {entry['content']}" for entry in self.chat
        )
        config = "\n".join([f"{k}: {v}" for k, v in self.config.items()])
        return f"Chat History:\n{chat_repr}\n\nConfig:\n{config}"
