from .chat import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

def get_local_llm(thread):
    if not hasattr(thread, 'eval_llm'):
        thread.eval_llm = get_llm(model="gpt-4.1-mini", temperature=0.0)
    return thread.eval_llm

system_prompt = (
    "You are given a question, the correct answer, and a guess. "
    "Your task is to determine whether the guess and the correct answer are semantically equivalent.\n"
    "Respond with only “Yes” or “No”. Do not include any other words or phrases.\n\n"
    "<format>\n"
    "Question: {question}\n"
    "Answer: {correct answer}\n"
    "Guess: {best guess}\n"
    "Yes / No\n"
    "</format>\n"
)
def gpt_eval(sample, thread):
    if sample["em"] == 1:
        return 1
    else:
        llm = get_local_llm(thread)
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Question: {sample['question']}\n"
                        f"Answer: {sample['answer']}\n"
                        f"Guess: {sample['prediction']}"
            )
        ]
        response = llm.invoke(prompt).content
        assert isinstance(response, str), "Response from LLM should be a string."
        response = response.strip().lower()

    if "yes" in response:
        return 1
    elif "no" in response:
        return 0
    else:
        return -1
