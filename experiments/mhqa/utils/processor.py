import math

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .metrics import compute_em
from .utils import extract_texts_and_confidences


# Base Processor Class
class BaseSampleProcessor:
    system_prompt = None  # Static class variable to hold the prompt
    few_shot_messages = []

    def _create_prompt(self, question):
        if self.system_prompt is None:
            raise NotImplementedError("System prompt must be defined in the subclass.")
        prompt: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]

        for fs_question, fs_response in self.few_shot_messages:
            prompt.append(HumanMessage(f"The question is: {fs_question}"))
            prompt.append(AIMessage(fs_response))

        prompt.append(HumanMessage(f"The question is: {question}"))

        return prompt

    def process_sample(self, sample, llm):
        question = sample["question"]
        prompt = self._create_prompt(question)
        response = llm.invoke(prompt).content

        assert isinstance(response, str), "Response from LLM should be a string."

        texts, confidences = extract_texts_and_confidences(response)

        # If no texts were extracted, try again with the response
        if len(texts) == 0:
            response = llm.invoke(prompt).content
            assert isinstance(response, str), "Response from LLM should be a string."
            texts, confidences = extract_texts_and_confidences(response)

        sample["response"] = response
        sample["texts"] = texts
        sample["confidences"] = confidences

        return sample

    def eval_sample(self, sample):
        prediction = sample["texts"][-1]
        confidence = sample["confidences"][-1]
        sample["prediction"] = str(prediction)
        sample["confidence"] = float(confidence)
        sample["em"] = int(compute_em(prediction, sample["answer"]))
        return sample

    # Implementing the __call__ function for the Base class to call process_sample
    def __call__(self, sample, llm):
        return self.process_sample(sample, llm)


# Concrete Processor for Direct Verb
class DirectProcessor(BaseSampleProcessor):
    system_prompt = (
        "Provide your best guess for the following multi-hop question. "
        "Give your anwer with no other words or explanation. "
        "Also provide how confident you are that the answer is correct. (between 0.0 and 1.0)\n\n"
        "<format>\n"
        "Answer: {most likely guess, as short as possible; not a complete sentence, just the guess!} "
        "Confidence: {confidence for the answer between 0.0 and 1.0} \n"
        "</format>\n"
    )


class TopKProcessor(BaseSampleProcessor):
    def __init__(self, k=5, normalize=False):
        self.normalize = normalize

        self.system_prompt = (
            f"Provide your top {k} best guesses for the following multi-hop question. "
            "Give your anwers with no other words or explanation. "
            "Also provide how confident you are that each answer is correct. (between 0.0 and 1.0)\n\n"
            "<format>\n"
            "Answer: {first guess, as short as possible; not a complete sentence, just the guess!} "
            "Confidence: {confidence for the first answer between 0.0 and 1.0} \n"
            "Answer: {second guess, as short as possible; not a complete sentence, just the guess!} "
            "Confidence: {confidence for the second answer between 0.0 and 1.0} \n"
            "...\n"
            "</format>\n"
        )

    def eval_sample(self, sample):
        matches = zip(sample["texts"], sample["confidences"])

        matches = sorted(matches, key=lambda x: x[1], reverse=True)

        prediction = matches[0][0]
        confidence = matches[0][1]
        sample["prediction"] = str(prediction)
        sample["confidence"] = float(confidence)
        sample["em"] = int(compute_em(prediction, sample["answer"]))

        if self.normalize:
            total_confidence = sum(sample["confidences"]) or 1.0
            sample["confidence"] = confidence / total_confidence

        return sample


# Concrete Processor for Cot Verb
class CotProcessor(BaseSampleProcessor):
    system_prompt = (
        "For the following multi-hop question let's think step by step. "
        "After your reasoning give your final answer on a new line with no other words or explanation. "
        "Also provide how confident you are the answer is correct. (0.0 to 1.0)\n\n"
        "<format>\n"
        "{thoughts}\n"
        "Final Answer: {most likely guess, as short as possible; not a complete sentence, just the guess!} "
        "Confidence: {confidence for your answer}\n"
        "</format>\n"
    )


# Concrete Processor for Cot Verb
class MultistepProcessor(BaseSampleProcessor):
    system_prompt = (
        "Provide your best guess for the following multi-hop question. "
        "Provide a step-by-step explanation of your thought process. "
        "Each step should only contain a single new fact and be on a new line. "
        "When enough information is present, give your final answer on a new line with no other words or explanation. "
        "Also provide how confident you are between 0.0 and 1.0 that a given line is correct.\n\n"
        "<format>\n"
        "Step 1: {first reasoning step} Confidence: {confidence for step 1}\n"
        "Step 2: {second reasoning step} Confidence: {confidence for step 2}\n"
        "...\n"
        "Final Answer: {most likely guess, as short as possible; not a complete sentence, just the guess!} "
        "Confidence: {final confidence for your answer}\n"
        "</format>\n"
    )

    def eval_sample(self, sample):
        sample = super().eval_sample(sample)
        sample["confidence"] = math.prod(sample["confidences"])
        # create min, max, final, mean, number of steps
        sample["min_confidence"] = min(sample["confidences"])
        sample["max_confidence"] = max(sample["confidences"])
        sample["final_confidence"] = sample["confidences"][-1]
        sample["mean_confidence"] = sum(sample["confidences"]) / len(
            sample["confidences"]
        )
        sample["num_steps"] = (
            len(sample["confidences"]) - 1
        )  # Exclude final answer step

        return sample


# Concrete Processor for Cot Verb
class MultistepFewshotProcessor(MultistepProcessor):
    few_shot_messages = [
        (
            "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?",
            """Step 1: Sacramento International Airport is located in Sacramento, California. Confidence: 1.0  
Step 2: Knox County Regional Airport is located in Maine. Confidence: 1.0  
Final Answer: Knox County Regional Airport Confidence: 1.0""",
        ),
        (
            "Chairmen of the Bored is an album that references a joke made by a cast member of what series?",
            """Step 1: "Chairmen of the Bored" is an album by Lord Finesse. Confidence: 0.9  
Step 2: The phrase "Chairmen of the Bored" is a reference to a joke made by a cast member of a series. Confidence: 0.85  
Step 3: "Chairmen of the Bored" is also the title of a sketch from the series "The Kids in the Hall." Confidence: 0.9  
Step 4: The sketch "Chairmen of the Bored" was performed by a cast member of "The Kids in the Hall." Confidence: 0.9  
Final Answer: The Kids in the Hall Confidence: 0.9""",
        ),
    ]


# Factory Method for creating processors
def get_processor(method_name) -> BaseSampleProcessor:
    if method_name == "direct":
        return DirectProcessor()
    elif method_name == "multistep":
        return MultistepProcessor()
    elif method_name == "multistep-fewshot":
        return MultistepFewshotProcessor()
    elif method_name == "cot":
        return CotProcessor()
    elif method_name == "top-k":
        return TopKProcessor()
    elif method_name == "top-k-norm":
        return TopKProcessor(normalize=True)
    else:
        raise ValueError(f"Invalid method: {method_name}")
