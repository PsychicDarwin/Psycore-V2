from src.rl.QModel import QModel
class PromptStage:
    def __init__(self, QModel = None, prompt_state = None):
        self.prompt_state = prompt_state

    def decide_between_prompts(self, prompt_1, prompt_2) -> tuple[str, bool]:
        if self.prompt_state == "original":
            return prompt_1, False
        elif self.prompt_state == "elaborated":
            return prompt_2, True
        elif self.prompt_state == "q_learning" or self.prompt_state == "q_training":
            raise NotImplementedError("Q learning is not implemented yet")