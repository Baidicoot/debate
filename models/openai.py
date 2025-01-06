from openai import AsyncOpenAI
from debate import Message, Transcript, APIMessages, DebateConfig, Judge, Agent
from typing import Optional, List, Dict
from prompts import load_prompts

PROMPTS = load_prompts("prompts.yaml")

class OpenAIAgent(Agent):
    def __init__(self, debate_config: DebateConfig, client: AsyncOpenAI, model: Optional[str] = None, debug_dummy: bool = False):
        self._debate_config = debate_config
        self._client = client
        self._model = model if model else "default"
        self._messages = []
        self._debug_dummy = debug_dummy

    def receive(self, message: Message):
        self._messages.append(message)

    async def respond(self):
        if self._debug_dummy:
            return "This is a dummy response"
        response = await self.generate(self.render())
        return response

    async def generate(self, messages: APIMessages, **kwargs) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

    def render(self) -> APIMessages:
        api_messages = [
            {
                "role": "system",
                "content": PROMPTS["debater"]["system_prompt"].format(
                    topic=self._debate_config.topic,
                    self_stance=self._debate_config.self_stance,
                    opponent_stance=self._debate_config.opponent_stance
                )
            }
        ]
        for msg in self._messages:
            if msg.speaker == "self":
                api_messages.append({
                    "role": "assistant",
                    "content": msg.content
                })
            elif msg.speaker == "opponent":
                api_messages.append({
                    "role": "user",
                    "content": PROMPTS["debater"]["opponent_speech_format"].format(
                        opponent_speech=msg.content
                    )
                })
        return api_messages

class OpenAIJudge(Judge):
    def __init__(self, config: DebateConfig, client: AsyncOpenAI, model: Optional[str] = None, debug_dummy: bool = False):
        self._config = config
        self._client = client
        self._model = model if model else "default"
        self._messages = []
        self._debug_dummy = debug_dummy

    def receive(self, message: Message):
        self._messages.append(message)

    def render(self) -> APIMessages:
        api_messages = [
            {
                "role": "user",
                "content": PROMPTS["judge"]["system_prompt"].format(
                    topic=self._config.topic,
                    debater_a_stance=self._config.self_stance,
                    debater_b_stance=self._config.opponent_stance
                )
            }
        ]
        rendered_transcript = ""
        round_number = 1
        for i, msg in enumerate(self._messages):
            if msg.speaker == "A" and i % 2 == 0:
                rendered_transcript += PROMPTS["judge"]["round_format_A"].format(
                    round_number=round_number,
                    debater_a_speech=msg.content
                )
            elif msg.speaker == "B" and i % 2 == 1:
                rendered_transcript += PROMPTS["judge"]["round_format_B"].format(
                    round_number=round_number,
                    debater_b_speech=msg.content
                )
                round_number += 1
            else:
                raise ValueError(f"Invalid speaker: {msg.speaker} at index {i}")
        
        rendered_transcript += PROMPTS["judge"]["judge_prompt"].format(
            topic=self._config.topic,
            debater_a_stance=self._config.self_stance,
            debater_b_stance=self._config.opponent_stance
        )
        
        api_messages.append({
            "role": "user",
            "content": rendered_transcript
        })
        return api_messages
    
    async def judge(self) -> Dict[str, float]:
        messages = self.render()
        logits = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            logprobs=True,
            top_logprobs=20,
            max_tokens=1
        )
        scores = {
            "A": None,
            "B": None
        }
        for logit in logits.choices[0].logprobs.content[0].top_logprobs:
            if logit.token == "A":
                scores["A"] = logit.logprob
            elif logit.token == "B":
                scores["B"] = logit.logprob
        return scores
