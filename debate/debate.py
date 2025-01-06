import typing
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from pydantic import BaseModel
import asyncio

class Message(BaseModel):
    """
    A turn in a debate.
    """
    content: str
    speaker: str

class Transcript(BaseModel):
    """
    A collection of messages.
    """
    messages: List[Message]

APIMessages = List[Dict[str, str]]

class HasTranscript(ABC):
    """
    A thing which has a transcript.
    """
    @abstractmethod
    def receive(self, message: Message):
        pass

class Agent(HasTranscript, ABC):
    """
    A thing which receives messages and can respond to them.
    """
    async def respond(self) -> str:
        pass

class Judge(HasTranscript, ABC):
    """
    A thing which receives messages and can judge them.
    """
    @abstractmethod
    def judge(self) -> Dict[str, float]:
        pass

class DebateConfig(BaseModel):
    """
    Debate configuration for a debate agent.
    """
    topic: str
    self_stance: str
    opponent_stance: str

class Debate:
    """
    A symmetric debate between two agents.
    The order in which they are shown to the judge should be randomized.
    """
    def __init__(self, debaters: Tuple[Agent, Agent], judge: Judge, iters: int = 3, debug_log: bool = False):
        self._debaters = debaters
        self._judge = judge
        self._iters = iters
        self._debug_log = debug_log

    async def do_round(self):
        """
        Do a round of the debate.
        """
        response_a, response_b = await asyncio.gather(
            self._debaters[0].respond(),
            self._debaters[1].respond()
        )
        
        self._debaters[0].receive(Message(content=response_a, speaker="self"))
        self._debaters[0].receive(Message(content=response_b, speaker="opponent"))

        self._debaters[1].receive(Message(content=response_b, speaker="self"))
        self._debaters[1].receive(Message(content=response_a, speaker="opponent"))

        if self._debug_log:
            print(f"Debater A: {response_a}")
            print(f"Debater B: {response_b}")

        self._judge.receive(Message(content=response_a, speaker="A"))
        self._judge.receive(Message(content=response_b, speaker="B"))
    
    async def judge(self) -> Dict[str, float]:
        return await self._judge.judge()
    
    async def run(self):
        for _ in range(self._iters):
            await self.do_round()
        return await self.judge()