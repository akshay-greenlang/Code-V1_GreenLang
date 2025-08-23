from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of agent's purpose")
    version: str = Field(default="0.0.1", description="Agent version")
    enabled: bool = Field(default=True, description="Whether agent is enabled")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific parameters")


class AgentResult(BaseModel):
    success: bool = Field(..., description="Whether the agent execution was successful")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseAgent(ABC):
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig(
            name=self.__class__.__name__,
            description=self.__class__.__doc__ or "Base agent"
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return True
    
    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return input_data
    
    def postprocess(self, result: AgentResult) -> AgentResult:
        return result
    
    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            if not self.config.enabled:
                return AgentResult(
                    success=False,
                    error=f"Agent {self.config.name} is disabled"
                )
            
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    error="Input validation failed"
                )
            
            processed_input = self.preprocess(input_data)
            result = self.execute(processed_input)
            return self.postprocess(result)
            
        except Exception as e:
            self.logger.error(f"Agent execution failed: {str(e)}")
            return AgentResult(
                success=False,
                error=str(e)
            )
    
    def __repr__(self):
        return f"{self.config.name}(version={self.config.version})"