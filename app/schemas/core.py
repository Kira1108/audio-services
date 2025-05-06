from pydantic import BaseModel, Field

class ASRResult(BaseModel):
    conversation_id: str = Field(..., description="The ID of the conversation")
    chunk_id: int = Field(..., description="The ID of the chunk")
    start_time: int = Field(..., description="The start time of the chunk in milliseconds")
    end_time: int = Field(..., description="The end time of the chunk in milliseconds")
    text: str = Field(..., description="The transcribed text")
    is_partial:bool = Field(..., description="Whether the result is partial or final")