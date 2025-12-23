from pydantic import BaseModel, Field
from typing import Optional


class CreditApplication(BaseModel):
    ID: Optional[int] = Field(None, example=12345)

    # Continuous / numeric features
    X1: float = Field(..., description="Credit limit")
    X5: float = Field(..., description="Age")

    # Categorical / ordinal features
    X2: int = Field(..., ge=1, le=2, description="Sex (1=male, 2=female)")
    X3: int = Field(..., ge=0, le=6, description="Education level")
    X4: int = Field(..., ge=0, le=3, description="Marital status")

    # Repayment status (ordinal)
    X6: int = Field(..., ge=-2, le=9)
    X7: int = Field(..., ge=-2, le=9)
    X8: int = Field(..., ge=-2, le=9)
    X9: int = Field(..., ge=-2, le=9)
    X10: int = Field(..., ge=-2, le=9)
    X11: int = Field(..., ge=-2, le=9)

    # Billing & payment history
    X12: float
    X13: float
    X14: float
    X15: float
    X16: float
    X17: float
    X18: float
    X19: float
    X20: float
    X21: float
    X22: float
    X23: float


class PredictionResponse(BaseModel):
    default_probability: float
    default_prediction: int
    threshold: float