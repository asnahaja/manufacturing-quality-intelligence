from pydantic import BaseModel, Field


class MachineInput(BaseModel):
    machine_id: str
    production_line: str
    machine_type: str

    machine_age_years: float = Field(ge=0)
    hours_since_maintenance: float = Field(ge=0)
    maintenance_event: int = Field(ge=0, le=1)

    load_factor: float = Field(ge=0)
    temperature_c: float
    vibration_mm_s: float = Field(ge=0)
    pressure_psi: float = Field(ge=0)
    current_a: float = Field(ge=0)
    rpm: float = Field(ge=0)
    cycle_time_sec: float = Field(gt=0)
    tool_wear_pct: float = Field(ge=0, le=100)
    machine_health_index: float
    quality_score: float

    timestamp: str


    # MachineInput defines the information a client must send to the API. Pydantic validates the values before the prediction endpoint receives them.