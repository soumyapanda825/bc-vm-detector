from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # WebSocket server
    ws_host: str = Field("0.0.0.0", description="WebSocket bind address")
    ws_port: int = Field(8765, description="WebSocket port")

    # Audio input from FreeSWITCH
    input_encoding: str = Field(
        "pcm_s16le", description="Input encoding: pcm_s16le or ulaw"
    )
    input_sample_rate: int = Field(8000, description="FreeSWITCH stream sample rate")

    # Internal analysis rate (webrtcvad requires 8/16/32/48 kHz)
    analysis_sample_rate: int = Field(16000, description="Analysis sample rate")
    frame_ms: int = Field(20, description="VAD/analysis frame size in ms")

    # Blank call detection
    silence_dbfs: float = Field(
        -50.0, description="RMS threshold in dBFS to consider a frame silent"
    )
    blank_timeout_s: float = Field(
        7.0, description="Continuous silence duration to declare BLANK"
    )
    blank_speech_ratio_threshold: float = Field(
        0.05,
        description="If speech frames / total frames < this over blank_timeout_s → BLANK",
    )

    # Voicemail detection
    vm_score_threshold: int = Field(
        60, description="Score (0-100) to declare VOICEMAIL"
    )
    beep_freq_low: float = Field(300.0, description="Lower bound of beep freq range Hz")
    beep_freq_high: float = Field(
        1200.0, description="Upper bound of beep freq range Hz"
    )
    beep_min_duration_ms: float = Field(
        100.0, description="Min beep tone duration in ms"
    )
    beep_energy_ratio: float = Field(
        0.60,
        description="Ratio of energy in beep band vs total to flag as tone",
    )

    # Pipeline
    analysis_timeout_s: float = Field(
        12.0, description="Max analysis window before defaulting to LIVE"
    )
    vad_aggressiveness: int = Field(
        2, description="WebRTC VAD aggressiveness 0-3 (3 = most aggressive)"
    )

    # Result publishing
    webhook_url: str = Field("", description="Optional HTTP POST URL for results")
    webhook_timeout_s: float = Field(5.0, description="HTTP webhook timeout")

    # Logging
    log_level: str = Field("INFO", description="Log level")


settings = Settings()
