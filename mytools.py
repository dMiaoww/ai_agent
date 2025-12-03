from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str
    user_location: str | None = None

# @tool
# def get_user_location(runtime: ToolRuntime[Context]) -> str:
#     """Retrieve user information based on user ID."""
#     user_id = runtime.context.user_id
#     return "Florida" if user_id == "1" else "SF"

@tool
def set_user_location(location: str, runtime: ToolRuntime[Context]) -> str:
    """Set user location in context."""
    runtime.context.user_location = location
    return f"User location set to {location}"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Set user location in context."""
    print(f"User location : {runtime.context.user_location}")
    return f"{runtime.context.user_location}"


@tool
def set_cmd_vel(vx: float, w: float) -> str:
    """Set command velocity."""
    print(f"Setting velocity: linear={vx}, angular={w}")
    return f"Setting velocity: linear={vx}, angular={w}"


tools = [get_weather_for_location, set_user_location, get_user_location, set_cmd_vel]