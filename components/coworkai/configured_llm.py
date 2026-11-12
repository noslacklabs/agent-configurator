from agent_configurator.chat_model import ConfiguredChatModel
from agent_configurator.client import ConfiguratorClient, ConfiguratorError
from lfx.custom.custom_component.component import Component
from lfx.field_typing import LanguageModel
from lfx.io import DropdownInput, FloatInput, IntInput, Output


class ConfiguredLLMComponent(Component):
    display_name = "Configured LLM"
    description = (
        "A model from the LLM Configurator, chosen by name. The configurator holds the provider key "
        "and makes the call, so no key is entered or stored here."
    )
    icon = "brain-circuit"
    name = "ConfiguredLLM"

    inputs = [
        DropdownInput(
            name="config_name",
            display_name="LLM Config",
            info="Active configs from the LLM Configurator. Use the refresh button to reload the list.",
            options=[],
            value="",
            required=True,
            refresh_button=True,
            real_time_refresh=True,
            combobox=True,
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            info="Leave empty to use the model's default.",
            advanced=True,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            info="0 uses the config's cap. A higher value is lowered to the cap (tokens per request).",
            value=0,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Language Model", name="model_output", method="build_model"),
    ]

    def update_build_config(self, build_config, field_value, field_name=None):
        try:
            names = ConfiguratorClient(timeout=10).config_names("chat")
        except ConfiguratorError as exc:
            # Keep whatever is selected so a flow still opens while the configurator is down.
            self.log(f"Could not load LLM configs: {exc}")
            return build_config
        build_config["config_name"]["options"] = names
        return build_config

    def build_model(self) -> LanguageModel:
        config_name = (self.config_name or "").strip()
        if not config_name:
            msg = "Pick an LLM config. Use the refresh button if the list is empty."
            raise ValueError(msg)

        max_tokens = int(self.max_tokens or 0)
        if max_tokens > 0:
            cap = ConfiguratorClient(timeout=10).max_tokens_cap(config_name)
            if cap and max_tokens > cap:
                self.log(f"Max Tokens {max_tokens} is above this config's cap, using {cap}.")
                max_tokens = cap

        temperature = self.temperature
        model = ConfiguredChatModel(
            config_name=config_name,
            temperature=float(temperature) if temperature not in (None, "") else None,
            max_tokens=max_tokens or None,
        )
        self.status = f"LLM config: {config_name}"
        return model
