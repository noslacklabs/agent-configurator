from agent_configurator.client import ConfiguratorClient, ConfiguratorError
from agent_configurator.embeddings import ConfiguredEmbeddings
from lfx.custom.custom_component.component import Component
from lfx.field_typing import Embeddings
from lfx.io import DropdownInput, IntInput, Output


class ConfiguredEmbeddingsComponent(Component):
    display_name = "Configured Embeddings"
    description = (
        "An embedding model from the LLM Configurator, chosen by name. Only configs that have an "
        "embedding model are listed. No key is entered or stored here."
    )
    icon = "binary"
    name = "ConfiguredEmbeddings"

    inputs = [
        DropdownInput(
            name="llm_name",
            display_name="LLM Config",
            info="Active configs that include an embedding model. Use the refresh button to reload the list.",
            options=[],
            value="",
            required=True,
            refresh_button=True,
            real_time_refresh=True,
            combobox=True,
        ),
        IntInput(
            name="batch_size",
            display_name="Batch Size",
            info="Texts sent per request. Each request counts toward the config's requests-per-minute limit.",
            value=64,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Embedding Model", name="embeddings", method="build_embeddings"),
    ]

    def update_build_config(self, build_config, field_value, field_name=None):
        try:
            names = ConfiguratorClient(timeout=10).config_names("embedding")
        except ConfiguratorError as exc:
            self.log(f"Could not load LLM configs: {exc}")
            return build_config
        build_config["llm_name"]["options"] = names
        return build_config

    def build_embeddings(self) -> Embeddings:
        config_name = (self.llm_name or "").strip()
        if not config_name:
            msg = "Pick an LLM config that has an embedding model."
            raise ValueError(msg)
        self.status = f"Embedding config: {config_name}"
        return ConfiguredEmbeddings(config_name=config_name, batch_size=int(self.batch_size or 64))
