import os

from openai import OpenAI

PROVIDER_URLS = {
    'foundry': 'https://mtha-testbed-proj-resource.openai.azure.com/openai/v1',
    'anthropic': 'https://api.anthropic.com/v1',
    'deepseek': 'https://api.deepseek.com',
    'fireworks': 'https://api.fireworks.ai/inference/v1',
    'gemini': 'https://generativelanguage.googleapis.com/v1beta/openai',
    'huggingface': 'https://router.huggingface.co/v1',
    'mistral': 'https://api.mistral.ai/v1',
    'openai': 'https://api.openai.com/v1',
    'perplexity': 'https://api.perplexity.ai',
    'qwen': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    'xai': 'https://api.x.ai/v1',
}


class Convo:
    """A stateful conversation with a single LLM, using the OpenAI-compatible chat interface."""

    def __init__(self, provider: str, model: str, system: str = None):
        # Provider strings keep their display capitalisation in config/experiments.json
        # (e.g. "Gemini", "OpenAI"); casefold for URL / env-var lookup.
        key = provider.casefold()
        if key not in PROVIDER_URLS:
            raise ValueError(f'Unknown provider {provider!r}. Known: {list(PROVIDER_URLS)}')
        self.provider = provider
        self.model = model
        self.messages = [{'role': 'system', 'content': system or 'You are a helpful assistant.'}]
        self.log_data = []
        self._client = OpenAI(
            base_url=PROVIDER_URLS[key],
            api_key=os.environ[f'{key.upper()}_API_KEY'],
        )

    def ask(self, prompt: str, reasoning_effort: str = None) -> str:
        """Send a message and return the reply. Appends both to conversation history."""
        self.messages.append({'role': 'user', 'content': prompt})
        params = {'model': self.model, 'messages': self.messages}
        if reasoning_effort is not None:
            params['reasoning_effort'] = reasoning_effort
        response = self._client.chat.completions.create(**params)
        content = response.choices[0].message.content
        self.log_data.append(response)
        self.messages.append({'role': 'assistant', 'content': content})
        return content

    def history(self) -> list:
        return self.messages

    def log(self) -> list:
        return self.log_data
