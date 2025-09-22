"""Client for making batch requests to OpenRouter's chat completions API."""

import logging
import math
from json.decoder import JSONDecodeError
from typing import Optional, Union

from openai import APIConnectionError, APITimeoutError, OpenAIError, RateLimitError
from pydantic import ValidationError as PydanticValidationError

from fenic._inference.common_openai.utils import handle_openai_compatible_response
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.openrouter.openrouter_profile_manager import (
    OpenRouterCompletionsProfileManager,
)
from fenic._inference.openrouter.openrouter_provider import OpenRouterModelProvider
from fenic._inference.rate_limit_strategy import (
    AdaptiveBackoffRateLimitStrategy,
    RateLimitStrategy,
    TokenEstimate,
)
from fenic._inference.request_utils import (
    generate_completion_request_key,
    parse_openrouter_rate_limit_headers,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core.error import ConfigurationError
from fenic.core.metrics import LMMetrics

TOOLS = "tools"

STRUCTURED_OUTPUTS = "structured_outputs"

RESPONSE_FORMAT = "response_format"
logger = logging.getLogger(__name__)


class OpenRouterBatchChatCompletionsClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Client for making batch requests to OpenRouter's chat completions API.

    Notes:
        - Uses the OpenAI SDK pointed at OpenRouter via base_url.
        - Default rate limiting uses AdaptiveBackoffRateLimitStrategy; provider backoffs still apply.
    """

    def __init__(
        self,
        model: str,
        rate_limit_strategy: RateLimitStrategy = None,
        queue_size: int = 100,
        max_backoffs: int = 10,
        profiles: Optional[dict[str, object]] = None,
        default_profile_name: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            model_provider=ModelProvider.OPENROUTER,
            model_provider_class=OpenRouterModelProvider(),
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=TiktokenTokenCounter(
                model_name=model, fallback_encoding="o200k_base"
            ),
        )
        self._model_parameters = model_catalog.get_completion_model_parameters(
            ModelProvider.OPENROUTER, model
        )
        self._profile_manager = OpenRouterCompletionsProfileManager(
            model_parameters=self._model_parameters,
            profile_configurations=profiles,
            default_profile_name=default_profile_name,
        )
        self._aio_client = OpenRouterModelProvider().aio_client
        self._metrics = LMMetrics()

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        profile = self._profile_manager.get_profile_by_name(request.model_profile)
        common_params = {
                "model": self.model,
                "messages": request.messages.to_message_list(),
                "max_completion_tokens": self._get_max_output_tokens(request),
                "n": 1,
            }

        if request.top_logprobs:
            common_params.update(
                {"logprobs": True, "top_logprobs": request.top_logprobs}
            )

        if request.temperature and self._model_parameters.supports_custom_temperature:
            common_params.update({"temperature": request.temperature})

        try:
            if request.structured_output:
                strategy = (
                    self._profile_manager.get_profile_by_name(request.model_profile).structured_output_strategy
                    or "prefer_response_format"
                )
                supports_structured = STRUCTURED_OUTPUTS in self._model_parameters.supported_parameters
                supports_tools = TOOLS in self._model_parameters.supported_parameters
                if supports_structured and supports_tools:
                    use_tools = strategy == "prefer_tools"
                else:
                    use_tools = supports_tools and not supports_structured

                if supports_structured and not use_tools:
                    common_params[RESPONSE_FORMAT] = request.structured_output.pydantic_model
                    response = await self._aio_client.chat.completions.parse(
                        **common_params, extra_body=profile.extra_body
                    )
                elif supports_tools:
                    response_schema = request.structured_output.json_schema
                    response_schema["additionalProperties"] = False
                    common_params[TOOLS] = [
                        {
                            "type": "function",
                            "function": {
                                "name": "output_formatter",
                                "description": "Format the output of the model to correspond strictly to the provided schema.",
                                "parameters": request.structured_output.json_schema,
                                "strict": True,
                            },
                        }
                    ]
                    response = await self._aio_client.chat.completions.create(
                        **common_params, extra_body=profile.extra_body
                    )
                else:
                    return FatalException(
                        ConfigurationError(
                            f"Model {self.model} does not support structured outputs, or tool calling, but the current "
                            f"request requires an output format. Select a different model that supports `structured_outputs`, or `tools`"
                        )
                    )
            else:
                response = await self._aio_client.chat.completions.create(
                    **common_params, extra_body=profile.extra_body
                )

            completion_choice, maybe_exception = handle_openai_compatible_response(
                model_provider=ModelProvider.OPENROUTER,
                model_name=self.model,
                request=request,
                response=response,
                request_key_generator=self.get_request_key,
            )
            if maybe_exception:
                return maybe_exception
            usage = response.usage
            cached_input_tokens = (
                usage.prompt_tokens_details.cached_tokens
                if usage.prompt_tokens_details
                else 0
            )
            uncached_input_tokens = usage.prompt_tokens - cached_input_tokens
            total_prompt_tokens = usage.prompt_tokens
            reasoning_tokens = (
                usage.completion_tokens_details.reasoning_tokens
                if usage.completion_tokens_details
                else 0
            )
            total_output_tokens = usage.completion_tokens
            completion_tokens = total_output_tokens - reasoning_tokens

            fenic_usage = ResponseUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_prompt_tokens + total_output_tokens,
                cached_tokens=cached_input_tokens,
                thinking_tokens=reasoning_tokens,
            )
            self._metrics.num_cached_input_tokens += cached_input_tokens
            self._metrics.num_uncached_input_tokens += uncached_input_tokens
            self._metrics.num_output_tokens += total_output_tokens
            self._metrics.num_requests += 1

            # Cost from OpenRouter usage, fallback to catalog if not provided
            model_extra = usage.model_extra
            cost_value = model_extra.get("cost")
            if isinstance(cost_value, (int, float)):
                self._metrics.cost += float(cost_value)
            else:
                self._metrics.cost += model_catalog.calculate_completion_model_cost(
                    model_provider=self.model_provider,
                    model_name=self.model,
                    uncached_input_tokens=uncached_input_tokens,
                    cached_input_tokens_read=cached_input_tokens,
                    output_tokens=total_output_tokens,
                )

            # If we used tool calls to generate the structured output, retrieve the content from the function args.
            if request.structured_output and completion_choice.message.tool_calls:
                completion = completion_choice.message.tool_calls[0].function.arguments
            else:
                completion = completion_choice.message.content
            return FenicCompletionsResponse(
                completion=completion,
                logprobs=completion_choice.logprobs,
                usage=fenic_usage,
            )
        except RateLimitError as e:
            if isinstance(self.rate_limit_strategy, AdaptiveBackoffRateLimitStrategy):
                rpm_hint, retry_at_s = parse_openrouter_rate_limit_headers(
                    e.response.headers
                )
                self.rate_limit_strategy.register_rate_limit_hint(rpm_hint, retry_at_s)
            return TransientException(e)
        except (APITimeoutError, APIConnectionError) as e:
            return TransientException(e)
        # encountered when the response is not valid JSON. can sometimes be fixed with a retry
        # sending the request to a different provider.
        except (JSONDecodeError, PydanticValidationError) as e:
            return TransientException(e)
        except OpenAIError as e:
            return FatalException(e)

    def get_request_key(self, request: FenicCompletionsRequest) -> str:
        """Generate a unique key for request deduplication."""
        return generate_completion_request_key(request)

    def estimate_tokens_for_request(
        self, request: FenicCompletionsRequest
    ) -> TokenEstimate:
        return TokenEstimate(
            input_tokens=self.token_counter.count_tokens(request.messages),
            output_tokens=self._get_max_output_tokens(request),
        )

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def _get_max_output_tokens(self, request: FenicCompletionsRequest) -> int:
        return request.max_completion_tokens + self._get_expected_additional_reasoning_tokens(request)

    # This is a slightly less conservative estimate than the OpenRouter documentation on how reasoning_effort is used to
    # generate a reasoning.max_tokens for models that only support reasoning.max_tokens.
    # These percentages are slightly lower, since our use-cases generally require fewer reasoning tokens.
    # https://openrouter.ai/docs/use-cases/reasoning-tokens#reasoning-effort-level
    def _get_expected_additional_reasoning_tokens(self, request: FenicCompletionsRequest) -> int:
        profile_config = self._profile_manager.get_profile_by_name(request.model_profile)
        additional_reasoning_tokens = 0
        if profile_config.reasoning_max_tokens:
            additional_reasoning_tokens = profile_config.reasoning_max_tokens
        elif profile_config.reasoning_effort == "low":
            additional_reasoning_tokens = math.ceil(0.15 * self._model_parameters.max_output_tokens)
        elif profile_config.reasoning_effort == "medium":
            additional_reasoning_tokens = math.ceil(0.30 * self._model_parameters.max_output_tokens)
        elif profile_config.reasoning_effort == "high":
            additional_reasoning_tokens = math.ceil(0.60 * self._model_parameters.max_output_tokens)
        return additional_reasoning_tokens
