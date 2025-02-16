import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not hasattr(self, 'errors'):
            self.errors = []
        if not hasattr(self, 'warnings'):
            self.warnings = []
        if not hasattr(self, 'metrics'):
            self.metrics = {}
        if not hasattr(self, 'timestamp'):
            self.timestamp = datetime.now()

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)

    def add_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value

    def merge(self, other: 'ValidationResult') -> None:
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.metrics.update(other.metrics)

class ModelValidator:
    def __init__(self):
        self.validation_thresholds = {
            "memory": {
                "accuracy": 0.85,
                "latency_ms": 100
            },
            "reasoning": {
                "accuracy": 0.80,
                "consistency": 0.85
            },
            "planning": {
                "success_rate": 0.75,
                "efficiency": 0.80
            },
            "action": {
                "accuracy": 0.90,
                "safety_score": 0.95
            },
            "model_selector": {
                "selection_accuracy": 0.85,
                "latency_ms": 50,
                "confidence_score": 0.80
            }
        }

        self.validation_history: Dict[str, List[ValidationResult]] = {}

    async def validate_model(
        self,
        component: str,
        model: Any,
        validation_data: Dict[str, Any],
        validation_config: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        try:
            logger.info(f"Starting validation for {component} model...")

            validation_method = getattr(self, f"_validate_{component}_model", None)
            if not validation_method:
                raise ValueError(f"No validation method found for component: {component}")

            validation_result = await validation_method(model, validation_data, validation_config)

            if component not in self.validation_history:
                self.validation_history[component] = []
            self.validation_history[component].append(validation_result)

            max_history = 100
            if len(self.validation_history[component]) > max_history:
                self.validation_history[component] = self.validation_history[component][-max_history:]

            return validation_result

        except Exception as e:
            logger.error(f"Model validation failed for {component}: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[str(e)],
                metrics={},
                timestamp=datetime.now()
            )

    async def _validate_model_selector_model(
        self,
        model: Dict[str, Any],
        validation_data: Dict[str, Any],
        validation_config: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        try:
            validation_result = ValidationResult(is_valid=True)
            metrics = {}
            warnings = []

            if "content_analyzer" in model:
                content_metrics = await self._validate_content_analyzer(
                    model["content_analyzer"],
                    validation_data.get("content_samples", [])
                )
                metrics.update(content_metrics)

            if "performance_predictor" in model:
                performance_metrics = await self._validate_performance_predictor(
                    model["performance_predictor"],
                    validation_data.get("performance_samples", [])
                )
                metrics.update(performance_metrics)

            if "routing_optimizer" in model:
                routing_metrics = await self._validate_routing_optimizer(
                    model["routing_optimizer"],
                    validation_data.get("routing_samples", [])
                )
                metrics.update(routing_metrics)

            thresholds = self.validation_thresholds["model_selector"]
            passed = all(
                metrics.get(metric, 0) >= threshold
                for metric, threshold in thresholds.items()
                if metric in metrics
            )

            for metric, value in metrics.items():
                if metric in thresholds:
                    threshold = thresholds[metric]
                    if 0.9 * threshold <= value < threshold:
                        warnings.append(f"{metric} is close to threshold: {value:.3f} >= {threshold}")

            validation_result.is_valid = passed
            validation_result.metrics = metrics
            validation_result.warnings = warnings

            return validation_result

        except Exception as e:
            logger.error(f"Model selector validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[str(e)],
                metrics={},
                timestamp=datetime.now()
            )

    async def _validate_nlp_processor_model(
        self,
        model: Any,
        validation_data: Dict[str, Any],
        validation_config: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        try:
            validation_result = ValidationResult(is_valid=True)
            metrics = {}
            warnings = []

            commands = validation_data.get("commands", [])
            expected_intents = validation_data.get("expected_intents", [])

            if not commands or not expected_intents:
                validation_result.add_error("Missing validation data")
                return validation_result

            correct_intents = 0
            total_latency = 0

            for cmd, expected in zip(commands, expected_intents):
                start_time = datetime.now()
                try:
                    result = await model.process_text(cmd)
                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    total_latency += latency

                    if (
                        result.category == expected["category"] and
                        result.action == expected["action"]
                    ):
                        correct_intents += 1
                except Exception as e:
                    warnings.append(f"Command processing failed: {str(e)}")

            accuracy = correct_intents / len(commands)
            avg_latency = total_latency / len(commands)

            metrics = {
                "accuracy": accuracy,
                "latency_ms": avg_latency
            }

            is_valid = (
                accuracy >= self.validation_thresholds["memory"]["accuracy"] and
                avg_latency <= self.validation_thresholds["memory"]["latency_ms"]
            )

            validation_result.is_valid = is_valid
            validation_result.metrics = metrics
            validation_result.warnings = warnings

            return validation_result

        except Exception as e:
            logger.error(f"NLP processor validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[str(e)],
                metrics={},
                timestamp=datetime.now()
            )

    async def _validate_action_executor_model(
        self,
        model: Any,
        validation_data: Dict[str, Any],
        validation_config: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        try:
            validation_result = ValidationResult(is_valid=True)
            metrics = {}
            warnings = []

            actions = validation_data.get("actions", [])
            expected_results = validation_data.get("expected_results", [])

            if not actions or not expected_results:
                validation_result.add_error("Missing validation data")
                return validation_result

            successful_actions = 0
            total_latency = 0

            for action, expected in zip(actions, expected_results):
                start_time = datetime.now()
                try:
                    result = await model.execute_action(
                        action_type=action["type"],
                        action=action["action"],
                        parameters=action["params"]
                    )
                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    total_latency += latency

                    if result["status"] == expected["status"]:
                        successful_actions += 1
                except Exception as e:
                    warnings.append(f"Action execution failed: {str(e)}")

            success_rate = successful_actions / len(actions)
            avg_latency = total_latency / len(actions)

            metrics = {
                "success_rate": success_rate,
                "latency_ms": avg_latency
            }

            is_valid = (
                success_rate >= self.validation_thresholds["action"]["accuracy"] and
                success_rate >= self.validation_thresholds["action"]["safety_score"]
            )

            validation_result.is_valid = is_valid
            validation_result.metrics = metrics
            validation_result.warnings = warnings

            return validation_result

        except Exception as e:
            logger.error(f"Action executor validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[str(e)],
                metrics={},
                timestamp=datetime.now()
            )

    async def _validate_memory_manager_model(
        self,
        model: Any,
        validation_data: Dict[str, Any],
        validation_config: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        try:
            validation_result = ValidationResult(is_valid=True)
            metrics = {}
            warnings = []

            test_memories = validation_data.get("test_memories", [])
            expected_results = validation_data.get("expected_results", [])

            if not test_memories or not expected_results:
                validation_result.add_error("Missing validation data")
                return validation_result

            successful_ops = 0
            total_latency = 0

            for memory, expected in zip(test_memories, expected_results):
                start_time = datetime.now()
                try:
                    await model.store(memory["key"], memory["value"])
                    retrieved = await model.retrieve(memory["key"])

                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    total_latency += latency

                    if retrieved == memory["value"]:
                        successful_ops += 1
                except Exception as e:
                    warnings.append(f"Memory operation failed: {str(e)}")

            accuracy = successful_ops / len(test_memories)
            avg_latency = total_latency / len(test_memories)

            metrics = {
                "accuracy": accuracy,
                "latency_ms": avg_latency
            }

            is_valid = (
                accuracy >= self.validation_thresholds["memory"]["accuracy"] and
                avg_latency <= self.validation_thresholds["memory"]["latency_ms"]
            )

            validation_result.is_valid = is_valid
            validation_result.metrics = metrics
            validation_result.warnings = warnings

            return validation_result

        except Exception as e:
            logger.error(f"Memory manager validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[str(e)],
                metrics={},
                timestamp=datetime.now()
            )

    async def _validate_onnx_integration_model(
        self,
        model: Any,
        validation_data: Dict[str, Any],
        validation_config: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        try:
            validation_result = ValidationResult(is_valid=True)
            metrics = {}
            warnings = []

            test_inputs = validation_data.get("test_inputs", [])
            expected_shapes = validation_data.get("expected_output_shapes", [])

            if not test_inputs or not expected_shapes:
                validation_result.add_error("Missing validation data")
                return validation_result

            successful_inferences = 0
            total_latency = 0

            for test_input, expected_shape in zip(test_inputs, expected_shapes):
                start_time = datetime.now()
                try:
                    output = await model.run_inference(test_input)
                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    total_latency += latency

                    if output.shape == expected_shape:
                        successful_inferences += 1
                except Exception as e:
                    warnings.append(f"ONNX inference failed: {str(e)}")

            success_rate = successful_inferences / len(test_inputs)
            avg_latency = total_latency / len(test_inputs)

            metrics = {
                "success_rate": success_rate,
                "latency_ms": avg_latency
            }

            is_valid = success_rate >= self.validation_thresholds["model_selector"]["selection_accuracy"]

            validation_result.is_valid = is_valid
            validation_result.metrics = metrics
            validation_result.warnings = warnings

            return validation_result

        except Exception as e:
            logger.error(f"ONNX integration validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[str(e)],
                metrics={},
                timestamp=datetime.now()
            )

    async def _validate_learning_system_model(
        self,
        model: Any,
        validation_data: Dict[str, Any],
        validation_config: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        try:
            validation_result = ValidationResult(is_valid=True)
            metrics = {}
            warnings = []

            test_experiences = validation_data.get("test_experiences", [])
            expected_progress = validation_data.get("expected_learning_progress", {})

            if not test_experiences or not expected_progress:
                validation_result.add_error("Missing validation data")
                return validation_result

            total_loss = 0
            correct_predictions = 0
            total_latency = 0

            for experience in test_experiences:
                start_time = datetime.now()
                try:
                    loss = await model.learn(experience)
                    total_loss += loss

                    prediction = await model.predict(experience["state"])
                    if prediction == experience["action"]:
                        correct_predictions += 1

                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    total_latency += latency
                except Exception as e:
                    warnings.append(f"Learning operation failed: {str(e)}")

            avg_loss = total_loss / len(test_experiences)
            accuracy = correct_predictions / len(test_experiences)
            avg_latency = total_latency / len(test_experiences)

            metrics = {
                "avg_loss": avg_loss,
                "accuracy": accuracy,
                "latency_ms": avg_latency
            }

            is_valid = (
                avg_loss <= expected_progress["loss_threshold"] and
                accuracy >= expected_progress["accuracy_threshold"]
            )

            validation_result.is_valid = is_valid
            validation_result.metrics = metrics
            validation_result.warnings = warnings

            return validation_result

        except Exception as e:
            logger.error(f"Learning system validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[str(e)],
                metrics={},
                timestamp=datetime.now()
            )

    async def _validate_content_analyzer(
        self,
        model: Any,
        content_samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        try:
            metrics = {}

            inputs = np.array([sample["input"] for sample in content_samples])
            expected = np.array([sample["expected"] for sample in content_samples])

            outputs = await self._run_inference(model, inputs)

            metrics["feature_accuracy"] = self._calculate_feature_accuracy(outputs, expected)
            metrics["latency_ms"] = self._calculate_average_latency(model, inputs)

            return metrics

        except Exception as e:
            logger.error(f"Content analyzer validation failed: {e}")
            return {"error_rate": 1.0}

    async def _validate_performance_predictor(
        self,
        model: Any,
        performance_samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        try:
            metrics = {}

            features = np.array([sample["features"] for sample in performance_samples])
            expected = np.array([sample["expected"] for sample in performance_samples])

            predictions = await self._run_inference(model, features)

            metrics["prediction_accuracy"] = self._calculate_prediction_accuracy(predictions, expected)
            metrics["confidence_score"] = self._calculate_confidence_score(predictions, expected)

            return metrics

        except Exception as e:
            logger.error(f"Performance predictor validation failed: {e}")
            return {"error_rate": 1.0}

    async def _validate_routing_optimizer(
        self,
        model: Any,
        routing_samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        try:
            metrics = {}

            features = np.array([sample["features"] for sample in routing_samples])
            expected = np.array([sample["expected"] for sample in routing_samples])

            selections = await self._run_inference(model, features)

            metrics["selection_accuracy"] = self._calculate_selection_accuracy(selections, expected)
            metrics["optimization_score"] = self._calculate_optimization_score(selections, expected)

            return metrics

        except Exception as e:
            logger.error(f"Routing optimizer validation failed: {e}")
            return {"error_rate": 1.0}

    async def _run_inference(
        self,
        model: Any,
        inputs: np.ndarray
    ) -> np.ndarray:
        try:
            if hasattr(model, "run"):
                return await model.run({"input": inputs})
            else:
                raise ValueError("Model does not have run method")

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def _calculate_feature_accuracy(
        self,
        outputs: np.ndarray,
        expected: np.ndarray
    ) -> float:
        try:
            similarities = np.sum(outputs * expected, axis=1) / (
                np.linalg.norm(outputs, axis=1) * np.linalg.norm(expected, axis=1)
            )
            return float(np.mean(similarities))

        except Exception as e:
            logger.error(f"Error calculating feature accuracy: {e}")
            return 0.0

    def _calculate_prediction_accuracy(
        self,
        predictions: np.ndarray,
        expected: np.ndarray
    ) -> float:
        try:
            mse = np.mean((predictions - expected) ** 2)
            return float(1.0 - min(mse / 100.0, 1.0))

        except Exception as e:
            logger.error(f"Error calculating prediction accuracy: {e}")
            return 0.0

    def _calculate_selection_accuracy(
        self,
        selections: np.ndarray,
        expected: np.ndarray
    ) -> float:
        try:
            correct = np.argmax(selections, axis=1) == np.argmax(expected, axis=1)
            return float(np.mean(correct))

        except Exception as e:
            logger.error(f"Error calculating selection accuracy: {e}")
            return 0.0

    def _calculate_optimization_score(
        self,
        selections: np.ndarray,
        expected: np.ndarray
    ) -> float:
        try:
            kl_div = np.sum(
                expected * np.log(expected / np.clip(selections, 1e-10, 1.0)),
                axis=1
            )
            return float(1.0 - min(np.mean(kl_div), 1.0))

        except Exception as e:
            logger.error(f"Error calculating optimization score: {e}")
            return 0.0

    def _calculate_confidence_score(
        self,
        predictions: np.ndarray,
        expected: np.ndarray
    ) -> float:
        try:
            confidence = np.max(predictions, axis=1)
            accuracy = np.argmax(predictions, axis=1) == np.argmax(expected, axis=1)
            weighted_confidence = confidence * accuracy
            return float(np.mean(weighted_confidence))

        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0

    def _calculate_average_latency(
        self,
        model: Any,
        inputs: np.ndarray
    ) -> float:
        try:
            import time
            latencies = []

            for input_data in inputs:
                start_time = time.time()
                _ = model.run({"input": input_data[np.newaxis, :]})
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)

            return float(np.mean(latencies))

        except Exception as e:
            logger.error(f"Error calculating average latency: {e}")
            return float('inf')

    async def validate_onnx_component(
        self,
        component_name: str,
        component: Any,
        validation_data: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        try:
            logger.info(f"Starting validation for {component_name}...")

            required_methods = ['initialize', 'get_embedding']
            for method in required_methods:
                if not hasattr(component, method):
                    return ValidationResult(
                        is_valid=False,
                        errors=[f"Component missing required method: {method}"],
                        metrics={}
                    )

            model_dir = getattr(component, 'model_dir', None)
            if model_dir:
                model_path = Path(model_dir) / "embedding.onnx"
                if not model_path.exists():
                    return ValidationResult(
                        is_valid=False,
                        errors=[f"Model file not found: {model_path}"],
                        metrics={}
                    )

            if hasattr(component, 'get_embedding'):
                try:
                    embedding = await component.get_embedding("test input")
                    if embedding is None:
                        return ValidationResult(
                            is_valid=False,
                            errors=["Embedding generation failed"],
                            metrics={}
                        )
                except Exception as e:
                    return ValidationResult(
                        is_valid=False,
                        errors=[f"Embedding validation failed: {str(e)}"],
                        metrics={}
                    )

            return ValidationResult(
                is_valid=True,
                metrics={
                    "model_loaded": 1.0,
                    "embedding_test": 1.0
                }
            )

        except Exception as e:
            logger.error(f"Validation failed for {component_name}: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[str(e)],
                metrics={}
            )