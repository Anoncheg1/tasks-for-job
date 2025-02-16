import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time

from .model_validator import ModelValidator, ValidationResult

logger = logging.getLogger(__name__)

@dataclass
class CrossComponentValidationResult:
    is_valid: bool
    component_results: Dict[str, ValidationResult]
    interaction_metrics: Dict[str, float]
    validation_time: datetime
    error_details: Optional[str] = None
    warnings: List[str] = None

@dataclass
class StressTestResult:
    passed: bool
    throughput: float
    latency_p95: float
    error_rate: float
    resource_usage: Dict[str, float]
    test_duration: float
    validation_time: datetime

class EnhancedValidator(ModelValidator):
    def __init__(self):
        super().__init__()
        self.cross_component_thresholds = {
            "end_to_end_latency_ms": 500,
            "interaction_consistency": 0.9,
            "error_propagation_rate": 0.1
        }

        self.stress_test_thresholds = {
            "min_throughput": 100,
            "max_latency_p95": 1000,
            "max_error_rate": 0.01,
            "max_cpu_usage": 0.8,
            "max_memory_usage": 0.8
        }

        self.cross_validation_history: List[CrossComponentValidationResult] = []
        self.stress_test_history: List[StressTestResult] = []

    async def validate_cross_component(
        self,
        components: Dict[str, Any],
        validation_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> CrossComponentValidationResult:
        try:
            logger.info("Starting cross-component validation...")

            component_results = {}
            for name, component in components.items():
                result = await self.validate_model(
                    name, component, validation_data.get(name, {})
                )
                component_results[name] = result

            interaction_metrics = await self._validate_interactions(
                components, validation_data
            )

            passed = all(
                interaction_metrics.get(metric, float('inf')) <= threshold
                for metric, threshold in self.cross_component_thresholds.items()
            )

            warnings = []
            for metric, value in interaction_metrics.items():
                if metric in self.cross_component_thresholds:
                    threshold = self.cross_component_thresholds[metric]
                    if 0.8 * threshold <= value <= threshold:
                        warnings.append(
                            f"{metric} is approaching threshold: {value:.2f} / {threshold}"
                        )

            result = CrossComponentValidationResult(
                is_valid=passed,
                component_results=component_results,
                interaction_metrics=interaction_metrics,
                validation_time=datetime.now(),
                warnings=warnings
            )

            self.cross_validation_history.append(result)
            if len(self.cross_validation_history) > 100:
                self.cross_validation_history = self.cross_validation_history[-100:]

            return result

        except Exception as e:
            logger.error(f"Cross-component validation failed: {e}")
            return CrossComponentValidationResult(
                is_valid=False,
                component_results={},
                interaction_metrics={},
                validation_time=datetime.now(),
                error_details=str(e)
            )

    async def run_stress_test(
        self,
        components: Dict[str, Any],
        test_data: Dict[str, Any],
        duration_seconds: int = 300,
        concurrent_users: int = 100
    ) -> StressTestResult:
        try:
            logger.info(f"Starting stress test with {concurrent_users} concurrent users...")
            start_time = datetime.now()

            async with ThreadPoolExecutor() as executor:
                tasks = []
                for _ in range(concurrent_users):
                    task = asyncio.create_task(
                        self._simulate_user_load(components, test_data)
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks)

            total_requests = sum(r["requests"] for r in results)
            total_errors = sum(r["errors"] for r in results)
            latencies = [lat for r in results for lat in r["latencies"]]

            throughput = total_requests / duration_seconds
            error_rate = total_errors / total_requests if total_requests > 0 else 1.0
            latency_p95 = np.percentile(latencies, 95) if latencies else float('inf')

            resource_usage = await self._measure_resource_usage()

            passed = (
                throughput >= self.stress_test_thresholds["min_throughput"] and
                latency_p95 <= self.stress_test_thresholds["max_latency_p95"] and
                error_rate <= self.stress_test_thresholds["max_error_rate"] and
                resource_usage["cpu"] <= self.stress_test_thresholds["max_cpu_usage"] and
                resource_usage["memory"] <= self.stress_test_thresholds["max_memory_usage"]
            )

            result = StressTestResult(
                passed=passed,
                throughput=throughput,
                latency_p95=latency_p95,
                error_rate=error_rate,
                resource_usage=resource_usage,
                test_duration=duration_seconds,
                validation_time=datetime.now()
            )

            self.stress_test_history.append(result)
            if len(self.stress_test_history) > 50:
                self.stress_test_history = self.stress_test_history[-50:]

            return result

        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            return StressTestResult(
                passed=False,
                throughput=0,
                latency_p95=float('inf'),
                error_rate=1.0,
                resource_usage={},
                test_duration=0,
                validation_time=datetime.now()
            )

    async def _validate_interactions(
        self,
        components: Dict[str, Any],
        validation_data: Dict[str, Any]
    ) -> Dict[str, float]:
        metrics = {}

        try:
            latencies = []
            for _ in range(100):
                start_time = datetime.now()
                await self._run_end_to_end_pipeline(components, validation_data)
                end_time = datetime.now()
                latencies.append((end_time - start_time).total_seconds() * 1000)

            metrics["end_to_end_latency_ms"] = np.mean(latencies)

            consistency_scores = []
            for _ in range(50):
                score = await self._check_interaction_consistency(
                    components, validation_data
                )
                consistency_scores.append(score)

            metrics["interaction_consistency"] = np.mean(consistency_scores)

            error_rates = []
            for _ in range(30):
                rate = await self._measure_error_propagation(
                    components, validation_data
                )
                error_rates.append(rate)

            metrics["error_propagation_rate"] = np.mean(error_rates)

            return metrics

        except Exception as e:
            logger.error(f"Interaction validation failed: {e}")
            return {
                "end_to_end_latency_ms": float('inf'),
                "interaction_consistency": 0.0,
                "error_propagation_rate": 1.0
            }

    async def _simulate_user_load(
        self,
        components: Dict[str, Any],
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        results = {
            "requests": 0,
            "errors": 0,
            "latencies": []
        }

        try:
            while True:
                start_time = datetime.now()
                try:
                    await self._run_end_to_end_pipeline(components, test_data)
                    results["requests"] += 1
                    end_time = datetime.now()
                    latency = (end_time - start_time).total_seconds() * 1000
                    results["latencies"].append(latency)
                except Exception:
                    results["errors"] += 1

                await asyncio.sleep(np.random.exponential(0.1))

        except asyncio.CancelledError:
            return results

    async def _measure_resource_usage(self) -> Dict[str, float]:
        try:
            import psutil

            cpu_percent = psutil.cpu_percent() / 100.0
            memory = psutil.virtual_memory()
            memory_used = memory.used / memory.total

            return {
                "cpu": cpu_percent,
                "memory": memory_used
            }

        except Exception as e:
            logger.error(f"Failed to measure resource usage: {e}")
            return {
                "cpu": 0.0,
                "memory": 0.0
            }

    async def _run_end_to_end_pipeline(
        self,
        components: Dict[str, Any],
        test_data: Dict[str, Any]
    ) -> bool:
        try:
            required_components = {
                "nlp_processor": False,
                "action_executor": False,
                "memory_manager": False,
                "onnx_integration": False
            }
            
            for component in required_components:
                if component in components and components[component] is not None:
                    required_components[component] = True

            if required_components["nlp_processor"] and test_data.get("nlp_processor", {}).get("commands"):
                command = test_data["nlp_processor"]["commands"][0]
                intent = await components["nlp_processor"].process_text(command)
                if not intent:
                    return False
                    
            if required_components["action_executor"] and test_data.get("action_executor", {}).get("actions"):
                action = test_data["action_executor"]["actions"][0]
                success = await components["action_executor"].execute_action(action)
                if not success:
                    return False
                    
            if required_components["memory_manager"] and test_data.get("memory_manager", {}).get("test_memories"):
                memory = test_data["memory_manager"]["test_memories"][0]
                success = await components["memory_manager"].store(memory)
                if not success:
                    return False
                    
            if required_components["onnx_integration"] and test_data.get("onnx_integration", {}).get("test_models"):
                for model_name, model_data in test_data["onnx_integration"]["test_models"].items():
                    outputs = await components["onnx_integration"].run_inference(
                        model_id=model_name,
                        inputs=model_data["inputs"]
                    )
                    
                    if not outputs or not any(
                        output.shape == model_data["expected_shape"] 
                        for output in outputs.values()
                    ):
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"End-to-end pipeline test failed: {str(e)}")
            return False

    async def _check_interaction_consistency(
        self,
        components: Dict[str, Any],
        test_data: Dict[str, Any]
    ) -> float:
        try:
            consistency_scores = []
            
            if all(k in components and components[k] is not None for k in ["nlp_processor", "action_executor"]):
                try:
                    command = test_data.get("nlp_processor", {}).get("commands", [None])[0]
                    if command:
                        intent = await components["nlp_processor"].process_text(command)
                        if intent:
                            result = await components["action_executor"].execute_action(
                                action_type=intent.category,
                                action=intent.action,
                                parameters=intent.parameters
                            )
                            consistency_scores.append(1.0 if result and result.get("status") == "success" else 0.0)
                except Exception as e:
                    logger.warning(f"NLP-Action consistency check failed: {str(e)}")
                    consistency_scores.append(0.0)
            
            if "memory_manager" in components and components["memory_manager"] is not None:
                try:
                    memory = test_data.get("memory_manager", {}).get("test_memories", [None])[0]
                    if memory:
                        await components["memory_manager"].store(memory["key"], memory["value"])
                        retrieved = await components["memory_manager"].retrieve(memory["key"])
                        consistency_scores.append(1.0 if retrieved == memory["value"] else 0.0)
                except Exception as e:
                    logger.warning(f"Memory consistency check failed: {str(e)}")
                    consistency_scores.append(0.0)
            
            if "onnx_integration" in components and components["onnx_integration"] is not None:
                try:
                    test_input = test_data.get("onnx_integration", {}).get("test_inputs", [None])[0]
                    if test_input is not None:
                        output1 = await components["onnx_integration"].run_inference(test_input)
                        output2 = await components["onnx_integration"].run_inference(test_input)
                        consistency_scores.append(1.0 if np.array_equal(output1, output2) else 0.0)
                except Exception as e:
                    logger.warning(f"ONNX consistency check failed: {str(e)}")
                    consistency_scores.append(0.0)
            
            return np.mean(consistency_scores) if consistency_scores else 0.0

        except Exception as e:
            logger.error(f"Interaction consistency check failed: {str(e)}")
            return 0.0

    async def _measure_error_propagation(
        self,
        components: Dict[str, Any],
        test_data: Dict[str, Any]
    ) -> float:
        try:
            error_counts = 0
            total_tests = 0
            
            if all(k in components and components[k] is not None for k in ["nlp_processor", "action_executor"]):
                total_tests += 1
                try:
                    result = await components["nlp_processor"].process_text("@#$%^&*")
                    error_counts += 1 if result is None or result.confidence < 0.5 else 0
                except Exception:
                    error_counts += 1
            
            if "memory_manager" in components and components["memory_manager"] is not None:
                total_tests += 1
                try:
                    result = await components["memory_manager"].retrieve("nonexistent_key_" + str(time.time()))
                    error_counts += 1 if result is None else 0
                except Exception:
                    error_counts += 1
            
            if "onnx_integration" in components and components["onnx_integration"] is not None:
                total_tests += 1
                try:
                    invalid_input = np.zeros((1, 1))
                    result = await components["onnx_integration"].run_inference(invalid_input)
                    error_counts += 1 if result is None else 0
                except Exception:
                    error_counts += 1
            
            return error_counts / total_tests if total_tests > 0 else 1.0

        except Exception as e:
            logger.error(f"Error propagation measurement failed: {str(e)}")
            return 1.0
