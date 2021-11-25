import tempfile
import timeout_decorator
from loguru import logger

from evaluator.evaluator import Learn2RaceEvaluator
from config import SubmissionConfig, SimulatorConfig, EnvConfig


pre_evaluate_model_file, pre_evaluate_model_path = tempfile.mkstemp()


def pre_evaluate(evaluator):
    try:
        evaluator.pre_evaluate()
    except timeout_decorator.TimeoutError:
        logger.info("Stopping pre-evaluation run")

    evaluator.save_agent_model(pre_evaluate_model_path)


def rollout(evaluator):
    evaluator.load_agent_model(pre_evaluate_model_path)
    scores = evaluator.evaluate()
    logger.success(f"Average metrics: {scores}")


def run_evaluation():
    submission_config = SubmissionConfig()
    simulator_config = SimulatorConfig()
    env_config = EnvConfig()

    evaluator = Learn2RaceEvaluator(
        submission_config=submission_config,
        sim_config=simulator_config,
        env_config=env_config,
    )

    evaluator.create_env(["Thruxton"])
    evaluator.init_agent()

    pre_evaluate(evaluator)
    rollout(evaluator)


if __name__ == "__main__":
    run_evaluation()
