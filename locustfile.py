from locust import HttpUser, task, constant_pacing, between
from locust.env import Environment
from locust.stats import stats_printer
from locust.runners import Runner
from matplotlib import pyplot as plt
import argparse
import datetime
import gevent
import random
import pandas as pd


class LLMUser(HttpUser):
    wait_time = between(0, 5)

    # @task
    # def generate(self):
    #     input_len = random.randint(200, 236)
    #     input_payload = {
    #         "tokens": [random.randint(1, 1000)] * input_len,
    #         "max_tokens": 20,
    #         "min_tokens": 20,
    #         "n": 8,
    #         "top_p": 0.8,
    #         "top_k": 32,
    #         "seed": [42],
    #     }
    #     self.client.post("/inference", json=input_payload)
    def on_start(self):
        pass
    
    @task
    def generate(self):
        text = "What is Triton Inference Server?"
        input_payload = {
            "text_input": text,
            "parameters": {
                "stream": False,
                "temperature": 0
            }
        }
        self.client.post("/v2/models/opt-125m/generate", json=input_payload)


def stats_history(runner: Runner) -> None:
    """Save current stats info to history for charts of report."""
    while True:
        stats = runner.stats
        if not stats.total.use_response_times_cache:
            break
        if runner.state != "stopped":
            r = {
                "time": datetime.datetime.utcnow().strftime("%H:%M:%S"),
                "current_rps": stats.total.current_rps or 0,
                "current_fail_per_sec": stats.total.current_fail_per_sec or 0,
                "response_time_percentile_95": stats.total.get_current_response_time_percentile(
                    0.95
                ),
                "response_time_percentile_90": stats.total.get_current_response_time_percentile(
                    0.90
                ),
                "avg_response_time": stats.total.avg_response_time,
                "user_count": runner.user_count or 0,
            }
            stats.history.append(r)
        gevent.sleep(0.5)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--max_user', type=int, default=10)
    parser.add_argument('--spawn_rate', type=int, default=1)
    parser.add_argument('--duration', type=int, default=300)
    parser.add_argument('--figure_path', type=str, default='./figure.jpg')
    return parser.parse_args()


def main(args: argparse.Namespace):
    env = Environment(user_classes=[LLMUser], host=args.host)
    env.create_local_runner()

    gevent.spawn(stats_printer(env.stats))
    gevent.spawn(stats_history, env.runner)

    env.runner.start(args.max_user, spawn_rate=args.spawn_rate)
    gevent.spawn_later(args.duration, lambda: env.runner.quit())
    env.runner.greenlet.join()

    df = pd.DataFrame(env.stats.history)

    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(
        df["current_rps"], df["response_time_percentile_90"], ".-", label="P90 Latency"
    )
    plt.plot(
        df["current_rps"], df["response_time_percentile_95"], ".-", label="P95 Latency"
    )

    plt.legend(loc="upper left")
    plt.title("Latency/Throuput")
    plt.xlabel("Request/sec")
    plt.ylabel("Latency")
    plt.grid(True)
    plt.savefig(args.figure_path)


if __name__ == "__main__":
    main(parse_arguments())