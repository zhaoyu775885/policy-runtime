# policy-runtime

A minimal client/server runtime for decoupling VLA policies from simulation environments.

```
┌─────────────────┐        TCP (JSON)        ┌─────────────────┐
│  Client (Env)   │ ───────────────────────── │  Server (Policy) │
│  EnvAdapter     │  obs → action             │  PolicyAdapter   │
└─────────────────┘                           └─────────────────┘
```

## Install

```bash
pip install policy-runtime
```

## Quick Start

### 1. Implement your adapters

```python
# my_policy.py
from policy_runtime import PolicyAdapter

class MyPolicy(PolicyAdapter):
    def start_episode(self, episode_info):
        self.model = load_model()

    def get_action(self, observation):
        return self.model.predict(observation)

    def end_episode(self):
        pass
```

```python
# my_env.py
from policy_runtime import EnvAdapter

class MyEnv(EnvAdapter):
    def reset(self):
        self.env = make_env()
        return {"task": "pick_and_place"}

    def get_observation(self):
        return {"image": self.env.render(), "qpos": self.env.qpos}

    def apply_action(self, action):
        self.env.step(action)

    def is_done(self):
        return self.env.done
```

### 2. Run

```bash
# Terminal 1: start policy server
python -m policy_runtime.server --port 5555 --adapter my_policy:MyPolicy

# Terminal 2: start env client
python -m policy_runtime.client --port 5555 --adapter my_env:MyEnv --num-episodes 10
```

## License

MIT
