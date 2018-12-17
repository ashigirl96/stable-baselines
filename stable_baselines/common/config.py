

class Config:
  def __init__(self) -> None:
    # For type hints
    # General information
    self.name: str = None
    self.parser = argparse.ArgumentParser()
    self.optimizer_fn: Opt = None
    self.actor_optimizer_fn: Opt = None
    self.critic_optimizer_fn: Opt = None
    self.network_fn: Mod = None
    self.discount = 0.99
    # Environment(Task) information
    self.task_fn: CTask = None
    self.task: Task = None
    self.task_name: str = None
    self.state_dim: np.ndarray = None
    self.action_dim: np.ndarray = None
    # Logging information
    self.logger: Logger = None
    self.log_interval = int(1e3)
    self.save_interval = 0
    self.eval_interval = 0
    self.eval_episodes = 10
    self.iteration_log_interval = 30
    # Actor Critic networks
    self.actor_network_fn: Mod = None
    self.critic_network_fn: Mod = None
    # Off Policy Experience Replay
    self.replay_fn: CReplay = None
    self.lock: torch.multiprocessing.Lock = None
    self.min_memory_size: int = None
    self.double_q = False
    self.target_network_update_freq: int = None
    self.exploration_steps: int = None
    # Continuous action noise distribution
    self.random_process_fn: CRandomProcess = None
    # For Pixel Atari
    self.history_length: int = None
    self.tag = 'vanilla'
    # For Parallelized Task (e.g. A2C)
    self.num_workers = 1
    # For Policy Gradient and Actor Critic
    self.gradient_clip: float = None
    self.entropy_weight = 0.01
    self.value_loss_weight = 1.0
    self.use_gae = False
    self.gae_tau = 1.0
    self.target_network_mix = 0.001
    self.state_normalizer = RescaleNormalizer()
    self.reward_normalizer = RescaleNormalizer()
    # Hyper-Parameters for
    self.max_steps = 0
    self.rollout_length: int = None
    self.categorical_v_min: float = None
    self.categorical_v_max: float = None
    self.categorical_n_atoms = 51
    self.num_quantiles: float = None
    self.optimization_epochs = 4
    self.num_mini_batches = 32
    self.termination_regularizer = 0
    self.sgd_update_frequency: int = None
    # Scheduler
    self.random_action_prob: Schedule = None
    self.async_actor = True
    # Temporal property
    self.__eval_env = None
    self.__log_dir = Path("/tmp/log")

  @property
  def log_dir(self) -> Path:
    return self.__log_dir

  @log_dir.setter
  def log_dir(self, log_directory_name: str):
    assert self.name is not None and \
           self.task_name is not None, "Must set name and task_name before set log_dir"
    path_log_dir = Path(log_directory_name)
    # If the indicated directory doesn't exist
    if not path_log_dir.exists() or self.name in log_directory_name and self.task_name in log_directory_name:
      self.__log_dir = path_log_dir
    else:
      name = '{}-{}-{}'.format(self.name, self.task_name, get_time_str())
      self.__log_dir = path_log_dir / name
    (self.__log_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    self.logger = get_logger(log_dir=self.__log_dir)
    self.logger.info('logging to {}'.format(self.__log_dir))

  @property
  def eval_env(self) -> Task:
    return self.__eval_env

  @eval_env.setter
  def eval_env(self, env) -> None:
    self.__eval_env = env
    self.state_dim = env.state_dim
    self.action_dim = env.action_dim
    self.task_name = env.name

  def add_argument(self, *args, **kwargs) -> None:
    self.parser.add_argument(*args, **kwargs)

  def save(self, name=None):
    name = name or self.log_dir.as_posix() + '/config.pkl'
    logger = self.__dict__.pop('logger')
    with open(name, 'wb') as config:
      cloudpickle.dump(self.__dict__, config, cloudpickle.DEFAULT_PROTOCOL)
    self.__dict__['logger'] = logger

  def load(self, name):
    if not Path(name).exists():
      raise ValueError(f'The file {name} is not exist.')
    with open(name, 'rb') as config:
      config_dict = cloudpickle.load(config)
    self._merge(config_dict)

  def _merge(self, config_dict=None) -> None:
    if config_dict is None:
      args = self.parser.parse_args()
      config_dict = args.__dict__
      del self.parser
    for key in config_dict.keys():
      setattr(self, key, config_dict[key])

  @contextlib.contextmanager
  def set_log_directory(self, name):
    assert name is not None, "Name the config name!"
    try:
      yield
    finally:
      self.name = name
      self._merge()
