from gym.envs.registration import register

# register(
# 	id='movingscreen-v0',
# 	entry_point='gym_movingscreen.envs:MovingScreenEnv'
# )

register(
	id='movingscreen-v1',
	entry_point='gym_movingscreen.envs:MovingScreenEnvXY'
)

register(
	id='movingscreen-v2',
	entry_point='gym_movingscreen.envs:MovingScreenEnvFace'
)