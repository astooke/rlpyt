

class AtariMixin:

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)
