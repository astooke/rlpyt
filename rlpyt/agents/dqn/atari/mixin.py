

class AtariMixin(object):

    def make_env_to_model_kwargs(self, env_spec):
        return dict(image_shape=env_spec.observation_space.shape,
                    output_size=env_spec.action_space.n)
