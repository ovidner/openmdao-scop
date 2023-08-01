from functools import wraps

import openmdao.api as om

from .modelling import Param, add_input_param, add_output_param


class FuncComp(om.ExplicitComponent):
    def initialize(self):
        # TODO: use type hints to infer inputs and outputs
        self.options.declare("func")
        self.options.declare("inputs", types=list)
        self.options.declare("outputs", types=list)

    def setup(self):
        self.input_params = {}
        self.output_params = {}
        self.func = self.options["func"]
        for input_ in self.options["inputs"]:
            add_input_param(self, input_)
            self.input_params[input_.name] = input_

        for output in self.options["outputs"]:
            add_output_param(self, output)
            self.output_params[output.name] = output

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        kwargs = {
            name: discrete_inputs[name] if param.discrete else inputs[name]
            for name, param in self.input_params.items()
        }
        # TODO: inspect the function signature and support non-kwargs
        output = self.func(**kwargs)
        # TODO: add safety checks!
        for idx, (name, param) in enumerate(self.output_params.items()):
            if isinstance(output, tuple):
                val = output[idx]
            elif isinstance(output, dict):
                val = output[name]
            else:
                val = output

            if param.discrete:
                discrete_outputs[name] = val
            else:
                outputs[name] = val


def func_comp(inputs: list[Param], outputs: list[Param]):
    def decorator(func):
        return FuncComp(func=func, inputs=inputs, outputs=outputs)

    return decorator
