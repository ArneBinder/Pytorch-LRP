import allennlp
import torch

from pytorch_lrp.inverter_util import RelevancePropagator


def fullname(o):
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.

    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__qualname__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__class__.__qualname__


class InnvestigateModel(torch.nn.Module):
    """
    ATTENTION:
        Currently, innvestigating a network only works if all
        layers that have to be inverted are specified explicitly
        and registered as a module. If., for example,
        only the functional max_poolnd is used, the inversion will not work.
    """

    def __init__(self, the_model, lrp_exponent=1, beta=.5, epsilon=1e-6,
                 method="e-rule", discard_modules=None):
        """
        Model wrapper for pytorch models to 'innvestigate' them
        with layer-wise relevance propagation (LRP) as introduced by Bach et. al
        (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).
        Given a class level probability produced by the model under consideration,
        the LRP algorithm attributes this probability to the nodes in each layer.
        This allows for visualizing the relevance of input pixels on the resulting
        class probability.

        Args:
            the_model: Pytorch model, e.g. a pytorch.nn.Sequential consisting of
                        different layers. Not all layers are supported yet.
            lrp_exponent: Exponent for rescaling the importance values per node
                            in a layer when using the e-rule method.
            beta: Beta value allows for placing more (large beta) emphasis on
                    nodes that positively contribute to the activation of a given node
                    in the subsequent layer. Low beta value allows for placing more emphasis
                    on inhibitory neurons in a layer. Only relevant for method 'b-rule'.
            epsilon: Stabilizing term to avoid numerical instabilities if the norm (denominator
                    for distributing the relevance) is close to zero.
            method: Different rules for the LRP algorithm, b-rule allows for placing
                    more or less focus on positive / negative contributions, whereas
                    the e-rule treats them equally. For more information,
                    see the paper linked above.
        """
        super(InnvestigateModel, self).__init__()
        self.model = the_model
        self.device = torch.device("cpu", 0)
        self.prediction = None
        self.r_values_per_layer = None
        self.only_max_score = None
        self.discard_modules = discard_modules or []
        # Initialize the 'Relevance Propagator' with the chosen rule.
        # This will be used to back-propagate the relevance values
        # through the layers in the innvestigate method.
        self.inverter = RelevancePropagator(lrp_exponent=lrp_exponent,
                                            beta=beta, method=method, epsilon=epsilon,
                                            device=self.device)

        # Parsing the individual model layers
        self.register_hooks(self.model)
        if method == "b-rule" and float(beta) in (-1., 0):
            which = "positive" if beta == -1 else "negative"
            which_opp = "negative" if beta == -1 else "positive"
            print("WARNING: With the chosen beta value, "
                  "only " + which + " contributions "
                  "will be taken into account.\nHence, "
                  "if in any layer only " + which_opp +
                  " contributions exist, the "
                  "overall relevance will not be conserved.\n")

    def cuda(self, device=None):
        self.device = torch.device("cuda", device)
        self.inverter.device = self.device
        return super(InnvestigateModel, self).cuda(device)

    def cpu(self):
        self.device = torch.device("cpu", 0)
        self.inverter.device = self.device
        return super(InnvestigateModel, self).cpu()

    def register_hooks(self, parent_module, name_prefix='model'):
        """
        Recursively unrolls a model and registers the required
        hooks to save all the necessary values for LRP in the forward pass.

        Args:
            parent_module: Model to unroll and register hooks for.

        Returns:
            None

        """
        print(f'add_rlp_forward_hook@{name_prefix} ({fullname(parent_module)})')
        parent_module.register_forward_hook(self.inverter.get_layer_fwd_hook(parent_module))

        if name_prefix == 'model':
            setattr(parent_module, "name_local", 'model')
            setattr(parent_module, "name_global", 'model')

        if isinstance(parent_module, torch.nn.ReLU):
            parent_module.register_backward_hook(self.relu_hook_function)

        #children_name_global = []
        for name, mod in parent_module.named_children():
            name_w_prefix = f'{name_prefix}.{name}' if name_prefix is not None else name
            if name_w_prefix in self.discard_modules:
                print(f'add_rlp_forward_hook@{name_w_prefix} ({fullname(mod)}): DISCARD')
                continue
            #if not isinstance(mod, self.inverter.allowed_pass_layers):
            #    children_name_global.append(name_w_prefix)
            setattr(mod, "name_local", name)
            setattr(mod, "name_global", name_w_prefix)
            try:
                parent_name = parent_module.name_global
                setattr(mod, "name_parent", parent_name)
            except AttributeError:
                pass
            self.register_hooks(mod, name_prefix=name_w_prefix)

        #setattr(parent_module, "name_children", children_name_global)

    @staticmethod
    def relu_hook_function(module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero.
        """
        return (torch.clamp(grad_in[0], min=0.0),)

    def __call__(self, in_tensor):
        """
        The innvestigate wrapper returns the same prediction as the
        original model, but wraps the model call method in the evaluate
        method to save the last prediction.

        Args:
            in_tensor: Model input to pass through the pytorch model.

        Returns:
            Model output.
        """
        return self.evaluate(in_tensor)

    def evaluate(self, *args, **kwargs):
        """
        Evaluates the model on a new input. The registered forward hooks will
        save all the data that is necessary to compute the relevance per neuron per layer.

        Args:
            in_tensor: New input for which to predict an output.

        Returns:
            Model prediction
        """
        # Reset module list. In case the structure changes dynamically,
        # the module list is tracked for every forward pass.
        self.inverter.reset_module_list()
        self.prediction = self.model(*args, **kwargs)
        return self.prediction

    def get_r_values_per_layer(self):
        if self.r_values_per_layer is None:
            print("No relevances have been calculated yet, returning None in"
                   " get_r_values_per_layer.")
        return self.r_values_per_layer

    def innvestigate(self, in_tensor=None, prediction=None, relevance=None, rel_for_class=None):
        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        Args:
            in_tensor: Input for which to evaluate the LRP algorithm.
                        If input is None, the last evaluation is used.
                        If no evaluation has been performed since initialization,
                        an error is raised.
            prediction: predicted output to use for backward pass
            relevance_tensor: relevance tensor for last layer
            rel_for_class (int): Index of the class for which the relevance
                        distribution is to be analyzed. If None, the 'winning' class
                        is used for indexing.

        Returns:
            Model output and relevances of nodes in the input layer.
            In order to get relevance distributions in other layers, use
            the get_r_values_per_layer method.
        """

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and prediction is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            if in_tensor is not None:
                self.evaluate(in_tensor)

            if prediction is None:
                prediction = self.prediction

            if relevance is None:
                # If no class index is specified, analyze for class
                # with highest prediction.
                if rel_for_class is None:
                    # Default behaviour is innvestigating the output
                    # on an arg-max-basis, if no class is specified.
                    org_shape = prediction.size()
                    # Make sure shape is just a 1D vector per batch example.
                    # TODO: is flatten instead correct?
                    prediction = prediction.view(-1, org_shape[-1])
                    max_v, _ = torch.max(prediction, dim=1, keepdim=True)
                    only_max_score = torch.zeros_like(prediction).to(self.device)
                    only_max_score[max_v == prediction] = prediction[max_v == prediction]
                    relevance_tensor = only_max_score.view(org_shape)
                    prediction = prediction.view(org_shape)

                else:
                    org_shape = prediction.size()
                    prediction = prediction.view(org_shape[0], -1)
                    only_max_score = torch.zeros_like(prediction).to(self.device)
                    only_max_score[:, rel_for_class] += prediction[:, rel_for_class]
                    relevance_tensor = only_max_score.view(org_shape)
                    prediction = prediction.view(org_shape)

                relevance = relevance_tensor.detach()
                del relevance_tensor
            # reset list to save relevance distributions per layer
            self.inverter.reset_relevances()
            r = self.inverter.get_relevance(module_name='model', relevance_in=relevance)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return prediction, r

    def forward(self, in_tensor):
        return self.model.forward(in_tensor)

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return self.model.extra_repr()
