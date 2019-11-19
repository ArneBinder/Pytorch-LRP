import torch
import numpy as np
import torch.nn.functional as F
import allennlp
from allennlp import models

from pytorch_lrp.explainable_modules.cnn_encoder import ExplainableCnnEncoder


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))


# not used
def module_tracker(fwd_hook_func):
    """
    Wrapper for tracking the layers throughout the forward pass.

    Args:
        fwd_hook_func: Forward hook function to be wrapped.

    Returns:
        Wrapped method.

    """
    def hook_wrapper(relevance_propagator_instance, layer, *args):
        #relevance_propagator_instance.module_list.append(layer)
        #relevance_propagator_instance.module_dict[layer.name_global] = layer
        return fwd_hook_func(relevance_propagator_instance, layer, *args)

    return hook_wrapper


### forward hooks

def max_pool_nd_fwd_hook(m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):
    # Ignore unused for pylint
    #_ = self

    # Save the return indices value to make sure
    tmp_return_indices = bool(m.return_indices)
    m.return_indices = True
    _, indices = m.forward(in_tensor[0])
    m.return_indices = tmp_return_indices
    setattr(m, "indices", indices)
    setattr(m, 'out_shape', out_tensor.size())
    setattr(m, 'in_shape', in_tensor[0].size())


def linear_fwd_hook(m, in_tensor: torch.Tensor,
                    out_tensor: torch.Tensor):

    setattr(m, "in_tensor", in_tensor[0])
    setattr(m, "out_shape", list(out_tensor.size()))
    return


def conv_nd_fwd_hook(m, in_tensor: torch.Tensor,
                     out_tensor: torch.Tensor):

    setattr(m, "in_tensor", in_tensor[0])
    setattr(m, 'out_shape', list(out_tensor.size()))
    return


def silent_pass(m, in_tensor: torch.Tensor,
                out_tensor: torch.Tensor):
    # Placeholder forward hook for layers that do not need
    # to store any specific data. Still useful for module tracking.
    pass



### inverted module methods

def get_conv_method(conv_module):
    """
    Get dimension-specific convolution.
    The forward pass and inversion are made in a
    'dimensionality-agnostic' manner and are the same for
    all nd instances of the layer, except for the functional
    that needs to be used.

    Args:
        conv_module: instance of convolutional layer.

    Returns:
        The correct functional used in the convolutional layer.

    """

    conv_func_mapper = {
        torch.nn.Conv1d: F.conv1d,
        torch.nn.Conv2d: F.conv2d,
        torch.nn.Conv3d: F.conv3d
    }
    return conv_func_mapper[type(conv_module)]


def get_inv_conv_method(conv_module):
    """
    Get dimension-specific convolution inversion layer.
    The forward pass and inversion are made in a
    'dimensionality-agnostic' manner and are the same for
    all nd instances of the layer, except for the functional
    that needs to be used.

    Args:
        conv_module: instance of convolutional layer.

    Returns:
        The correct functional used for inverting the convolutional layer.

    """

    conv_func_mapper = {
        torch.nn.Conv1d: F.conv_transpose1d,
        torch.nn.Conv2d: F.conv_transpose2d,
        torch.nn.Conv3d: F.conv_transpose3d
    }
    return conv_func_mapper[type(conv_module)]


def get_inv_max_pool_method(max_pool_instance):
    """
    Get dimension-specific max_pooling layer.
    The forward pass and inversion are made in a
    'dimensionality-agnostic' manner and are the same for
    all nd instances of the layer, except for the functional
    that needs to be used.

    Args:
        max_pool_instance: instance of max_pool layer.

    Returns:
        The correct functional used in the max_pooling layer.

    """

    #conv_func_mapper = {
    #    torch.nn.MaxPool1d: F.max_unpool1d,
    #    MaxPool1dAll: F.max_unpool1d,
    #    torch.nn.MaxPool2d: F.max_unpool2d,
    #    torch.nn.MaxPool3d: F.max_unpool3d
    #}
    #return conv_func_mapper[type(max_pool_instance)]
    if isinstance(max_pool_instance, torch.nn.MaxPool1d):
        return F.max_unpool1d
    if isinstance(max_pool_instance, torch.nn.MaxPool2d):
        return torch.nn.MaxPool2d
    if isinstance(max_pool_instance, torch.nn.MaxPool3d):
        return F.max_unpool3d
    raise NotImplementedError("The network contains MaxPool layers that"
                              " are currently not supported {0:s}".format(str(max_pool_instance)))


def pass_inverted(propagator, m, relevance_in):
    return relevance_in


def time_distributed_inverse(propagator, m, relevance_in):
    shape_in = list(relevance_in.shape)
    relevance_in_reshaped = relevance_in.reshape([shape_in[0] * shape_in[1]] + shape_in[2:])
    r = m._module.lrp_backward(relevance_in_reshaped)
    shape_out = [shape_in[0], shape_in[1]] + list(r.shape)[1:]
    return r.reshape(shape_out)


def crf_tagger_inverse(propagator, m, relevance_in):
    r = m.tag_projection_layer.lrp_backward(relevance_in['logits'])
    if m._feedforward is not None:
        r = m._feedforward.lrp_backward(r)

    r = m.encoder.lrp_backward(r)

    return r


def cnn_encoder_inverse(propagator, m, relevance_in):
    if m.projection_layer:
        relevance_in = m.projection_layer.lrp_backward(relevance_in)

    relevance_split = relevance_in.split(m._num_filters, dim=1)
    rels = []
    for i, _rel in enumerate(relevance_split):
        rel_unpooled = getattr(m, f'maxpool_layer_{i}').lrp_backward(_rel)
        rel_unactivated = m._activation.lrp_backward(rel_unpooled)
        rel_unconv = getattr(m, f'conv_layer_{i}').lrp_backward(rel_unactivated)
        rels.append(rel_unconv)

    rel = sum(rels)
    rel = torch.transpose(rel, 1, 2)
    return rel


def basic_classifier_inverse(propagator, m, relevance_in):
    r = m._classification_layer.lrp_backward(relevance_in['logits'])
    if hasattr(m, '_feedforward'):
        r = m._feedforward.lrp_backward(r)
    r = m._seq2vec_encoder.lrp_backward(r)

    if m._seq2seq_encoder:
        r = m._seq2seq_encoder.lrp_backward(r)
    return r


def sequential_inverse(propagator, m, relevance_in):
    r = relevance_in

    for l in m[::-1]:
        r = l.lrp_backward(r)
    return r


def feedforward_inverse(propagator, m, relevance_in):
    r = relevance_in
    for layer, activation, dropout in list(zip(m._linear_layers, m._activations, m._dropout))[::-1]:
        # forward: output = dropout(activation(layer(output)))
        r = dropout.lrp_backward(r)
        r = activation.lrp_backward(r)
        r = layer.lrp_backward(r)
    return r


def softmax_inverse(propagator, m, relevance_in):
    # Only layer that does not conserve relevance. Mainly used
    # to make probability out of the log values. Should probably
    # be changed to pure passing and the user should make sure
    # the layer outputs are sensible (0 would be 100% class probability,
    # but no relevance could be passed on).
    if relevance_in.sum() < 0:
        relevance_in[relevance_in == 0] = -1e6
        relevance_in = relevance_in.exp()
        if not propagator.warned_log_softmax:
            print("WARNING: LogSoftmax layer was "
                  "turned into probabilities.")
            propagator.warned_log_softmax = True
    return relevance_in


def linear_inverse(propagator, m, relevance_in):

    if propagator.method == "e-rule":
        m.in_tensor = m.in_tensor.pow(propagator.p)
        w = m.weight.pow(propagator.p)
        norm = F.linear(m.in_tensor, w, bias=None)

        norm = norm + torch.sign(norm) * propagator.eps
        relevance_in[norm == 0] = 0
        norm[norm == 0] = 1
        relevance_out = F.linear(relevance_in / norm,
                                 w.t(), bias=None)
        relevance_out *= m.in_tensor
        del m.in_tensor, norm, w, relevance_in

        return relevance_out

    if propagator.method == "b-rule":
        out_c, in_c = m.weight.size()
        w = m.weight.repeat((4, 1))
        # First and third channel repetition only contain the positive weights
        w[:out_c][w[:out_c] < 0] = 0
        w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0
        # Second and fourth channel repetition with only the negative weights
        w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
        w[-out_c:][w[-out_c:] > 0] = 0

        # Repeat across channel dimension (pytorch always has channels first)
        m.in_tensor = m.in_tensor.repeat((1, 4))
        m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
        m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
        m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0

        # Normalize such that the sum of the individual importance values
        # of the input neurons divided by the norm
        # yields 1 for an output neuron j if divided by norm (v_ij in paper).
        # Norm layer just sums the importance values of the inputs
        # contributing to output j for each j. This will then serve as the normalization
        # such that the contributions of the neurons sum to 1 in order to
        # properly split up the relevance of j amongst its roots.

        norm_shape = m.out_shape
        norm_shape[1] *= 4
        norm = torch.zeros(norm_shape).to(propagator.device)

        for i in range(4):
            norm[:, out_c * i:(i + 1) * out_c] = F.linear(
                m.in_tensor[:, in_c * i:(i + 1) * in_c], w[out_c * i:(i + 1) * out_c], bias=None)

        # Double number of output channels for positive and negative norm per
        # channel.
        norm_shape[1] = norm_shape[1] // 2
        new_norm = torch.zeros(norm_shape).to(propagator.device)
        new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
        new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
        norm = new_norm

        # Some 'rare' neurons only receive either
        # only positive or only negative inputs.
        # Conservation of relevance does not hold, if we also
        # rescale those neurons by (1+beta) or -beta.
        # Therefore, catch those first and scale norm by
        # the according value, such that it cancels in the fraction.

        # First, however, avoid NaNs.
        mask = norm == 0
        # Set the norm to anything non-zero, e.g. 1.
        # The actual inputs are zero at this point anyways, that
        # is why norm is zero in the first place.
        norm[mask] = 1
        # The norm in the b-rule has shape (N, 2*out_c, *spatial_dims).
        # The first out_c block corresponds to the positive norms,
        # the second out_c block corresponds to the negative norms.
        # We find the rare neurons by choosing those nodes per channel
        # in which either the positive norm ([:, :out_c]) is zero, or
        # the negative norm ([:, :out_c]) is zero.
        rare_neurons = (mask[:, :out_c] + mask[:, out_c:])

        # Also, catch new possibilities for norm == zero to avoid NaN..
        # The actual value of norm again does not really matter, since
        # the pre-factor will be zero in this case.

        norm[:, :out_c][rare_neurons] *= 1 if propagator.beta == -1 else 1 + propagator.beta
        norm[:, out_c:][rare_neurons] *= 1 if propagator.beta == 0 else -propagator.beta
        # Add stabilizer term to norm to avoid numerical instabilities.
        norm += propagator.eps * torch.sign(norm)
        input_relevance = relevance_in.squeeze(dim=-1).repeat(1, 4)
        input_relevance[:, :2*out_c] *= (1 + propagator.beta) / norm[:, :out_c].repeat(1, 2)
        input_relevance[:, 2*out_c:] *= -propagator.beta / norm[:, out_c:].repeat(1, 2)
        inv_w = w.t()
        relevance_out = torch.zeros_like(m.in_tensor)
        for i in range(4):
            relevance_out[:, i*in_c:(i+1)*in_c] = F.linear(
                input_relevance[:, i*out_c:(i+1)*out_c],
                weight=inv_w[:, i*out_c:(i+1)*out_c], bias=None)

        relevance_out *= m.in_tensor

        sum_weights = torch.zeros([in_c, in_c * 4, 1]).to(propagator.device)
        for i in range(in_c):
            sum_weights[i, i::in_c] = 1
        relevance_out = F.conv1d(relevance_out[:, :, None], weight=sum_weights, bias=None)

        del sum_weights, input_relevance, norm, rare_neurons, \
            mask, new_norm, m.in_tensor, w, inv_w

        return relevance_out


def max_pool_nd_inverse(propagator, layer_instance, relevance_in):

    # In case the output had been reshaped for a linear layer,
    # make sure the relevance is put into the same shape as before.
    relevance_in = relevance_in.view(layer_instance.out_shape)

    invert_pool = get_inv_max_pool_method(layer_instance)
    inverted = invert_pool(relevance_in, layer_instance.indices,
                           layer_instance.kernel_size, layer_instance.stride,
                           layer_instance.padding, output_size=layer_instance.in_shape)
    del layer_instance.indices
    return inverted


def conv_nd_inverse(propagator, m, relevance_in):

    # In case the output had been reshaped for a linear layer,
    # make sure the relevance is put into the same shape as before.
    #relevance_in = relevance_in.view(m.out_shape)
    if list(m.out_shape) != list(relevance_in.shape):
        print(f'WARNING: relevance_in.shape ({relevance_in.shape}) is not the same as m.out_shape ({m.out_shape})')

    # Get required values from layer
    inv_conv_nd = get_inv_conv_method(m)
    conv_nd = get_conv_method(m)

    if propagator.method == "e-rule":
        with torch.no_grad():
            m.in_tensor = m.in_tensor.pow(propagator.p).detach()
            w = m.weight.pow(propagator.p).detach()
            norm = conv_nd(m.in_tensor, weight=w, bias=None,
                           stride=m.stride, padding=m.padding,
                           groups=m.groups)

            norm = norm + torch.sign(norm) * propagator.eps
            relevance_in[norm == 0] = 0
            norm[norm == 0] = 1
            relevance_out = inv_conv_nd(relevance_in/norm,
                                        weight=w, bias=None,
                                        padding=m.padding, stride=m.stride,
                                        groups=m.groups)
            relevance_out *= m.in_tensor
            del m.in_tensor, norm, w
            return relevance_out

    if propagator.method == "b-rule":
        with torch.no_grad():
            w = m.weight

            out_c, in_c = m.out_channels, m.in_channels
            repeats = np.array(np.ones_like(w.size()).flatten(), dtype=int)
            repeats[0] *= 4
            w = w.repeat(tuple(repeats))
            # First and third channel repetition only contain the positive weights
            w[:out_c][w[:out_c] < 0] = 0
            w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0
            # Second and fourth channel repetition with only the negative weights
            w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
            w[-out_c:][w[-out_c:] > 0] = 0
            repeats = np.array(np.ones_like(m.in_tensor.size()).flatten(), dtype=int)
            repeats[1] *= 4
            # Repeat across channel dimension (pytorch always has channels first)
            m.in_tensor = m.in_tensor.repeat(tuple(repeats))
            m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
            m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
            m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0
            groups = 4

            # Normalize such that the sum of the individual importance values
            # of the input neurons divided by the norm
            # yields 1 for an output neuron j if divided by norm (v_ij in paper).
            # Norm layer just sums the importance values of the inputs
            # contributing to output j for each j. This will then serve as the normalization
            # such that the contributions of the neurons sum to 1 in order to
            # properly split up the relevance of j amongst its roots.
            norm = conv_nd(m.in_tensor, weight=w, bias=None, stride=m.stride,
                           padding=m.padding, dilation=m.dilation, groups=groups * m.groups)
            # Double number of output channels for positive and negative norm per
            # channel. Using list with out_tensor.size() allows for ND generalization
            new_shape = m.out_shape
            new_shape[1] *= 2
            new_norm = torch.zeros(new_shape).to(propagator.device)
            new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
            new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
            norm = new_norm
            # Some 'rare' neurons only receive either
            # only positive or only negative inputs.
            # Conservation of relevance does not hold, if we also
            # rescale those neurons by (1+beta) or -beta.
            # Therefore, catch those first and scale norm by
            # the according value, such that it cancels in the fraction.

            # First, however, avoid NaNs.
            mask = norm == 0
            # Set the norm to anything non-zero, e.g. 1.
            # The actual inputs are zero at this point anyways, that
            # is why norm is zero in the first place.
            norm[mask] = 1
            # The norm in the b-rule has shape (N, 2*out_c, *spatial_dims).
            # The first out_c block corresponds to the positive norms,
            # the second out_c block corresponds to the negative norms.
            # We find the rare neurons by choosing those nodes per channel
            # in which either the positive norm ([:, :out_c]) is zero, or
            # the negative norm ([:, :out_c]) is zero.
            rare_neurons = (mask[:, :out_c] + mask[:, out_c:])

            # Also, catch new possibilities for norm == zero to avoid NaN..
            # The actual value of norm again does not really matter, since
            # the pre-factor will be zero in this case.

            norm[:, :out_c][rare_neurons] *= 1 if propagator.beta == -1 else 1 + propagator.beta
            norm[:, out_c:][rare_neurons] *= 1 if propagator.beta == 0 else -propagator.beta
            # Add stabilizer term to norm to avoid numerical instabilities.
            norm += propagator.eps * torch.sign(norm)
            spatial_dims = [1] * len(relevance_in.size()[2:])

            input_relevance = relevance_in.repeat(1, 4, *spatial_dims)
            input_relevance[:, :2*out_c] *= (1 + propagator.beta) / norm[:, :out_c].repeat(1, 2, *spatial_dims)
            input_relevance[:, 2*out_c:] *= -propagator.beta / norm[:, out_c:].repeat(1, 2, *spatial_dims)
            # Each of the positive / negative entries needs its own
            # convolution. TODO: Can this be done in groups, too?

            relevance_out = torch.zeros_like(m.in_tensor)
            # Weird code to make up for loss of size due to stride
            tmp_result = result = None
            for i in range(4):
                tmp_result = inv_conv_nd(
                    input_relevance[:, i*out_c:(i+1)*out_c],
                    weight=w[i*out_c:(i+1)*out_c],
                    bias=None, padding=m.padding, stride=m.stride,
                    groups=m.groups)
                result = torch.zeros_like(relevance_out[:, i*in_c:(i+1)*in_c])
                tmp_size = tmp_result.size()
                slice_list = [slice(0, l) for l in tmp_size]
                result[slice_list] += tmp_result
                relevance_out[:, i*in_c:(i+1)*in_c] = result
            relevance_out *= m.in_tensor

            sum_weights = torch.zeros([in_c, in_c * 4, *spatial_dims]).to(propagator.device)
            for i in range(m.in_channels):
                sum_weights[i, i::in_c] = 1
            relevance_out = conv_nd(relevance_out, weight=sum_weights, bias=None)

            del sum_weights, m.in_tensor, result, mask, rare_neurons, norm, \
                new_norm, input_relevance, tmp_result, w

            return relevance_out


class RelevancePropagator:
    """
    Class for computing the relevance propagation and supplying
    the necessary forward hooks for all layers.
    """

    # Implemented rules for relevance propagation.
    available_methods = ["e-rule", "b-rule"]

    def __init__(self, lrp_exponent, beta, method, epsilon, device, additional_allowed_pass_layers=(), additional_no_forward_hook_layers=()):
        # All layers that do not require any specific forward hooks.
        # This is due to the fact that they are all one-to-one
        # mappings and hence no normalization is needed (each node only
        # influences exactly one other node -> relevance conservation
        # ensures that relevance is just inherited in a one-to-one manner, too).
        self.allowed_pass_layers = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                    torch.nn.BatchNorm3d,
                                    torch.nn.ReLU, torch.nn.ELU, Flatten,
                                    torch.nn.Dropout, torch.nn.Dropout2d,
                                    torch.nn.Dropout3d,
                                    torch.nn.Softmax,
                                    torch.nn.LogSoftmax,
                                    torch.nn.Sigmoid,
                                    allennlp.modules.seq2seq_encoders.PassThroughEncoder) \
                                   + additional_allowed_pass_layers
        self.no_forward_hook_layers = self.allowed_pass_layers + \
                                      (torch.nn.ModuleList,
                                       allennlp.modules.TimeDistributed,
                                       allennlp.modules.FeedForward,
                                       models.crf_tagger.CrfTagger,
                                       ExplainableCnnEncoder,
                                       models.BasicClassifier,
                                       torch.nn.Sequential) + \
                                      additional_no_forward_hook_layers
        self.device = device
        self.layer = None
        self.p = lrp_exponent
        self.beta = beta
        self.eps = epsilon
        self.warned_log_softmax = False
        if method not in self.available_methods:
            raise NotImplementedError("Only methods available are: " +
                                      str(self.available_methods))
        self.method = method

    def get_backward_function(self, layer):
        if isinstance(layer,
                      (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
            return max_pool_nd_inverse

        elif isinstance(layer,
                      (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            return conv_nd_inverse

        elif isinstance(layer, torch.nn.LogSoftmax):
            return softmax_inverse
        elif isinstance(layer, self.allowed_pass_layers):
            # The above layers are one-to-one mappings of input to
            # output nodes. All the relevance in the output will come
            # entirely from the input node. Given the conservation
            # of relevance, the input is as relevant as the output.
            return pass_inverted

        elif isinstance(layer, torch.nn.Linear):
            return linear_inverse
        elif isinstance(layer, allennlp.modules.FeedForward):
            return feedforward_inverse
        elif isinstance(layer, allennlp.modules.TimeDistributed):
            return time_distributed_inverse
        elif isinstance(layer, models.crf_tagger.CrfTagger):
            return crf_tagger_inverse
        elif isinstance(layer, ExplainableCnnEncoder):
            return cnn_encoder_inverse
        elif isinstance(layer, models.BasicClassifier):
            return basic_classifier_inverse
        elif isinstance(layer, (torch.nn.Sequential, torch.nn.ModuleList)):
            return sequential_inverse
        else:
            raise NotImplementedError("The network contains layers that"        
                                      " are currently not supported {0:s}".format(str(layer)))

    def get_layer_fwd_hook(self, layer):
        """
        Each layer might need to save very specific data during the forward
        pass in order to allow for relevance propagation in the backward
        pass. For example, for max_pooling, we need to store the
        indices of the max values. In convolutional layers, we need to calculate
        the normalizations, to ensure the overall amount of relevance is conserved.

        Args:
            layer: Layer instance for which forward hook is needed.

        Returns:
            Layer-specific forward hook.

        """

        if isinstance(layer,
                      (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
            return max_pool_nd_fwd_hook

        if isinstance(layer,
                      (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            return conv_nd_fwd_hook

        if isinstance(layer, self.no_forward_hook_layers):
            return silent_pass  # No hook needed.

        if isinstance(layer, torch.nn.Linear):
            return linear_fwd_hook

        raise NotImplementedError("The network contains layers that"
                                  " are currently not supported {0:s}".format(str(layer)))






