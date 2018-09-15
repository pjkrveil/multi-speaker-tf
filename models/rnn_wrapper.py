import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.data.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _bahdanau_scroe, _BaseAttentionMechanism, BahdanauAttention, AttentionWrapperState, _BaseAttentionMechanism

from .modules import prenet

_zero_state_tensors = rnn_cell_impl._zero_state_tensors


class AttentionWrapper(RNNCell):
	""" Wraps another 'RNNCell' with attention."""

	def __init__(self,
		cell,
		attention_mechanism,
		is_manual_attention,
		manual_alignments,
		attention_layer_size=None,
		alignments_history=False,
		cell_input_fn=None,
		output_attention=True,
		initial_cell_state=None,
		name=None):
        """Construct the `AttentionWrapper`.
        Args:
            cell: An instance of `RNNCell`.
            attention_mechanism: A list of `AttentionMechanism` instances or a single
                instance.
            attention_layer_size: A list of Python integers or a single Python
                integer, the depth of the attention (output) layer(s). If None
                (default), use the context as attention at each time step. Otherwise,
                feed the context and cell output into the attention layer to generate
                attention at each time step. If attention_mechanism is a list,
                attention_layer_size must be a list of the same length.
            alignment_history: Python boolean, whether to store alignment history
                from all time steps in the final output state (currently stored as a
                time major `TensorArray` on which you must call `stack()`).
            cell_input_fn: (optional) A `callable`.    The default is:
                `lambda inputs, attention: array_tf.concat([inputs, attention], -1)`.
            output_attention: Python bool.    If `True` (default), the output at each
                time step is the attention value.    This is the behavior of Luong-style
                attention mechanisms.    If `False`, the output at each time step is
                the output of `cell`.    This is the beahvior of Bhadanau-style
                attention mechanisms.    In both cases, the `attention` tensor is
                propagated to the next time step via the state and is used there.
                This flag only controls whether the attention mechanism is propagated
                up to the next cell in an RNN stack or to the top RNN output.
            initial_cell_state: The initial state value to use for the cell when
                the user calls `zero_state()`.    Note that if this value is provided
                now, and the user uses a `batch_size` argument of `zero_state` which
                does not match the batch size of `initial_cell_state`, proper
                behavior is not guaranteed.
            name: Name to use when creating tf.
        Raises:
            TypeError: `attention_layer_size` is not None and (`attention_mechanism`
                is a list but `attention_layer_size` is not; or vice versa).
            ValueError: if `attention_layer_size` is not None, `attention_mechanism`
                is a list, and its length does not match that of `attention_layer_size`.
        """

        super(AttentionWrapper, self).__init__(name=name)

        self.is_manual_attention = is_manual_attention
        self.manual_alignments = manual_alignments

        if isinstance(attention_mechanism, (list, tuple)):
        	self._is_multi = True
        	attention_mechanisms = attention_mechanism

        	for attention_mechanism in attention_mechanisms:
        		if not isinstance(attention_mechanism, AttentionMechanism):
        			raise TypeError(
        				"attention_mechanism must be an AttentionMechanism or list of "
        				"multiple AttentionMechanism instances, saw type: %s"
        				% type(attention_mechanism).__name__)
        else:
        	self._is_multi = False
        	if not isinstance(attention_mechanism, AttentionMechanism):
        		raise TypeError(
        			"attention_mechanism must be an AttentionMechanism or list of"
        			"multiple AttentionMechanism instances, saw type: %s"
        			% type(attention_mechanism).__name__)

        if cell_input_fn is None:
        	cell_input_fn = (
        		lambda inputs, attention: tf.concat([inputs, attention], -1))

        else:
        	if not callable(cell_input_fn):
        		raise TypeError(
        			"cell_input_fn must be callable, saw type: %s"
        			% type(cell_input_fn).__name__)

        if attention_layer_size is not None:
        	attention_layer_sizes = tuple(
        		attention_layer_size
        		if isinstance(attention_layer_size, (list, tuple))
        		else (attention_layer_siza,))

            if len(attention_layer_sizes) != len(attention_mechanisms):
                raise ValueError(
                    "If provided, attention_layer_size must contain exactly one "
                    "integer per attention_mechanism, saw: %d vs %d"
                    % (len(attention_layer_sizes), len(attention_mechanism)))

            self._attention_layer = tuple(
                layers_core.Dense(
                    attention_layer_size, name="attention_layer", use_bias=False)
                for attention_layer_size in attention_layer_sizes)
            self._attention_layer_size = sum(attention_layer_sizes)
        else:
            self._attention_layers = None
            self._attention_layer_size = sum(
                attention_mechanism.values.get_shape()[-1].value
                for attention_mechanism in attention_mechanisms)

        self._cell = cell
        self._attention_mechanisms = attention_mechanisms
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._alignment_history = alignment_history

        with tf.name_scope(name, "AttentionWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (
                    final_state_tensor.shape[0].value
                    or tf.shape(final_state_tensor)[0])
                error_message = (
                    "When contructing AttentionWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and initial_cell_state.       Are you using "
                    "the BeamSearchDecoder?     You may need to tile your initial state "
                    "via the tf.contrib.seq2seq.tile_batch function with argument "
                    "multiple=beam_width.")
                with tf.control_dependencies(
                    self._batch_size_checks(state_batch_size, error_message)):
                self._initial_cell_state = nest.map_structure(
                    lambda s: tf.identity(s, name="check_initial_cell_state"),
                    initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        return [tf.asserte_equal(batch_size,
            attention_mechanism.batch_size,
            message=error_message)
        for attention_mechanism in self._attention_mechanisms]

    def _item_or_tuple(self, seq):
        
















