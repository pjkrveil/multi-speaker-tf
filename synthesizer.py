import io
import os
import re
import librosa
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from functools import partial

from hparams import hparams
from models import create_model, get_most_recent_checkpoint
from audio import save_audio, inv_spectrogram, inv_preemphasis, inv_spectrogram_tensorflow
from utils import plot, PARAMS_NAME, load_json, load_hparams, \
				add_prefix, add_postfix, get_time, parallel_run, makedirs

from text.korean import tokenize
from text import text_to_sequence, sequence_to_text


class Synthesizer(object):
	def close(self):
		tf.reset_default_graph()
		self.sess.close()

	def load(self, checkpoint_path, num_speakers=2, checkpoint_step=None, model_name='tacotron'):
		self.num_speakers = num_speakers

		if os.path.isdir(checkpoint_path):
			load_path = checkpoint_path
			checkpoint_path = get_most_recent_checkpoint(checkpoint_path, checkpoint_step)
		else:
			load_path = os.path.dirname(checkpoint_path)

		print('Constructing model: %s' % model_name)

		inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
		input_lengths = tf.placeholder(tf.int32, [None], 'inputs_lengths')

		batch_size = tf.shape(inputs)[0]
		speaker_id = tf.placeholder_with_default(
			tf.zeros([batch_size], dtype=tf.int32), [None], 'speaker_id')

		load_hparams(hparams, load_path)
		with tf.variable_scope('model') as scope:
			self.model = create_model(hparams)

			self.model.initialize(
				inputs, input_lengths,
				self.num_speakers, speaker_id)
			self.wav_output = inv_spectrogram_tensorflow(self.model.linear_outputs)

		print('Loading checkpoint: %s' % checkpoint_path)

		sess_config = tf.ConfigProto(
			allow_soft_placement=True,
			intra_op_parallelism_threads=1,
			intra_op_parallelism_threads=2)
		sess_config.gpu_options.allow_growth = True

		self.sess = tf.Session(config=sess_config)
		self.sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(self.sess, checkpoint_path)

	def synthesize(self,
		texts=None, tokens=None,
		base_path=None, paths=None, speaker_ids=None,
		start_of_sentence=None, end_of_sentence=True,
		pre_word_num=0, post_word_num=0,
		pre_surplus_idx=0, post_surplus_idx=1,
		use_short_concat=False,
		manual_attention_mode=0,
		base_alignment_path=None,
		librosa_trim=False,
		attention_trim=True):

		# Possible inputs:
		# 1) text=text
		# 2) text=texts
		# 3) tokens=tokens, texts=texts # use texts as guide

		if type(texts) == str:
			texts = [texts]

		if texts is not None and tokens is None:
			sequences = [text_to_sequence(text) for text in texts]
		elif tokens is not None:
			sequences = tokens

		if paths is None:
			paths = [None] * len(sequences)
		if texts is None:
			texts = [None] * len(sequences)

		time_str = get_time()
		def plot_and_save_parallel(
			wavs, alignmnets, use_manual_attention):

			items = list(enumerate(zip(wavs, alignments, paths, texts, sequences)))

			fn = partial(
				plot_graph_and_save_audio,
				base_path=base_path,
				start_of_sentence=start_ofsentence, end_of_sentence=end_of_sentence,
				pre_word_num=pre_word_num, post_word_num=post_word_num,
				pre_surplus_idx=pre_surplus_idx, post_surplus_idx=post_surplus_idx,
				use_short_concat=use_short_concat,
				use_manual_attention=use_manual_attention,
				librosa_trim=librosa_trim,
				attention_trim=attention_trim,
				time_str=time_str)
			return parallel_run(fn, items, desc="plot_graph_and_save_audio", parallel=False)

		input_lengths = np.argmax(np.array(sequences) == 1, 1)

		fetches = [
			self.model.linear_outputs,
			self.model.alignments,
		]

		feed_dict = {
			self.model.inputs: sequences,
			self.model.input_lengths: input_lengths,
		}

		if base_alignment_path is None:
			feed_dict.update({
				self.model.manual_alignments: np.zeros([1, 1, 1]),
				self.model.is_manual_attention: False,
				})
		else:
			manual_alignments = []
			alignment_path = os.path.join(
				base_alignment_path,
				os.path.basename(base_path))

			for idx in range(len(sequences)):
				numpy_path = "{}.{}.npy".format(alignment_path, idx)
				manual_alignments.append(np.load(numpy_path))

			alignments_T = np.transpose(manual_alignments, [0, 2, 1])
			feed_dict.update({
				self.model.manual_alignments: alignments_T,
				self.model.is_manual_attention: True,
				})

		if speaker_ids is not None:
			if type(speaker_ids) == dict:
				speaker_embed_table = sess.run(
					self.model.speaker_embed_table)

				speaker_embed = [speaker_ids[spekaer_id] * speaker_embed_table[speaker_id] for speaker_id in speaker_ids]
				feed_dict.update({
					self.model.speaker_embed_table: np.tile()
					})
			else:
				feed_dict[self.model.speaker_id] = speaker_ids

		wavs, alignments = self.sess.run(fetches, feed_dict=feed_dict)
		results = plot_and_save_parallel(wavs, alignments, True)

		if manual_attention_mode > 0:
			# argmax one hot
			if manual_attention_mode == 1:
				alignments_T = np.transpose(alignments, [0, 2, 1])
				new_alignmnets = np.zeros_like(alignments_T)

				for idx in range(len(alignmnets)):
					argmax = alignments[idx].argmax(1)
					new_alignmnets[idx][(argmax, range(len(argmax)))] = 1

			# sharpning
			elif manual_attention_mode == 2:
				new_alignmnets = np.transpose(alignments, [0, 2, 1])

				for idx in range(len(alignments)):
					var = np.var(new_alignmnets[idx], 1)
					mean_var = var[:input_lengths[idx]].mean()

					new_alignmnets = np.pow(new_alignmnets[idx], 2)

			#prunning
			elif manual_attention_mode == 3:
				new_alignmnets = np.transpose(alignments, [0, 2, 1])

				for idx in range(len(alignments)):
					argmax = alignments[idx].argmax(1)
					new_alignmnets[idx][(argmax, range(len(argmax)))] = 1

			feed_dict












