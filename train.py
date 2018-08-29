import os
import time
import math
import argparse
import traceback
import subprocess
import numpy as np
from jamo import h2j
import tensorflow as tf
from datetime import datetime
from functools import partial

log = infolog.log

def create_batch_inputs_from_texts(tests):
	sequences = [text_to_sequence(text) for in texts]

	inputs = _prepare_inputs(sequences)
	input_lengths = np.asarray([len(x) for x in inputs], dtype=np.int32)

	for idx, (seq, text) in enumerate(zip(inputs, texts)):
		recovered_text = sequence_to_text(seq, skip_eos_and_pad=True)
		if recovered text != h2j(text):
			log(" [{}] {}".format(idx, text))
			log(" [{}] {}".format(idx, recovered_text))
			log("="*30)

	return inputs, input_lengths


# getting Github commit
def get_git_commit():
	subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])		# Verify client is clean
	commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
	log('Git commit: %s' % commit)
	return commit


# adding statistics for calculating losses
def add_stats(model, model2=None, scope=name='train'):
	with tf.variable_scope(scope_name) as scope:
		summaries = [
				tf.summary.scalar('loss_mel', model.mel_loss),
				tf.summary.scalar('loss_linear', model.linear_loss),
				tf.summary.scalar('loss', model.loss_without_coeff),
		]

		if scope_name == 'train':
			gradient_norms = [tf.norm(grad) for grad in model.gradients if grad is not None]

			summaries.extend([
				tf.summary.scalar('learning_rate', model.learning_rate),
				tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)),
			])

	if model2 is not None:
		with tf.variable_scope('gap_test-train') as scope:
			summaries.extend([
				tf.summary.scalar('loss_mel',
					model.mel_loss - model2.mel_loss),
				tf.summary.scalar('loss_linear',
					model.linear_loss - model2.linear_loss),
				tf.summary.scalar('loss',
					model.loss_without_coeff - model2.loss_without_coeff),
			])

	return tf.summary.merge(summaries)


def	save_and_plot_fn(args, log_dir, step, loss, prefix):
	idx, (seq, spec, align) = args

	audio_path = os.path.join(
		log_dir, '{}-step-{:09d}-audio{:03d}.wav'.format(prefix, step, idx))
	align_path = os.path.join(
		log_dir, '{}-step-{:09d}-audio{:03d}.png'.format(prefix, step, idx))

	waveform = inv_spectrogram(spec.T)
	save_audio(waveform, audio_path)

	info_text = 'step={:d}, loss={:.5f}'.format(step, loss)
	plot.plot_alignment(
		align, align_path, info=info_text,
		text=sequence_to_text(seq,
			skip_eos_and_pad=True, combine_jamo=True))
		

def save_and_plot(sequences, spectrograms,
		alignments, log_dir, step, loss, prefix):
	
	fn = partial(save_and_plot_fn,
		log_dir=log_dir, step=step, loss=loss, prefix=prefix)
	items = list(enumerate(zip(sequences, spectrograms, alignments)))

	parallel_run(fn, items, parallel=False)
	log('Test finished for step {}.'.format(step))



def train(log_dir, config):
	config.data_paths = config.data_paths

	data_dirs = [os.path.join(data_path, "data") \
			for data_path in config.data_paths]
	num_speakers = len(data_dirs)
	config.num_test = config.num_test_per_speaker * num_speakers

	if num_speakers > 1 and hparams.model_type not in ["deepvoice", "simple"]:
		raise Exception("[!] Unknown model_type for multi-speaker: {}".format(config.model_type))

	commit = get_git_commit() if config.git else 'None'
	checkpoint_path = os.path.join(log_dir, 'model.ckpt')

































