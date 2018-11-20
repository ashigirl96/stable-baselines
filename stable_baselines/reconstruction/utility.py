import tensorflow as tf
import re

SUMMARY_TYPE = ['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']


def summary(tensor_collection,
            observation_shape,
            summary_type=SUMMARY_TYPE,
            scope=None,
            ignore=[],
            ) -> tf.summary.Summary:
  """Summary.
  Usage:
      1. summary(tensor)
      2. summary([tensor_a, tensor_b])
      3. summary({tensor_a: 'a', tensor_b: 'b})
  """

  def _summary(tensor: tf.Tensor, name, summary_type):
    """Attach a lot of summaries to a Tensor."""
    if name is None:
      # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
      # session. This helps the clarity of presentation on tensorboard.
      name = re.sub('%s_[0-9]*/' % 'tower', '', tensor.name)
      name = re.sub(':', '-', name)

    summaries = []
    if len(tensor.shape) == 0:
      summaries.append(tf.summary.scalar(name, tensor))
    if 0 < len(tensor.shape) < 4 or name in ignore:
      if 'mean' in summary_type:
        mean = tf.reduce_mean(tensor)
        summaries.append(tf.summary.scalar(name + '/mean', mean))
      if 'stddev' in summary_type:
        mean = tf.reduce_mean(tensor)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
        summaries.append(tf.summary.scalar(name + '/stddev', stddev))
      if 'max' in summary_type:
        summaries.append(tf.summary.scalar(name + '/max', tf.reduce_max(tensor)))
      if 'min' in summary_type:
        summaries.append(tf.summary.scalar(name + '/min', tf.reduce_min(tensor)))
      if 'sparsity' in summary_type:
        summaries.append(tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor)))
      if 'histogram' in summary_type:
        summaries.append(tf.summary.histogram(name, tensor))
    if len(tensor.shape) == 4:
      print('image {}, {}'.format(name, tensor))
      summaries.append(tf.summary.image(name + '/image', tensor))
    if name == 'recons_losses':
      print('recons_losses', summaries)
    return tf.summary.merge(summaries)

  if not isinstance(tensor_collection, (list, tuple, dict)):
    tensor_collection = [tensor_collection]

  with tf.name_scope(scope, 'summary'):
    summaries = []
    if isinstance(tensor_collection, (list, tuple)):
      for tensor in tensor_collection:
        summaries.append(_summary(tensor, None, summary_type))
    else:
      for tensor, name in tensor_collection.items():
        summaries.append(_summary(tensor, name, summary_type))

  return tf.summary.merge(summaries)
