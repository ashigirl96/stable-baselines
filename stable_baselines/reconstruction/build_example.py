from stable_baselines.common.successor_features import encode_cnn, decode_cnn
import tensorflow as tf


def main():
  graph = tf.Graph()
  with graph.as_default():
    x = tf.placeholder(tf.float32, [None, 84, 84, 4])
    mu, sigma_sq = encode_cnn(x)
    print(mu)

    recons_image = decode_cnn(mu)
    print(recons_image)

if __name__ == '__main__':
  main()
