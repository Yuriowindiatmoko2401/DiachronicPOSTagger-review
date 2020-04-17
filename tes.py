import numpy as np
import sys
import tensorflow as tf

a = np.array([[7, 11, 12],
           [13, 14, 15]])
np.argmax(a)

np.random.permutation(10)
# array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
    
np.random.permutation([1, 4, 9, 12, 15])
# array([15,  1,  9,  4, 12])
    
arr = np.arange(9).reshape((3, 3))
np.random.permutation(arr)
# array([[6, 7, 8],
       # [0, 1, 2],
       # [3, 4, 5]]

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)
# >> array([[1, 2],
       #    [3, 4],
       #    [5, 6]])

arrays = [np.random.randn(3, 4) for _ in range(10)]
np.stack(arrays, axis=0).shape

np.repeat(3, 4)
# >> array([3, 3, 3, 3])

x = np.array([[1,2],[3,4]])
np.repeat(x, 2)
# >> array([1, 1, 2, 2, 3, 3, 4, 4])
    
np.repeat(x, 3, axis=1)
# >> array([[1, 1, 1, 2, 2, 2],
        #   [3, 3, 3, 4, 4, 4]])

np.repeat(x, [1, 2], axis=0)
# >> array([[1, 2],
        #   [3, 4],
        #   [3, 4]])

print(arrays,flush=True)
sys.stdout.flush()

print("By Decade",flush=True)
# model.average_perplexity(test_data, embed_data, feedforward, args.loadperplex, 0)
sys.stdout.flush()

# Samples sentences and examines ten best predicted cases
print("Sample Sentences",flush=True)
sys.stdout.flush()

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

t1 = [
      [[1, 2, 3], 
       [4, 5, 6]],
      
      [[1, 2, 3], 
       [4, 5, 6]]
]

t2 = [
      [[7, 8, 9], 
      [10, 11, 12]],

      [[7, 8, 9], 
      [10, 11, 12]]
]

tf.concat([t1, t2], 2)  
# [
#  [[1, 2, 3, 7, 8, 9], 
#  [4, 5, 6, 10, 11, 12]],
#  [[1, 2, 3, 7, 8, 9], 
#  [4, 5, 6, 10, 11, 12]]
#]
# <tf.Tensor 'concat_4:0' shape=(2, 2, 6) dtype=int32>

""" notes
dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging 
dirnames:  ['DiachronicPOSTagger'] 
filenames:  ['Detecting_Syntactic_Change_Using_Neural_POS_Tagger.pdf', '.DS_Store', 
'NER_Characteristic_Medical_Herbs_Using_Modified_HMM_Approach.pdf', 'notes_for_task.md', 'TH-01.pdf']

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger 
dirnames:  ['img', '.git', 'Plots'] 
filenames:  ['approach_methods.md', '.DS_Store', 'tes.py', 'README.md', '.gitignore', 'from_paper_notes.md', 
'data_processing.py', 'lstm.py']

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/img 
dirnames:  [] 
filenames:  ['lstm_1.png', 'first_pca_year_to_year_2.png', 'first_pca_year_to_year_1.png', 
'perplexity_8.png', 'perplexity_9.png', 'Coefficient-of-Determination-1.png', 'perplexity_10.png', 
'perplexity_1.png', 'Coefficient-of-Determination.jpg', 'perplexity_7.png', 'perplexity_6.png', 'perplexity_5.png']

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git 
dirnames:  ['objects', 'info', 'logs', 'hooks', 'refs'] 
filenames:  ['ORIG_HEAD', 'config', 'HEAD', 'description', 'index', 'packed-refs', 'FETCH_HEAD']

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/objects 
dirnames:  ['pack', 'info'] 
filenames:  []

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/objects/pack 
dirnames:  [] 
filenames:  ['pack-6e8c3e5a2e579a31ab3a489bf8e8230553d36002.idx', 'pack-6e8c3e5a2e579a31ab3a489bf8e8230553d36002.pack']

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/objects/info 
dirnames:  [] filenames:  []

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/info 
dirnames:  [] filenames:  ['exclude']

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/logs 
dirnames:  ['refs'] filenames:  ['HEAD']

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/logs/refs 
dirnames:  ['heads', 'remotes'] filenames:  []

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/logs/refs/heads 
dirnames:  [] filenames:  ['master']

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/logs/refs/remotes 
dirnames:  ['origin'] filenames:  []

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/logs/refs/remotes/origin 
dirnames:  [] filenames:  ['HEAD']

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/hooks 
dirnames:  [] 
filenames:  ['commit-msg.sample', 'pre-rebase.sample', 'pre-commit.sample', 'applypatch-msg.sample', 
'fsmonitor-watchman.sample', 'pre-receive.sample', 'prepare-commit-msg.sample', 'post-update.sample', 
'pre-applypatch.sample', 'pre-push.sample', 'update.sample']

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/refs 
dirnames:  ['heads', 'tags', 'remotes'] 
filenames:  []

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/refs/heads 
dirnames:  [] 
filenames:  ['master']

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/refs/tags 
dirnames:  [] filenames:  []

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/refs/remotes 
dirnames:  ['origin'] filenames:  []

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/.git/refs/remotes/origin 
dirnames:  [] filenames:  ['HEAD']

dirpath:  /Users/yurio/Kuliah/semester_2/text_analytics/POS_tagging/DiachronicPOSTagger/Plots 
dirnames:  [] 
filenames:  ['1864_FINAL.png', '1875.png', 'pca_2D.png', 'tsne_1D.png', '1902_FINAL.png', '1915_dot.png', 
'1876.png', '1886_sent.png', '1814_sent.png', '1946_sent.png', 'loss_vs_batch.png', 'pca_1D.png', 'tsne_2D.png', 
'1822_FINAL.png', '1891_FINAL.png', '1937_dot.png', '1895.png', '1856.png', 'year_embed_cluster.png', '1950.png', 
'1854_FINAL.png', '1854_sent.png']

"""

"""
Signature:
tf.contrib.layers.xavier_initializer(
    uniform=True,
    seed=None,
    dtype=tf.float32,
)

Source:

def xavier_initializer(uniform=True, seed=None, dtype=dtypes.float32):

  Returns an initializer performing "Xavier" initialization for weights.

  This function implements the weight initialization from:

  Xavier Glorot and Yoshua Bengio (2010):
           [Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.](
           http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

  This initializer is designed to keep the scale of the gradients roughly the
  same in all layers. 
  In uniform distribution this ends up being the range:
  `x = sqrt(6. / (in + out)); [-x, x]` and 
  for normal distribution a standard
  deviation of `sqrt(2. / (in + out))` is used.

  Args:
    uniform: Whether to use uniform or normal distributed random initialization.
    seed: A Python integer. Used to create random seeds. See
          `tf.compat.v1.set_random_seed` for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer for a weight matrix.
  
  return variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                      uniform=uniform, seed=seed, dtype=dtype)
"""