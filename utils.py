import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os


class VectorizeChar:
    def __init__(self, max_len=100):
        self.vocab = (
            ["-", "#", "<", ">"]
            + list(string.printable[:95].replace('-','').replace('#','').replace('<','').replace('>',''))
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] #+ [0] * pad_len

    def get_vocabulary(self):
        return self.vocab

class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
        self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28
    ):
        super().__init__()
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n=== Predictions after epoch {epoch + 1} ===")

        source = self.batch["source"]
        target = self.batch["target"].numpy()
        bs = min(5, source.shape[0])  # Виводимо лише перші 5 прикладів
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()

        for i in range(bs):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text.replace('-', '')}")
            print(f"prediction: {prediction}\n")



class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        epoch = tf.cast(epoch, tf.float32)

        warmup_lr = (
                self.init_lr
                + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / self.decay_epochs,
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = tf.cast(step // tf.cast(self.steps_per_epoch, tf.int64), tf.int32)
        return self.calculate_lr(epoch)



def get_data(df: pd.DataFrame)->list:
    data = []
    for  idx,row in df.iterrows():
      data.append({"audio": row['filename'], "text": row['transcript'].strip()})
    return data

def path_to_features(path):
    def _load(path_):
        path_str = path_.numpy().decode("utf-8")

        if not os.path.exists(path_str):
            return tf.zeros((1, 20), dtype=tf.float32)

        x = tf.io.read_file(path_str)
        x = tf.io.decode_raw(x, out_type=tf.float32)
        x = tf.reshape(x, [-1, 20])

        mean = tf.reduce_mean(x, axis=0, keepdims=True)
        std = tf.math.reduce_std(x, axis=0, keepdims=True)
        x = (x - mean) / (std + 1e-6)

        return tf.cast(x, tf.float32)  # ⬅️ Оце виправляє твою помилку

    return tf.py_function(_load, [path], tf.float32)


def txt_to_labels(txt:str, vectorizer: VectorizeChar):
  return tf.convert_to_tensor(vectorizer(txt),dtype=tf.int64)






def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def wer(ground_truth, prediction):    
    word_distance = levenshtein(ground_truth.split(), prediction.split())
    word_length = len(ground_truth.split())
    wer = word_distance/word_length
    wer = min(wer, 1.0)
    return wer
    
def cer(ground_truth, prediction):
    char_distance = levenshtein(ground_truth, prediction)
    char_length = len(ground_truth)
    cer=char_distance/char_length
    cer = min(cer, 1.0)
    return  cer
