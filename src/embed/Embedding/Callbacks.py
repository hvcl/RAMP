from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
import copy


class FixEmbed(CallbackAny2Vec):
    def __init__(self, model, walks=None):
        self.model = model
        self.duplicated_model = BaseWordEmbeddingsModel()
        self.memo = {}
        self.walks = walks

        self.epoch_cnt = 1

    def on_epoch_begin(self, model):
        if self.epoch_cnt == 1:
            self.duplicated_model = copy.deepcopy(model, self.memo)
            self.check_embeddings(model)
        self.epoch_cnt = 0

    def on_train_begin(self, model):
        print('Fix Cell-line Embeddings')
        #print('No Embeddings')

    def check_embeddings(self, model):
        vocab_list = self.model.vocab

        for wv_key in vocab_list:
            if self.walks is not None and wv_key not in self.walks:
                # Continue to avoid error from trying to get embedding vectors which don't exist
                continue
            model.wv.vectors[model.wv.vocab[wv_key].index] = \
                self.model.wv.vectors[self.model.wv.vocab[wv_key].index]


class InitializeEmbed(CallbackAny2Vec):
    def __init__(self, model):
        self.model = model

    def on_train_begin(self, model):
        vocab_list = self.model.vocab
        for wv_key in vocab_list:
            model.wv.vectors[model.wv.vocab[wv_key].index] = \
                self.model.wv.vectors[self.model.wv.vocab[wv_key].index]
        print('Initialized Embeddings with Pretrained GDSC!')


def get_callback_func(args, files, walks):
    callbacks = []

    if args.init_gdsc:
        pretrained_embedding = KeyedVectors.load_word2vec_format(files['GDSC_EMBEDDING_FILE'])
        callbacks.append(InitializeEmbed(pretrained_embedding))

    if args.fix_gdsc:
        pretrained_embedding = KeyedVectors.load_word2vec_format(files['GDSC_EMBEDDING_FILE'])
        callbacks.append(FixEmbed(pretrained_embedding, walks))

    if args.fix_ext:
        model_extract_celline = KeyedVectors.load_word2vec_format(files['EXPRESSION_PCA_FILE'])
        callbacks.append(FixEmbed(model_extract_celline))
    return callbacks

