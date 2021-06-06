import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding,GlobalAveragePooling1D
from tensorflow.keras import Model
import argparse
import pickle
import fasttext
import numpy as np
import array as arr
import random
import time
class Example(Model):
    def __init__(self,embedding_matrix):
        super(Example,self).__init__()
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding_layer=Embedding(input_dim=self.vocab_size,output_dim=self.embedding_dim,embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),trainable=False,mask_zero=True)
        self.average_pooling=GlobalAveragePooling1D()
        self.dense_layer=Dense(1,bias_initializer='zeros',activation='softmax')
        
    def call(self,documents):
        print("Shape of documents batch=",documents.shape,flush=True)
        documents=self.embedding_layer(documents)
        print("Shape of  documents batch after embedding layer=",documents.shape,flush=True)
        documents=self.average_pooling(documents)
        print("Shape of  documents batch after average pooling layer=",documents.shape,flush=True)
        documents_class=self.dense_layer(documents)
        print("Shape of documents batch class after dense layer,document_class=",documents_class.shape,flush=True)

        return documents_class
def load_vocabulary(file_path):
    vocabulary=dict()
    with open(file_path + '/vocabulary', 'rb') as f:
        vocabulary = pickle.load(f)
    return vocabulary
def reduce_vocabulary_and_compute_fasttext_embeddings(file_path,fasttext_model_path,proportion_of_vocabulary_to_keep=0.2):
    #reducing the size of the vocabulary
    vocabulary=load_vocabulary(file_path)
    number_of_elements_to_keep=round(proportion_of_vocabulary_to_keep*len(vocabulary))
    reduced_vocabulary={token:value[1] for token,value in vocabulary.items() if value[1]<number_of_elements_to_keep}
    del(vocabulary)
    #Computing the embedding matrix
    model = fasttext.load_model(fasttext_model_path)
    vocab_size = len(reduced_vocabulary)
    embedding_matrix = np.zeros((vocab_size, 300), dtype=np.float32)
    index = 0
    for token in reduced_vocabulary:
        array_embeddings = arr.array('f', model[token])
        embedding_matrix[index] = array_embeddings
        index += 1

    return embedding_matrix,vocab_size
def generate_training_batches(batch_size,number_of_documents_to_generate,vocab_size):
    # Generating documents and labels
    pos=0
    while(pos+batch_size<number_of_documents_to_generate):
        document_batch=np.array(np.random.choice(np.arange(1, vocab_size), size=500, replace=True, p=None),dtype=np.int32)
        label_batch=np.array(np.random.randint(2),dtype=np.float32)
        for i in range(1,batch_size):
            document_batch=np.vstack((document_batch,np.array( np.random.choice(np.arange(1, vocab_size), size=500, replace=True, p=None),dtype=np.int32)))
            label_batch=np.append(label_batch,np.random.randint(2))
        document_batch=tf.keras.preprocessing.sequence.pad_sequences(document_batch, padding='post',maxlen=550)
        pos+=batch_size
        yield document_batch,label_batch


def main():
    parser = argparse.ArgumentParser()
    #Learning rate
    parser.add_argument('--lr', nargs="?", type=float)
    parser.add_argument('-f', '--fasttext_path', nargs="?", type=str)
    parser.add_argument('-v', '--vocab_path', nargs="?", type=str)
    parser.add_argument('-r', '--reduction_rate', nargs="?", type=float, default=0.2)
    args = parser.parse_args()
    start0=time.time()
    
    tf.debugging.set_log_device_placement(True)

    #Generating the embedding matrix
    embedding_matrix, vocab_size = reduce_vocabulary_and_compute_fasttext_embeddings(args.vocab_path,
                                                                                     args.fasttext_path,
                                                                                     args.reduction_rate)
    end=time.time()
    print("Time to reduce vocabulary and compute embeddings",round(end-start0),flush=True)
    start=time.time()
    #Creating an instance of the Model Class
    model=Example(embedding_matrix=embedding_matrix)
    #Creating a loss and optimizer object  BinaryCrossentropy(from_logits=True)
    loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer=tf.keras.optimizers.Adam(args.lr)
    #Specifing the metrics for evaluation for test and training . These two entities will store results over the different epochs
    #Training
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    # #Testing
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    """Training step"""
    @tf.function
    def train_step(training_documents,training_labels):
        with tf.GradientTape() as tape:
            predictions=model(training_documents)
            print("predictions shape =",predictions.shape,flush=True)
            loss=loss_function(training_labels,predictions)
        #Calculate gradients for each trainable variable
        gradients=tape.gradient(loss,model.trainable_variables)
        #Use the gradients to change the values of the trainable variables
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        #Calculate the mean of the training loss over epochs
        train_loss(loss)
        #Calculate the accuracy over epochs
        train_accuracy(training_labels,predictions)
    # @tf.function
    # def test_step(test_documents,test_labels):
    #     #predict
    #     predictions=model(test_documents,trainable=False)
    #     loss=loss_function(test_labels,predictions)
    #     test_loss(loss)
    #     test_accuracy(test_labels,predictions)

    
    for epoch in range(3):
        #resetting metrics at each epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        # test_loss.reset_states()
        # test_accuracy.reset_states()

        #Using a generator to generate batches
        for document_batch,label_batch in generate_training_batches(batch_size=64,number_of_documents_to_generate=2400000,vocab_size=vocab_size):
            with tf.device('/GPU:0'):
                train_step(document_batch,label_batch)

                print(f'Epoch {epoch + 1}, '
                    f'Loss: {train_loss.result()}, '
                    f'Accuracy: {train_accuracy.result() *100}'
                ,flush=True)
    end=time.time()
    print("Time for training =",round(end-start),flush=True)
    print("Total time = ",round(end-start0),flush=True)
if __name__ == "__main__":
    main()
