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
        self.dense_layer=Dense(1,bias_initializer='zeros',activation='softmax')
        self.embedding_matrix=embedding_matrix
    def call(self,documents_batch):
        print("Shape of documents batch=",documents_batch.shape,flush=True)      
        #Multiply the batch of dense documents of size (batch_size,vocab_size) by embedding matrix of size (vocab_size,embedding_size)
        #We need the tf.cast so we have the same type of tensor in order to be able to do the multiplication
        documents_embeddings_weighted_by_frequency=tf.matmul(tf.cast(documents_batch, tf.float32),self.embedding_matrix)
#         documents_embeddings_weighted_by_frequency=tf.sparse.sparse_dense_matmul(documents_batch,self.embedding_matrix)        

        print("Shape of documents_embeddings_weighted_by_frequency batch=",documents_embeddings_weighted_by_frequency.shape,flush=True)
        
        documents_class=self.dense_layer(documents_embeddings_weighted_by_frequency)
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
    probability=[350/500]
    for i in range(1,19):
        probability.append(150/(500*18))

    while(pos+batch_size<number_of_documents_to_generate):
        #First elements for document indices and document frequency
        #We need this so we will be able to use np.append for the document_frequency and document indices
        #For documents indices we have to generate the sequence before transforming it so we can be able to have unique indices
        #inside each documents
        document_indices_batch=np.random.choice(np.arange(1, vocab_size), size=500, replace=False, p=None)
        document_indices_batch=np.array([[0,index] for index in document_indices_batch],dtype=np.int32)
        
        document_frequency_batch=np.array(np.random.choice(np.arange(1,20),size=500,replace=True,p=probability),dtype=np.int32)
#         document_frequency_batch=np.array(np.random.choice(np.arange(1,20),size=500,replace=True,p=probability),dtype=np.float32)

        label_batch=np.array(np.random.randint(2),dtype=np.float32)
        for i in range(1,batch_size):
            new_document_indices=np.random.choice(np.arange(1, vocab_size), size=500, replace=False, p=None)
            new_document_indices=np.array([[i,index] for index in new_document_indices],dtype=np.int32)
            document_indices_batch=np.append(document_indices_batch,new_document_indices,axis=0)
            #No axis has been specified for document frequency so all the values will be appended in a vector shape array
            document_frequency_batch=np.append(document_frequency_batch,np.random.choice(np.arange(1,20),size=500,replace=True,p=probability))
            label_batch=np.append(label_batch,np.random.randint(2))
        document_batch = tf.SparseTensor(indices=document_indices_batch, values=document_frequency_batch, dense_shape=[batch_size,vocab_size])
        document_batch = tf.sparse.reorder(document_batch)
        document_batch=tf.sparse.to_dense(document_batch)                                                
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
        for document_batch,label_batch in generate_training_batches(batch_size=64,number_of_documents_to_generate=100000,vocab_size=vocab_size):
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
